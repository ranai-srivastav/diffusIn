import os, sys
from typing import Dict, Callable, List
import argparse
import yaml
import time

from pathlib import Path

sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
        str(Path("include/OpenVision").resolve()),
        str(Path("include/diffusion_policy").resolve()),
    ]
)
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from einops import reduce

## Encoder Dependencies
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import adjust_brightness
from PIL import Image

# Diffusion Dependencies
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from OpenVision.src.convert_upload.open_clip.factory import (
    create_vision_encoder_and_transforms,
)

from diffusion_layers import (
    Conv1dBlock,
    Upsample1d,
    Downsample1d,
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    SinusoidalPosEmb,
)
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_dataloaders import load_data, load_chunked_data
from transformers import CLIPModel, CLIPProcessor
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from utils import dict_apply

import wandb
import gc

def init_wandb(entity="mrsd-smores", project="diffusIn-training", config_dict=None):
    """Initialize WandB with configuration"""
    wandb.login()
    wandb.init(entity=entity, project=project)
    if config_dict:
        wandb.config.update(config_dict)

def save_config(config_dict, output_path):
    """Save configuration to YAML file"""
    os.makedirs(output_path, exist_ok=True)
    config_file = output_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved configuration to {config_file}")

# Action space:      [left_arm_qpos (6),             # absolute joint position
#                         left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
#                         right_arm_qpos (6),            # absolute joint position
#                         right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)


## Vision Encoder
class VisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = None

    def preprocess(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL.Image")

    def forward(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a Torch.Tensor")


class OpenVisionEncoder(VisionEncoder):
    def __init__(self):
        super().__init__()
        hf_repo = "UCSC-VLAA/openvision-vit-tiny-patch16-384"

        self.vision_encoder = create_vision_encoder_and_transforms(
            model_name=f"hf-hub:{hf_repo}"
        )

    def preprocess(self, image):
        # image = image.convert("RGB")
        tensor_conv = Compose(
            [
                Resize((384, 384)),
                # ToTensor(),
                # TODO: Check how normalize was supposed to be used, the range of inputs gers changed to [-1.8,2.2] after this step
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return tensor_conv(image)

    def forward(self, image: torch.Tensor):
        return self.vision_encoder(image)  # Adding batch dimension


class CLIPEncoder(VisionEncoder):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(self, raw_image: Image.Image):
        raw_image = raw_image.convert("RGB")
        inputs = self.processor(images=raw_image, return_tensors="pt", padding=True)
        return inputs

    def forward(self, image):
        vision_outputs = self.model.vision_model(**image)
        image_embeds = vision_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)


## Diffusion Model
class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        obs_dim,
        action_dim,
        obs_horizon,
        vision_encoder: VisionEncoder,
        device,
        multiview=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.device = device
        self.multiview = multiview

        self.vision_encoder = vision_encoder

        # TODO: What order does Conv!D exepect it in? Is it (B, C, L) or (B, L, C)?
        if multiview:
            self.obs_feature_dim = obs_dim * obs_horizon * 3  # 3 views
        else:
            self.obs_feature_dim = obs_dim * obs_horizon

        self.noise_predictor = ConditionalUnet1D(
            action_dim=action_dim, global_cond_dim=self.obs_feature_dim
        )

        self.to(device=device)

    def forward(self, image, pos, noisy_actions, timesteps):

        # Generating vision embedding
        image_preproc = self.vision_encoder.preprocess(
            image
        )  # image preproc shape (B, obs_horizon, 3, 384, 384)
        image_features = self.vision_encoder(
            image_preproc.flatten(end_dim=1)
        )  # Shape of image features: B * obs_horizon, D
        image_features = image_features.reshape(
            *image_preproc.shape[:2], -1
        )  # Shape of image features flattened: B, obs_horizon, D
        # vision embedding shape (B, obs_horizon, D)

        # concatenate vision feature and agent positions
        # TODO:Agent positions need to be raw inputs or embeddings?
        if self.multiview:
            pos_repeated = pos.repeat(1, 3, 1)  # B, obs_horizon, state_dim -> B, obs_horizon*3, state_dim
            obs_features = torch.cat([image_features, pos_repeated], dim=-1)  # D -> D + state_dim = obs_dim
            obs_cond = obs_features.flatten(start_dim=1)
        else:
            obs_features = torch.cat([image_features, pos], dim=-1)  # D -> D + state_dim = obs_dim
            obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # predict the noise residual
        noise_pred = self.noise_predictor(noisy_actions, timesteps, global_cond=obs_cond)

        return noise_pred


## Trainer Class
class TrainDiffusIn:
    def __init__(
        self,
        model: DiffusionModel,
        pred_horizon,
        execution_horizon,
        obs_horizon,
        # dataloaders
        train_dataloader,
        val_dataloader,
        stats,
        # training params
        diffusion_timesteps,
        num_epochs,
        ema_power,
        vision_lr,
        noise_predictor_lr,
        weight_decay,
        # logging params
        track_wandb,
        save_every,
        files_output_path,
        debug=False,
        multiview=False,
    ):
        # model params
        self.model=model
        self.pred_horizon=pred_horizon
        self.execution_horizon=execution_horizon
        self.obs_horizon=obs_horizon
        self.device=model.device

        # dataloaders
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.stats=stats

        # training params
        self.diffusion_timesteps=diffusion_timesteps
        self.num_epochs=num_epochs
        self.ema_power=ema_power
        self.vision_lr=vision_lr
        self.noise_predictor_lr=noise_predictor_lr
        self.weight_decay=weight_decay
        self.multiview=multiview

        # logging params
        self.track_wandb=track_wandb
        self.save_every=save_every
        self.files_output_path=files_output_path
        self.debug=debug

        ## Exponential Moving Average improves Training Stability
        self.ema = EMAModel(parameters=model.parameters(), power=ema_power)

        # Standard ADAM optimizer
        # Note that EMA parameters are not optimized
        self.optimizer = torch.optim.AdamW(
            params=[
                {
                    "params": model.vision_encoder.parameters(),
                    "lr": self.vision_lr,
                    "name": "vision_encoder",
                },  # ViT was trained with 4e-3 LR
                {
                    "params": model.noise_predictor.parameters(),
                    "lr": self.noise_predictor_lr,
                    "name": "noise_predictor",
                },
            ],
            lr=self.noise_predictor_lr,
            weight_decay=self.weight_decay,
        )

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader) * self.num_epochs,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            # the choice of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=model.action_dim,
            obs_dim=0,
            max_n_obs_steps=obs_horizon,
            fix_obs_steps=True,
            action_visible=False
        )

        # L2 loss
        self.loss_fn = torch.nn.MSELoss(reduction="none")

        self.ema_nets = self.model

    def training_step(self, batch):
        nimage = batch["image"]
        nagent_pos = batch["q_pos"]
        naction = batch["action"] # B x pred_horizon x action_dim
        B = naction.shape[0]

        # generate global conditioning vector
        # Vision encoder input needs to be B, obs_horizon, 3, 384, 384
        if self.multiview:
            # B, [top1, top2, angle1, angle2, vis1, vis2], 3, 384, 384
            this_nimage = nimage[:, :self.obs_horizon*3,...]
        else:
            this_nimage = nimage[:, :self.obs_horizon,...]
        this_nagent_pos = nagent_pos[:, :self.obs_horizon,...]

        # generate conditioning mask
        condition_mask = self.mask_generator(naction.shape).to(self.device)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=self.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(B,),
            device=self.device,
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            naction, noise, timesteps
        )

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning mask
        noisy_actions[condition_mask] = naction[condition_mask]

        # predict the noise
        noise_pred = self.model(
            this_nimage, this_nagent_pos, noisy_actions, timesteps
        )

        # calculate loss
        loss = self.loss_fn(noise_pred, noise)
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    def train(self):
        """Training Loop for Diffusion Model"""
        gc.collect()
        torch.cuda.empty_cache()


        print(f"Training for {self.num_epochs} epochs with batch size {self.train_dataloader.batch_size}")
        with tqdm(range(self.num_epochs), desc="Epoch") as t_global:
            # epoch loop
            for epoch_idx in t_global:
                epoch_loss = list()
                # batch loop
                with tqdm(self.train_dataloader, desc="Batch", leave=False) as t_epoch:
                    for nbatch in t_epoch:
                        # move batch to device
                        nbatch = dict_apply(nbatch, lambda x: x.to(self.device, non_blocking=True))
                        loss = self.training_step(nbatch)
                        
                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # step lr scheduler every batch
                        self.lr_scheduler.step()

                        self.ema.step(self.model.parameters())
                        
                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        t_epoch.set_postfix(loss=loss_cpu)
                        if self.track_wandb:
                            wandb.log(
                                {
                                    "train/loss": loss_cpu,
                                    "train/vision_lr": self.lr_scheduler.get_last_lr()[0],
                                    "train/noise_lr": self.lr_scheduler.get_last_lr()[1],
                                    "train/epoch": epoch_idx,
                                }
                            )

                t_global.set_postfix(loss=np.mean(epoch_loss))
                if self.track_wandb:
                    wandb.log({"train/epoch_loss": np.mean(epoch_loss)})

                
                if (epoch_idx+1) % self.save_every == 0:
                    # Save this model
                    self.ema.copy_to(self.ema_nets.parameters())
                    torch.save(
                        {
                            "model_state_dict": self.ema_nets.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                        },
                        f"{self.files_output_path}/diffusion_model_checkpoint_e{epoch_idx}.pth",
                    )
                    print(
                        f"Saved last_model_checkpoint at {self.files_output_path}/diffusion_model_checkpoint_e{epoch_idx}.pth"
                    )
                    
                if self.track_wandb:
                    for wandb_file in os.listdir(self.files_output_path):
                        wandb.save(f"{self.files_output_path}/{wandb_file}")

        # Save last model checkpoint
        self.ema.copy_to(self.ema_nets.parameters())
        torch.save(
            {
                "model_state_dict": self.ema_nets.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            },
            f"{self.files_output_path}/last_diffusion_model_checkpoint.pth",
        )
        print(
            f"Saved last_model_checkpoint at {self.files_output_path}/last_diffusion_model_checkpoint.pth"
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--num-episodes", type=int, default=15,
                       help="Number of episodes to load from dataset (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--num-train-timesteps", type=int, default=100,
                       help="Number of diffusion timesteps (default: 100)")
    parser.add_argument("--ema-power", type=float, default=0.75,
                       help="EMA power (default: 0.75)")
    parser.add_argument("--vision-lr", type=float, default=4e-4,
                       help="Learning rate for vision encoder (default: 4e-4)")
    parser.add_argument("--noise-predictor-lr", type=float, default=1e-4,
                       help="Learning rate for noise predictor (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay for optimizer (default: 1e-6)")
    parser.add_argument("--multiview", action="store_true",
                       help="Use multiview images (default: False)")
    
    # Model arguments
    parser.add_argument("--state-dim", type=int, default=14,
                       help="State dimension (default: 14)")
    parser.add_argument("--action-dim", type=int, default=14,
                       help="Action dimension (default: 14)")
    parser.add_argument("--pred-horizon", type=int, default=8,
                       help="Prediction horizon (default: 8)")
    parser.add_argument("--execution-horizon", type=int, default=4,
                       help="Execution horizon (default: 4)")
    parser.add_argument("--obs-horizon", type=int, default=2,
                       help="Number of observation steps to condition on (default: 2)")
    parser.add_argument("--vision-encoder", type=str, default="OpenVision-vit-tiny",
                       choices=["OpenVision-vit-tiny", "CLIP"],
                       help="Vision encoder type (default: OpenVision-vit-tiny)")
    
    # Path arguments
    parser.add_argument("--dataset-path", type=str, default="data_recorded",
                       help="Path to dataset directory (default: data)")
    
    # WandB arguments
    parser.add_argument("--no-track", action="store_true",
                       help="Do NOT track on WandB")
    parser.add_argument("--wandb-entity", type=str, default="mrsd-smores",
                       help="WandB entity name (default: mrsd-smores)")
    parser.add_argument("--wandb-project", type=str, default="diffusIn-training",
                       help="WandB project name (default: diffusIn-training)")
    
    # Other arguments
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save model every N epochs (default: 10)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode (default: False)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
                       help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    # Set up dirs
    dataset_path = Path(args.dataset_path).absolute()
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
    files_output_path = Path(dataset_path / f"diffusion_policy_models_{time.strftime('%Y%m%d_%H%M%S')}").absolute()
    
    DEBUG = args.debug
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Initialize vision encoder
    if args.vision_encoder == "CLIP":
        vision_encoder = CLIPEncoder().to(device=device)
        args.vision_feature_dim = 512
    else:
        vision_encoder = OpenVisionEncoder().to(device=device)
        args.vision_feature_dim = 192

    # Compute observation_dim from vision_feature_dim + state_dim
    args.observation_dim = args.vision_feature_dim + args.state_dim

    # Prepare base configuration (we will save after loading dataset stats)
    config_dict = {
        "training": {
            "num_epochs": args.num_epochs,
            "num_episodes": args.num_episodes,
            "batch_size": args.batch_size,
            "num_train_timesteps": args.num_train_timesteps,
            "ema_power": args.ema_power,
            "vision_lr": args.vision_lr,
            "noise_predictor_lr": args.noise_predictor_lr,
            "weight_decay": args.weight_decay,
            "multiview": args.multiview,
        },
        "model": {
            "vision_feature_dim": args.vision_feature_dim,
            "state_dim": args.state_dim,
            "observation_horizon": args.obs_horizon,
            "observation_dim": args.observation_dim,
            "action_dim": args.action_dim,
            "pred_horizon": args.pred_horizon,
            "execution_horizon": args.execution_horizon,
            "vision_encoder": args.vision_encoder,
        }
    }
    print(f"Using device: {device}")
    # Initialize model
    model = DiffusionModel(
        state_dim=args.state_dim,
        obs_dim=args.observation_dim,
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        vision_encoder=vision_encoder,
        device=device,
    )

    model.to(device=device)
    print("Initialized Diffusion Model:")
    # print(model)
    
    # Dataset and Dataloader
    # NOTE: Cannot pass num_episodes = 1 because train/val split fails
    # Load chunked sequence dataset
    train_dataloader, val_dataloader, norm_dataset_stats, is_sim = load_chunked_data(
        dataset_path, args.num_episodes, ["top"], args.batch_size, 1, args.pred_horizon, args.obs_horizon, args.multiview
    )

    print(f"Dataset stats: {norm_dataset_stats}")
    print(f"Files output path: {files_output_path}")

    # Add dataset_stats to config and save
    dataset_stats_serialized = {
        "action_mean": np.asarray(norm_dataset_stats["action_mean"]).tolist(),
        "action_std": np.asarray(norm_dataset_stats["action_std"]).tolist(),
        "qpos_mean": np.asarray(norm_dataset_stats["qpos_mean"]).tolist(),
        "qpos_std": np.asarray(norm_dataset_stats["qpos_std"]).tolist(),
        "qvel_mean": np.asarray(norm_dataset_stats["qvel_mean"]).tolist(),
        "qvel_std": np.asarray(norm_dataset_stats["qvel_std"]).tolist(),
    }

    config_dict["dataset_stats"] = dataset_stats_serialized
    save_config(config_dict, files_output_path)

    # Initialize WandB if tracking is enabled
    if not args.no_track:
        wandb_config = {
            "num_epochs": args.num_epochs,
            "num_episodes": args.num_episodes,
            "batch_size": args.batch_size,
            "num_train_timesteps": args.num_train_timesteps,
            "vision_feature_dim": args.vision_feature_dim,
            "state_dim": args.state_dim,
            "observation_horizon": args.obs_horizon,
            "observation_dim": args.observation_dim,
            "action_dim": args.action_dim,
            "pred_horizon": args.pred_horizon,
            "execution_horizon": args.execution_horizon,
            "vision_encoder": args.vision_encoder,
            "vision_lr": args.vision_lr,
            "noise_predictor_lr": args.noise_predictor_lr,
            "weight_decay": args.weight_decay,
            "multiview": args.multiview,
            "device": str(device),
        }
        init_wandb(entity=args.wandb_entity, project=args.wandb_project, config_dict=wandb_config)
    
    # Initialize trainer
    trainer = TrainDiffusIn(
        # model params
        model=model,
        pred_horizon=args.pred_horizon,
        execution_horizon=args.execution_horizon,
        obs_horizon=args.obs_horizon,
        # dataloaders
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stats=norm_dataset_stats,
        # training params
        diffusion_timesteps=args.num_train_timesteps,
        num_epochs=args.num_epochs,
        ema_power=args.ema_power,
        vision_lr=args.vision_lr,
        noise_predictor_lr=args.noise_predictor_lr,
        weight_decay=args.weight_decay,
        multiview=args.multiview,
        # logging params
        track_wandb=not args.no_track,
        save_every=args.save_every,
        files_output_path=files_output_path,
        debug=DEBUG,
    )
    trainer.train()
