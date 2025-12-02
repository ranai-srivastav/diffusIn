import os, sys
import argparse
import yaml
import time

from pathlib import Path

sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
        str(Path("include/OpenVision").resolve()),
    ]
)
import numpy as np
from tqdm import tqdm
from torchsummary import summary

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
from diffusion_dataloaders import load_data
from transformers import CLIPModel, CLIPProcessor

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

    def preprocess(self, image: Image.Image):
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
        vision_encoder,
        device,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.device = device

        self.vision_encoder = vision_encoder

        # TODO: What order does Conv!D exepect it in? Is it (B, C, L) or (B, L, C)?
        self.noise_predictor = ConditionalUnet1D(
            action_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
        )

        self.to(device)

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
        train_dataloader,
        val_dataloader,
        stats,
        action_dim,
        obs_horizon,
        action_horizon,
        execution_horizon,
        device,
        diffusion_timesteps,
        num_epochs,
        vision_lr,
        noise_predictor_lr,
        weight_decay,
        track_wandb=True,
        ema_power=0.75,
        files_output_path=None,
        debug=False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.stats = stats
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.execution_horizon = execution_horizon
        self.device = device
        self.num_epochs = num_epochs
        self.diffusion_timesteps = diffusion_timesteps
        self.track_wandb = track_wandb
        self.files_output_path = files_output_path
        self.debug = debug
        self.vision_lr = vision_lr
        self.noise_predictor_lr = noise_predictor_lr
        self.weight_decay = weight_decay

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

        # L2 loss
        self.loss_fn = torch.nn.MSELoss()

        self.ema_nets = self.model

    def train(self):
        """Training Loop for Diffusion Model"""
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Training for {self.num_epochs} epochs with batch size {self.train_dataloader.batch_size}")
        least_val_loss = float("inf")
        with tqdm(range(self.num_epochs), desc="Epoch") as t_global:
            # epoch loop
            for epoch_idx in t_global:
                epoch_loss = list()
                # batch loop

                with tqdm(self.train_dataloader, desc="Batch", leave=False) as t_epoch:
                    for nbatch in t_epoch:
                        # Find valid start and end indices for observation and action sequences from the ORIGINAL unpadded data
                        # Using different start and end indices for each sample in the batch
                        # so that the model does not overfit to fixed positions in the sequences
                        start_index_action = np.random.randint(self.obs_horizon, nbatch["lengths"] - self.action_horizon)      
                        start_index_obs = start_index_action - self.obs_horizon
                        
                        #TODO: Can be moved into dataloader for efficiency
                        indices_arr = []
                        B = nbatch["image"].shape[0]
                        lower = self.obs_horizon
                        upper = (torch.min(nbatch["lengths"]) - self.action_horizon).item()
                        indices = np.array(range(lower, upper, self.action_horizon))
                        stacked_indices = np.tile(indices, (B, 1))
                        for i in range(stacked_indices.shape[0]):
                            np.random.shuffle(stacked_indices[i])
                        
                        batch_loss = 0.0
                        with tqdm(range(stacked_indices.shape[1]), desc="Sequence", leave=False) as t_seq:
                            for i in range(stacked_indices.shape[1]):
                                # Using numpy advanced indexing to select variable length sequences from the padded batch
                                start_index_obs = stacked_indices[:, i] - self.obs_horizon
                                start_index_action = stacked_indices[:, i]
                                # Create indices between start and end indices for each element in the batch
                                obs_idx = start_index_obs[:, None] + np.arange(self.obs_horizon)[None, :]
                                action_idx = start_index_action[:, None] + np.arange(self.action_horizon)[None, :]
                                
                                # Splice out the relevant sequences using the above indices
                                nimage = nbatch["image"][np.arange(B)[:, None], obs_idx].to(self.device)
                                nagent_pos = nbatch["q_pos"][np.arange(B)[:, None], obs_idx].to(self.device)
                                naction = nbatch["action"][np.arange(B)[:, None], action_idx].to(self.device)

                                # Save images to visualize later if needed
                                if self.debug:
                                    for img_idx in range(nbatch["image"].shape[1]):
                                        img = (
                                            nbatch["image"][0, img_idx, :, :, :]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                        img = (img * 255).astype(np.uint8)  # 3 x 480 x 640
                                        img_pil = Image.fromarray(
                                            np.transpose(img, (1, 2, 0))
                                        )  # H x W x 3
                                        os.makedirs(
                                            "data/diffusion_training_vis", exist_ok=True
                                        )
                                        img_pil.save(
                                            f"data/diffusion_training_vis/epoch{epoch_idx}_img{img_idx}.png"
                                        )

                                # sample noise to add to actions
                                noise = torch.randn(
                                    naction.shape, device=self.device
                                )  # randn is random normal
                                # sample a diffusion iteration for each data point
                                # NOTE: What is this step doing? <- samples a random timestep to add noise up to.
                                # this teaches the model to denoise from any timestep. more efficient than training on all timesteps,
                                # because we know the mathematical relationship between any timestep in the diffusion process
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

                                # predict the noise
                                noise_pred = self.model(
                                    nimage, nagent_pos, noisy_actions, timesteps
                                )

                                # calculate loss
                                loss_val = self.loss_fn(noise_pred, noise)
                                
                                # Calculate gradients for every batch instead of every sequence
                                # Average gradients to avoid exploding gradients
                                (loss_val / stacked_indices.shape[1]).backward()
                                # Add mean of every sequence loss to get batch loss
                                batch_loss += (loss_val.detach() / stacked_indices.shape[1])
                        
                        # optimize
                        # this is different from standard pytorch behavior #NOTE: Understand why they are doing so
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # step lr scheduler every batch
                        self.lr_scheduler.step()

                        self.ema.step(self.model.parameters())
                        
                        # logging
                        loss_cpu = batch_loss.item()
                        epoch_loss.append(loss_cpu)
                        t_epoch.set_postfix(loss=loss_cpu)
                        # self.loss_per_ep[epoch_idx].append(loss_cpu)
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

                # Per trajectory loss over time
                # for traj_idx in range(NUM_EPISODES):
                #     wandb.log({f"traj/traj_{traj_idx}": self.loss_per_ep[traj_idx][-1]})

                # Save this model
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

                # Save this as the best model if validation loss improves
                if least_val_loss > loss_cpu:
                    least_val_loss = loss_cpu
                    torch.save(
                        {
                            "model_state_dict": self.ema_nets.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                        },
                        f"{self.files_output_path}/best_diffusion_model_e{epoch_idx}.pth",
                    )
                    print(
                        f"Saved best_model_checkpoint at {self.files_output_path}/best_diffusion_model_e{epoch_idx}.pth"
                    )
                    
                if self.track_wandb:
                    for wandb_file in os.listdir(self.files_output_path):
                        wandb.save(f"{self.files_output_path}/{wandb_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of episodes in dataset (default: 100)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for training (default: 1)")
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
    
    # Model arguments
    parser.add_argument("--state-dim", type=int, default=14,
                       help="State dimension (default: 14)")
    parser.add_argument("--obs-horizon", type=int, default=8,
                       help="Observation horizon (default: 8)")
    parser.add_argument("--action-dim", type=int, default=14,
                       help="Action dimension (default: 14)")
    parser.add_argument("--action-horizon", type=int, default=8,
                       help="Action horizon (default: 8)")
    parser.add_argument("--execution-horizon", type=int, default=4,
                       help="Execution horizon (default: 4)")
    parser.add_argument("--vision-encoder", type=str, default="OpenVision-vit-tiny",
                       choices=["OpenVision-vit-tiny", "CLIP"],
                       help="Vision encoder type (default: OpenVision-vit-tiny)")
    
    # Path arguments
    parser.add_argument("--dataset-path", type=str, default="data",
                       help="Path to dataset directory (default: data)")
    
    # WandB arguments
    parser.add_argument("--no-track", action="store_true",
                       help="Do NOT track on WandB")
    parser.add_argument("--wandb-entity", type=str, default="mrsd-smores",
                       help="WandB entity name (default: mrsd-smores)")
    parser.add_argument("--wandb-project", type=str, default="diffusIn-training",
                       help="WandB project name (default: diffusIn-training)")
    
    # Other arguments
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode (default: False)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
                       help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
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
    
    # Save configuration to YAML file in output directory
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
        },
        "model": {
            "vision_feature_dim": args.vision_feature_dim,
            "state_dim": args.state_dim,
            "observation_horizon": args.obs_horizon,
            "observation_dim": args.observation_dim,
            "action_dim": args.action_dim,
            "action_horizon": args.action_horizon,
            "execution_horizon": args.execution_horizon,
            "vision_encoder": args.vision_encoder,
        }
    }
    save_config(config_dict, files_output_path)
    
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
    
    # Dataset and Dataloader
    # NOTE: Cannot pass num_episodes = 1 because train/val split fails
    train_dataloader, val_dataloader, norm_dataset_stats, is_sim = load_data(
        dataset_path, args.num_episodes, ["top"], args.batch_size, 1
    )
    
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
            "action_horizon": args.action_horizon,
            "execution_horizon": args.execution_horizon,
            "vision_encoder": args.vision_encoder,
            "vision_lr": args.vision_lr,
            "noise_predictor_lr": args.noise_predictor_lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
        }
        init_wandb(entity=args.wandb_entity, project=args.wandb_project, config_dict=wandb_config)
    
    # Initialize trainer
    trainer = TrainDiffusIn(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stats=norm_dataset_stats,
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        execution_horizon=args.execution_horizon,
        device=device,
        diffusion_timesteps=args.num_train_timesteps,
        num_epochs=args.num_epochs,
        track_wandb=not args.no_track,
        ema_power=args.ema_power,
        files_output_path=files_output_path,
        debug=DEBUG,
        vision_lr=args.vision_lr,
        noise_predictor_lr=args.noise_predictor_lr,
        weight_decay=args.weight_decay,
    )
    trainer.train()
