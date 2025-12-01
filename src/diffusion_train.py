import os, sys
import collections

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
from diffusion_inference import setup_env
from transformers import CLIPModel, CLIPProcessor

import wandb

# Env dependencies
import act.sim_env as act_sim_env
import act.utils as act_utils
import imageio

## Torch Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Tunable Params
NUM_EPOCHS = 100
NUM_EPISODES = 100
DATASET_PATH = Path("data").absolute()
DATASET_PATH = (
    DATASET_PATH
    if DATASET_PATH.exists()
    else "DATASET IS AS LOST AS YOU ARE - NOT FOUND IN THE GIVEN PATH"
)
FILES_OUTPUT_PATH = Path("data/diffusion_policy_models2").absolute()
# Change the name of the pth file here
VIS_WEIGHTS_FILENAME = "best_diffusion_model_e61.pth"
CHECKPOINT_PATH = FILES_OUTPUT_PATH / VIS_WEIGHTS_FILENAME
BATCH_SIZE = 1
NUM_TRAIN_TIMESTEPS = 100
VISION_FEATURE_DIM = 192
STATE_DIM = 14
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM
ACTION_DIM = 14
ACTION_HORIZON = 8
EXECUTION_HORIZON = 4
DEBUG = True
WANDB = False

if WANDB==True:
    wandb.login()
    wandb.init(entity="mrsd-smores", project="diffusIn-training")
    config = {
        "num_epochs": NUM_EPOCHS,
        "num_episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "vision_feature_dim": VISION_FEATURE_DIM,
        "state_dim": STATE_DIM,
        "observation_horizon": OBSERVATION_HORIZON,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "vision_encoder": "OpenVision-vit-tiny-patch16-384",
    }

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
        ema_power=0.75,
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

        ## Exponential Moving Average improves Training Stability
        self.ema = EMAModel(parameters=model.parameters(), power=ema_power)

        # Standard ADAM optimizer
        # Note that EMA parameters are not optimized
        self.optimizer = torch.optim.AdamW(
            params=[
                {
                    "params": model.vision_encoder.parameters(),
                    "lr": 4e-4,
                    "name": "vision_encoder",
                },  # ViT was trained with 4e-3 LR
                {
                    "params": model.noise_predictor.parameters(),
                    "lr": 1e-4,
                    "name": "noise_predictor",
                },
            ],
            lr=1e-4,
            weight_decay=1e-6,
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
        self.loss_per_ep = {key: [10.0] for key in range(NUM_EPISODES)}

    def train(self):
        """Training Loop for Diffusion Model"""
        self.ema_nets = self.model
        least_val_loss = float("inf")
        os.makedirs(FILES_OUTPUT_PATH, exist_ok=True)
        with tqdm(range(self.num_epochs), desc="Epoch") as t_global:
            # epoch loop
            for epoch_idx in t_global:
                epoch_loss = list()
                # batch loop

                with tqdm(self.train_dataloader, desc="Batch", leave=False) as t_epoch:
                    for nbatch in t_epoch:
                        end_index_obs = np.random.randint(
                            self.obs_horizon,
                            nbatch["image"].shape[1] - self.action_horizon,
                        )
                        start_index_obs = end_index_obs - self.obs_horizon
                        assert (
                            start_index_obs >= 0
                        ), "start_index_obs is negative!"  # sanity check

                        start_index_action = end_index_obs
                        end_index_action = start_index_action + self.action_horizon
                        assert (
                            end_index_action < nbatch["action"].shape[1]
                        ), "end_index_action exceeds episode length!"  # sanity check
                        # print(start_index_action, end_index_action, start_index_obs, end_index_obs)

                        # device transfer
                        # load a batch of data from expert trajectory: image, agent_pos, action
                        nimage = nbatch["image"][:, start_index_obs:end_index_obs].to(
                            self.device
                        )  # (batch_size, obs_horizon, 3, 480, 640)
                        # Save images to visualize later if needed
                        if DEBUG:
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

                        nagent_pos = nbatch["q_pos"][
                            :, start_index_obs:end_index_obs
                        ].to(
                            self.device
                        )  # Shape: (batch_size, obs_horizon, state_dim)
                        naction = nbatch["action"][
                            :, start_index_action:end_index_action
                        ].to(
                            self.device
                        )  # Shape: (batch_size, action_horizon, action_dim)
                        B = nagent_pos.shape[0]  # batch size

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

                        # logging
                        loss_cpu = loss_val.item()
                        epoch_loss.append(loss_cpu)
                        t_epoch.set_postfix(loss=loss_cpu)
                        # self.loss_per_ep[epoch_idx].append(loss_cpu)
                        if WANDB:
                            wandb.log(
                                {
                                    "train/loss": loss_cpu,
                                    "train/vision_lr": self.lr_scheduler.get_last_lr()[0],
                                    "train/noise_lr": self.lr_scheduler.get_last_lr()[1],
                                    "train/epoch": epoch_idx,
                                }
                            )

                        # optimize
                        loss_val.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior #NOTE: Understand why they are doing so
                        self.lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        self.ema.step(self.model.parameters())  # NOTE: Understand why.

                t_global.set_postfix(loss=np.mean(epoch_loss))
                if WANDB:
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
                    f"{FILES_OUTPUT_PATH}/last_diffusion_model_checkpoint.pth",
                )
                print(
                    f"Saved last_model_checkpoint at {FILES_OUTPUT_PATH}/last_diffusion_model_checkpoint.pth"
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
                        f"{FILES_OUTPUT_PATH}/best_diffusion_model_e{epoch_idx}.pth",
                    )
                    print(
                        f"Saved best_model_checkpoint at {FILES_OUTPUT_PATH}/best_diffusion_model_e{epoch_idx}.pth"
                    )
        if WANDB:
            for wandb_file in os.listdir(FILES_OUTPUT_PATH):
                wandb.save(f"{FILES_OUTPUT_PATH}/{wandb_file}")

    def eval(self, env, max_steps=500, render=False):  # default values taken from TRI example, should change
        """Evaluation Loop for Diffusion Model"""
        # |o|o|o|o|o|o|o|o|                 observations: 8
        # |p|p|p|p|p|p|p|p|                 action predictions: 8
        # | | | | | | | | |a|a|a|a|a|       actions executed: 4
        print(f"Device: {self.device}")
        pred_horizon = self.action_horizon

        # Load the model checkpoint
        self.ema_nets = self.model
        checkpoint_path = CHECKPOINT_PATH
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}")
            return
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.ema_nets.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from {checkpoint_path}")
        self.ema_nets.eval()

        # Reset environment with random peg and socket pose
        peg_pose, socket_pose = act_utils.sample_insertion_pose()
        act_sim_env.BOX_POSE[0] = np.concatenate([peg_pose, socket_pose])
        ts = env.reset()
        # TODO: Check if "angle" camera is the right one since training is done with "top" camera
        onscreen_cam = "angle"

        # Init obs deque with initial observations
        # obs keys:'qpos', 'qvel', 'env_state', 'images'
        # keys in obs['images']: 'top', 'angle', 'vis'. Image shape: (480, 640, 3)
        obs = ts.observation
        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        if render:
            imgs = [env._physics.render(height=480, width=640, camera_id=onscreen_cam)]
        else:
            imgs = []
        rewards = []
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval SimInsertion") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x["images"]["top"] for x in obs_deque])
                agent_poses = np.stack([x["qpos"] for x in obs_deque])

                if DEBUG:
                    # Visualize current images in obs_deque
                    for i, x in enumerate(obs_deque):
                        os.makedirs("data/local_debug", exist_ok=True)
                        Image.fromarray(x["images"]["top"]).save(f"data/local_debug/obs_deque_current_{i}.png")
                    

                # normalize observation
                nagent_poses = (agent_poses - self.stats["qpos_mean"]) / self.stats[
                    "qpos_std"
                ]
                # normalize images to [0,1]
                nimages = images / 255.0

                # Reshape images from (obs_horizon, h, w, c) to (1, obs_horizon, c, h, w)
                nimages = nimages.transpose(0, 3, 1, 2)
                nimages = np.expand_dims(nimages, axis=0)

                # Reshape agent poses from (obs_horizon, state_dim) to (1, obs_horizon, state_dim)
                nagent_poses = np.expand_dims(nagent_poses, axis=0)

                # device transfer
                nimages = torch.from_numpy(nimages).to(
                    self.device, dtype=torch.float32
                )  # (1, obs_horizon, 3, 480, 640)
                nagent_poses = torch.from_numpy(nagent_poses).to(
                    self.device, dtype=torch.float32
                )  # (1, obs_horizon, state_dim(14))

                # infer action
                with torch.no_grad():
                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, self.action_dim), device=self.device
                    )
                    naction = noisy_action

                    if DEBUG:
                        # Visualize the initial noisy action
                        self.visualize_actions(naction[0].cpu().numpy(), save_path="data/local_debug/initial_noisy_action.png")
                    
                    # init scheduler
                    # NOTE: TRI example uses the same scheduler for training and inference.
                    # may consider using different schedulers, eg DDIM, or reference https://arxiv.org/pdf/2301.10677
                    self.noise_scheduler.set_timesteps(self.diffusion_timesteps)

                    # performs single diffusion sample from pure noise to denoised action sequence
                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = self.ema_nets(nimages, nagent_poses, naction, k)

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred, timestep=k, sample=naction
                        ).prev_sample

                # unnormalize action
                naction = (
                    naction.detach().to("cpu").numpy()
                )  # (B, pred_horizon, action_dim)
                naction = naction[0]  # (pred_horizon, action_dim)
                action_pred = (
                    naction * self.stats["action_std"] + self.stats["action_mean"]
                )

                if DEBUG:
                    # Visualize the initial noisy action
                    self.visualize_actions(action_pred, save_path="data/local_debug/denoised_action.png")
                
                # only take execution_horizon number of actions
                action = action_pred[
                    : self.execution_horizon, :
                ]  # (execution_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    ts = env.step(action[i])
                    obs = ts.observation
                    reward = ts.reward
                    # obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)

                    if render:
                        imgs.append(
                            env._physics.render(
                                height=480, width=640, camera_id=onscreen_cam
                            )
                        )

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)

                    # TODO: Find a way to detect completion of the task and quit early
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print("Score: ", max(rewards))

        if render:
            # save vis as gif
            os.makedirs("data/diffusion_eval_vis", exist_ok=True)
            imageio.mimsave(f"data/diffusion_eval_vis/{VIS_WEIGHTS_FILENAME[:-4]}.gif", imgs, fps=33)

    def visualize_actions(self, actions, save_path="action_visualization.png"):
        """Visualize action sequences as line plots for each action dimension"""
        import matplotlib.pyplot as plt

        # Create side-by-side subplots: first 7 action dims on the left, remaining on the right
        action_horizon_local, action_dim_local = actions.shape
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

        t = np.arange(action_horizon_local)
        left_dims = range(min(7, action_dim_local))
        right_dims = range(7, action_dim_local)

        for d in left_dims:
            axes[0].plot(t, actions[:, d], label=f"Action Dim {d}")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Action Value")
        axes[0].set_title("Left arm")
        axes[0].legend()
        axes[0].grid(True)

        for d in right_dims:
            axes[1].plot(t, actions[:, d], label=f"Action Dim {d}")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Action Value")
        axes[1].set_title("Right arm")
        axes[1].legend()
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

        # action_horizon, action_dim = actions.shape
        # plt.figure(figsize=(15, 5))
        # for dim in range(action_dim):
        #     plt.plot(range(action_horizon), actions[:, dim], label=f"Action Dim {dim}")
        # plt.xlabel("Timestep")
        # plt.ylabel("Action Value")
        # plt.title("Action Sequence Visualization")
        # plt.legend()
        # plt.grid()
        # plt.savefig(save_path)
        # plt.close()


if __name__ == "__main__":
    vision_encoder = OpenVisionEncoder().to(device=DEVICE)
    model = DiffusionModel(
        state_dim=STATE_DIM,
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBSERVATION_HORIZON,
        vision_encoder=vision_encoder,
        device=DEVICE,
    )
    ## Dataset and Dataloader
    # NOTE: Cannot pass num_episodes = 1 because train/val split fails
    train_dataloader, val_dataloader, norm_dataset_stats, is_sim = load_data(
        DATASET_PATH, NUM_EPISODES, ["top"], BATCH_SIZE, 1
    )
    trainer = TrainDiffusIn(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stats=norm_dataset_stats,
        action_dim=ACTION_DIM,
        obs_horizon=OBSERVATION_HORIZON,
        action_horizon=ACTION_HORIZON,
        execution_horizon=EXECUTION_HORIZON,
        device=DEVICE,
        diffusion_timesteps=NUM_TRAIN_TIMESTEPS,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train()

    env = act_sim_env.make_sim_env("sim_insertion")
    trainer.eval(env, max_steps=125, render=True)
