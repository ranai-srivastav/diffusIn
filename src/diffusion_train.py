import os, sys

from pathlib import Path
sys.path.extend(
    [str(Path("include").resolve()),
     str(Path("include/act").resolve()),
     str(Path("include/OpenVision").resolve())]
)
import numpy as np
from tqdm import tqdm
from torchsummaryX import summary

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


## Torch Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Tunable Params
NUM_EPOCHS = 100
NUM_EPISODES = 1
DATASET_PATH = Path("data").absolute()
DATASET_PATH = (
    DATASET_PATH
    if DATASET_PATH.exists()
    else "DATASET IS AS LOST AS YOU ARE - NOT FOUND IN THE GIVEN PATH"
)
BATCH_SIZE = 16
NUM_TRAIN_TIMESTEPS = 100
VISION_FEATURE_DIM = 512
STATE_DIM = 14
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM
ACTION_DIM = 16

### Action space:      [left_arm_pose (7),             # position and quaternion for end effector
###                         left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
###                         right_arm_pose (7),            # position and quaternion for end effector
###                         right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)
# NOTE: action space doesn't include joint angles, only ee?


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
        image = image.convert("RGB")
        tensor_conv = Compose(
            [
                Resize((384, 384)),
                ToTensor(),
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return tensor_conv(image)

    def forward(self, image: torch.Tensor):
        return self.vision_encoder(torch.unsqueeze(image, 0))  # Adding batch dimension


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
            obs_horizon,
            action_dim,
            vision_encoder,
            device,):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.device = device

        self.vision_encoder = vision_encoder

        self.noise_predictor = ConditionalUnet1D(
            input_dim=state_dim, 
            global_cond_dim=obs_dim * obs_horizon)
        
        self.to(device)
        
    def forward(self, image, pos, noisy_actions, timesteps):
        
        # Generating vision embedding
        image_preproc = self.vision_encoder.preprocess(image) # image preproc shape (B, obs_horizon, 3, 384, 384)
        image_features = self.vision_encoder(image_preproc.flatten(end_dim=1))
        image_features = image_features.reshape(*image_preproc.shape[:2], -1)
        # vision embedding shape (B, obs_horizon, D)

        #TODO image embeddings are currently of dim 192 but the Conv1D expects `input_dim`. Need to align
        
        # concatenate vision feature and agent positions
        obs_features = torch.cat([image_features, pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # predict the noise residual
        noise_pred = self.noise_predictor(
            noisy_actions, timesteps, global_cond=obs_cond
        )
        
        return noise_pred

## Trainer Class
class TrainDiffusIn:
    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader,
            stats,
            action_dim,
            obs_horizon,
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
        self.device = device
        self.num_epochs = num_epochs
        self.diffusion_timesteps = diffusion_timesteps

        ## Exponential Moving Average improves Training Stability
        self.ema = EMAModel(parameters=model.parameters(), power=ema_power)

        # Standard ADAM optimizer
        # Note that EMA parameters are not optimized
        self.optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

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
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        # L2 loss
        self.loss_fn = torch.nn.MSELoss()

    def train(self):
        """ Training Loop for Diffusion Model """
        with tqdm(range(self.num_epochs), desc="Epoch") as t_global:
            # epoch loop
            for epoch_idx in t_global:
                epoch_loss = list()
                # batch loop
                with tqdm(self.train_dataloader, desc="Batch", leave=False) as t_epoch:
                    for nbatch in t_epoch:

                        # device transfer
                        # load a batch of data from expert trajectory: image, agent_pos, action
                        nimage = nbatch["image"][:, :self.obs_horizon].to(self.device)
                        nagent_pos = nbatch["agent_pos"][:, :self.obs_horizon].to(self.device)
                        naction = nbatch["action"].to(self.device)
                        B = nagent_pos.shape[0] # batch size

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device) #randn is random normal
                        # sample a diffusion iteration for each data point 
                        #NOTE: What is this step doing? <- samples a random timestep to add noise up to. 
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
                        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                        # predict the noise
                        noise_pred = self.model(nimage, nagent_pos, noisy_actions, timesteps)

                        # calculate loss
                        loss_val = self.loss_fn(noise_pred, noise)

                        # optimize
                        loss_val.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior #NOTE: Understand why they are doing so
                        self.lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        self.ema.step(self.model.parameters()) #NOTE: Understand why.

                        # logging
                        loss_cpu = loss_val.item()
                        epoch_loss.append(loss_cpu)
                        t_epoch.set_postfix(loss=loss_cpu)
                t_global.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        self.ema_nets = self.model
        self.ema.copy_to(self.ema_nets.parameters())

        # TODO Save Model Checkpoint. This is GROSSLY INCORRECT <- why?
        os.makedirs("data/diffusion_policy_models", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.ema_nets.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            },
            "data/diffusion_policy_models/diffusion_model_checkpoint.pth",
        )

    def eval(self, env, pred_horizon=16, action_horizon=8, max_steps=200): # default values taken from TRI example, should change
        """ Evaluation Loop for Diffusion Model """
        #|o|o|                             observations: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        self.ema_nets.load_state_dict(torch.load(
            "data/diffusion_policy_models/diffusion_model_checkpoint.pth"
        )["model_state_dict"], map_location=self.device)

        self.ema_nets.eval()

        # TODO dataloader stuff to set up obs_deque

        with tqdm(total=max_steps, desc="Eval SimInsertion") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x["image"] for x in obs_deque])
                agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

                # normalize observation
                nagent_poses = (agent_poses-self.stats["q_mean"]) / self.stats["q_std"]
                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(
                    self.device, dtype=torch.float32
                )
                # (2,2)

                # infer action
                with torch.no_grad():
                    # TODO: pad data when len(obs_deque) < obs_horizon

                    # initialize action from Guassian noise
                    noisy_action = torch.randn((B, pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

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
                naction = naction.detach().to("cpu").numpy() # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = naction*self.stats["action_std"] + self.stats["action_mean"]

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                # TODO fix when we have env implemented
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode="rgb_array"))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print("Score: ", max(rewards))

        # visualize
        from IPython.display import Video

        vwrite("vis.mp4", imgs)
        Video("vis.mp4", embed=True, width=256, height=256)


# print(summary(model, 
#         torch.zeros((BATCH_SIZE, OBSERVATION_HORIZON, 3, 64, 64), device=DEVICE), 
#         torch.zeros((BATCH_SIZE, OBSERVATION_HORIZON, 2), device=DEVICE), 
#         torch.zeros((BATCH_SIZE, STATE_DIM), device=DEVICE)))

if __name__ == "__main__":
    vision_encoder = OpenVisionEncoder().to(device=DEVICE)
    model = DiffusionModel(
        state_dim=STATE_DIM,
        obs_dim=OBSERVATION_DIM,
        obs_horizon=OBSERVATION_HORIZON,
        vision_encoder=vision_encoder,
        device=DEVICE,)
    ## Dataset and Dataloader
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
        device=DEVICE,
        diffusion_timesteps=NUM_TRAIN_TIMESTEPS,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train()

    env, curr_render = setup_env()  # TODO set up mujoco env for eval
    trainer.eval(env)