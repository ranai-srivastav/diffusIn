import os, sys

sys.path.append(
    os.path.abspath("/home/parth/cmu/sem_3/planning/project/diffusIn/include")
)
from tqdm import tqdm
from pathlib import Path
import numpy as np

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
from act.utils import load_data
from transformers import CLIPModel, CLIPProcessor

## Torch Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Tunable Params
NUM_EPOCHS = 100
NUM_EPISODES = 50
DATASET_PATH = Path("../data/sim_insertion_scripted").absolute()
DATASET_PATH = (
    DATASET_PATH
    if DATASET_PATH.exists()
    else "DATASET IS AS LOST AS YOU ARE - NOT FOUND IN THE GIVEN PATH"
)
BATCH_SIZE = 16
NUM_TRAIN_TIMESTEPS = 500
VISION_FEATURE_DIM = 512
STATE_DIM = 14
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM


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
    ):
        super().__init__()
        self.vision_encoder = OpenVisionEncoder()

        self.noise_predictor = ConditionalUnet1D(
            input_dim=STATE_DIM, global_cond_dim=OBSERVATION_DIM * OBSERVATION_HORIZON
        )


noise_scheduler = DDPMScheduler(
    num_train_timesteps=NUM_TRAIN_TIMESTEPS,
    # the choice of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)

## Dataset and Dataloader
train_dataloader, val_dataloader, norm_dataset_stats, is_sim = load_data(
    DATASET_PATH, NUM_EPISODES, ["top"], BATCH_SIZE, 1
)

model = DiffusionModel().to(device=DEVICE)

## Exponential Moving Average improves Training Stability
ema = EMAModel(parameters=model.parameters(), power=0.75)

# Standard ADAM optimizer
# Note that EMA parameters are not optimized
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * NUM_EPOCHS,
)

# L2 loss
loss = torch.nn.MSELoss()

with tqdm(range(NUM_EPOCHS), desc="Epoch") as t_global:
    # epoch loop
    for epoch_idx in t_global:
        epoch_loss = list()
        # batch loop
        with tqdm(train_dataloader, desc="Batch", leave=False) as t_epoch:
            for nbatch in t_epoch:

                # device transfer
                nimage = nbatch["image"][:, :OBSERVATION_HORIZON].to(DEVICE)
                nagent_pos = nbatch["agent_pos"][:, :OBSERVATION_HORIZON].to(DEVICE)
                naction = nbatch["action"].to(DEVICE)
                B = nagent_pos.shape[0]

                # encoder vision features
                image_features = model.vision_encoder(nimage.flatten(end_dim=1))
                image_features = image_features.reshape(*nimage.shape[:2], -1)
                # (B, obs_horizon, D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

                # sample noise to add to actions
                noise = torch.randn(
                    naction.shape, device=DEVICE
                )  # NOTE: Should this be uniform? Or Gaussian?

                # sample a diffusion iteration for each data point      #NOTE: Should this be uniform? Or Gaussian?
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config["num_train_timesteps"],
                    (B,),
                    device=DEVICE,
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                # predict the noise residual
                noise_pred = model.noise_predictor(
                    noisy_actions, timesteps, global_cond=obs_cond
                )

                loss_val = loss(noise_pred, noise)

                # optimize
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(model.parameters())

                # logging
                loss_cpu = loss_val.item()
                epoch_loss.append(loss_cpu)
                t_epoch.set_postfix(loss=loss_cpu)
        t_global.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = model
ema.copy_to(ema_nets.parameters())

# TODO Save Model Checkpoint
torch.save(
    {
        "model_state_dict": ema_nets.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    },
    "diffusion_model_checkpoint.pth",
)
