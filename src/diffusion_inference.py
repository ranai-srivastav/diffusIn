import os, sys

sys.path.append(os.path.abspath("/home/ranai/MRSD/diffusIn/include"))
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
from act.utils import EpisodicDataset, load_data, get_norm_stats

from collections import deque
from torch.utils.data import DataLoader

## Torch Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Configure params according to inference
NUM_EPOCHS = 100
NUM_EPISODES = 1
DATASET_PATH = Path("../data").absolute()
DATASET_PATH = (
    DATASET_PATH
    if DATASET_PATH.exists()
    else "DATASET IS AS LOST AS YOU ARE - NOT FOUND IN THE GIVEN PATH"
)
BATCH_SIZE = 16
NUM_TRAIN_TIMESTEPS = 500
VISION_FEATURE_DIM = 512
STATE_DIM = 16
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM

import act.ee_sim_env as act_sim_env


def setup_env():
    env = act_sim_env.make_ee_sim_env("sim_insertion")
    env.reset()

    # save visualization and rewards
    curr_render = env.render(mode="rgb_array")

    return env, curr_render


def inference_loop(dataset_path, test_indices, model_checkpoint_path):

    norm_stats = get_norm_stats(dataset_path, len(test_indices))

    # load images, qpos, and qvel from hdf5 dataset
    test_dataset = EpisodicDataset(test_indices, dataset_path, "top", norm_stats)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=OBSERVATION_HORIZON,
        prefetch_factor=OBSERVATION_HORIZON,
    )

    obs_deque = deque(
        [np.zeros((1, STATE_DIM))] * OBSERVATION_HORIZON, maxlen=OBSERVATION_HORIZON
    )

    for data in range(OBSERVATION_HORIZON):
        images, qpos, actions, is_pad = data
        obs_deque.append(
            {"image": images[0, data], "agent_pos": qpos[0, data, -STATE_DIM:]}
        )  # TODO Might be wrong

    for data in tqdm(test_dataloader):
        images, qpos, actions, is_pad = data
