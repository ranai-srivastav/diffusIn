import os, sys
sys.path.append(os.path.abspath("/home/ranai/MRSD/diffusIn/include"))
from tqdm import tqdm
from pathlib import Path
import numpy as np

## Encoder Dependencies
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import adjust_brightness
from PIL import Image

#Diffusion Dependencies
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from OpenVision.src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms
from diffusion_layers import Conv1dBlock, Upsample1d, Downsample1d, ConditionalResidualBlock1D, ConditionalUnet1D, SinusoidalPosEmb
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from act.utils import EpisodicDataset, load_data, get_norm_stats

from collections import deque
from torch.utils.data import DataLoader

## Torch Params
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Configure params according to inference
NUM_EPOCHS = 100
NUM_EPISODES = 100
DATASET_PATH = Path("../data").absolute()
DATASET_PATH = DATASET_PATH if DATASET_PATH.exists() else "DATASET IS AS LOST AS YOU ARE - NOT FOUND IN THE GIVEN PATH"
BATCH_SIZE = 16
NUM_TRAIN_TIMESTEPS = 500
VISION_FEATURE_DIM = 512
STATE_DIM = 16
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM

import act.ee_sim_env as act_sim_env

def setup():
    env = act_sim_env.make_ee_sim_env("sim_insertion")
    env.reset()

    # save visualization and rewards
    curr_render = env.render(mode='rgb_array')
    
    return env, curr_render


def inference_loop(dataset_path, test_indices, model_checkpoint_path):
    
    norm_stats = get_norm_stats(dataset_path, len(test_indices))
    
    # load images, qpos, and qvel from hdf5 dataset
    test_dataset = EpisodicDataset(test_indices, dataset_path, "top", norm_stats)
    test_dataloader =  DataLoader(test_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  pin_memory=True, 
                                  num_workers=OBSERVATION_HORIZON, 
                                  prefetch_factor=OBSERVATION_HORIZON)
    
    obs_deque = deque([np.zeros((1, STATE_DIM))] * OBSERVATION_HORIZON, maxlen=OBSERVATION_HORIZON)
    
    for data in range(OBSERVATION_HORIZON):
        images, qpos, actions, is_pad = data
        obs_deque.append({'image': images[0, data], 'agent_pos': qpos[0, data, -STATE_DIM:]}) #TODO Might be wrong
    
    for data in tqdm(test_dataloader):
        images, qpos, actions, is_pad = data
        
        
        
    

def one_step_inference(env, model, curr_observation):
    with tqdm(total=NUM_TRAIN_TIMESTEPS, desc="Eval SimInsertion") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets['vision_encoder'](nimages) #TODO check if forward/__call__ implementation is correct
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    vwrite('vis.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)