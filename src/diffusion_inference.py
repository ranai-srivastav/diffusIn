import os, sys
import argparse
from pathlib import Path
import collections
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml

# Env dependencies
import act.sim_env as act_sim_env
import act.utils as act_utils
import imageio
import matplotlib.pyplot as plt

# Diffusion dependencies
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_train import DiffusionModel, CLIPEncoder, OpenVisionEncoder

def parse_config(config_path):
    """
    Parse configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file (can be a string or Path object)
        
    Returns:
        dict: Configuration dictionary with the following structure:
            {
                "training": {
                    "num_epochs": int,
                    "num_episodes": int,
                    "batch_size": int,
                    "num_train_timesteps": int,
                    "ema_power": float,
                },
                "model": {
                    "vision_feature_dim": int,
                    "state_dim": int,
                    "observation_horizon": int,
                    "observation_dim": int,
                    "action_dim": int,
                    "action_horizon": int,
                    "execution_horizon": int,
                    "vision_encoder": str,
                },
                "wandb": {
                    "entity": str,
                    "project": str,
                    "track": bool,
                },
                "debug": bool,
                "device": str,
            }
            
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    # Validate config structure and set defaults for missing keys
    validated_config = {
        "training": config.get("training", {}),
        "model": config.get("model", {}),
    }
    
    return validated_config

class InferDiffusIn:
    def __init__(self, file_path, checkpoint_path=None, debug=False):

        config = parse_config(Path(file_path) / "config.yaml")
        self.num_epochs = config["training"].get("num_epochs", 100)
        self.diffusion_timesteps = config["training"].get("num_train_timesteps", 100)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.action_horizon = config["model"].get("action_horizon", 8)
        self.execution_horizon = config["model"].get("execution_horizon", 4)
        self.debug = debug
        self.file_path = file_path

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["training"].get("num_train_timesteps", 100),
            # the choice of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # load model
        if config["model"].get("vision_encoder", "OpenVision-vit-tiny-patch16-384") == "CLIP":
            self.vision_encoder = CLIPEncoder().to(self.device)
        else:
            self.vision_encoder = OpenVisionEncoder().to(self.device)
        self.ema_nets = DiffusionModel(
            state_dim=config["model"].get("state_dim", 14),
            obs_dim=config["model"].get("observation_dim", 24),
            action_dim=config["model"].get("action_dim", 14),
            obs_horizon=config["model"].get("observation_horizon", 8),
            vision_encoder=self.vision_encoder,
            device=self.device,
        )
        if checkpoint_path is None:
            checkpoint_path = file_path / ("best_diffusion_model.pth")
        self.checkpoint_path = checkpoint_path
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}")
            return
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.ema_nets.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from {checkpoint_path}")

    def eval(self, env, max_steps=500, render=False):  # default values taken from TRI example, should change
        """Evaluation Loop for Diffusion Model"""
        # |o|o|o|o|o|o|o|o|                 observations: 8
        # |p|p|p|p|p|p|p|p|                 action predictions: 8
        # | | | | | | | | |a|a|a|a|a|       actions executed: 4
        print(f"Device: {self.device}")
        pred_horizon = self.action_horizon
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

                if self.debug:
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

                    if self.debug:
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

                if self.debug:
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
            imageio.mimsave(f"data/diffusion_eval_vis/{self.checkpoint_path}.gif", imgs, fps=15)

    def visualize_actions(self, actions, save_path="action_visualization.png"):
        """Visualize action sequences as line plots for each action dimension"""

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
    env = act_sim_env.make_sim_env("sim_insertion")
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference")
    
    # Training arguments
    parser.add_argument("--file_dir_path", type=str, required=True,
                        help="Path to the config.yaml file")
    args = parser.parse_args()
    evaluator = InferDiffusIn(file_path=args.file_dir_path)
    evaluator.eval(env, max_steps=125, render=True)