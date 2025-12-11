import os, sys
import argparse
from pathlib import Path
import collections
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml


CAMERA_KEYS = ("top", "angle", "vis")  # available in env observations
FPS = 50

sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
        str(Path("include/OpenVision").resolve()),
    ]
)

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
        dict: Configuration dictionary
            
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

    # Load dataset stats
    dataset_stats = config.get("dataset_stats", None)
    if dataset_stats is None:
        raise ValueError(f"Config file does not contain dataset stats: {config_path}")

    stats = {
        "action_mean": np.asarray(dataset_stats.get("action_mean")),
        "action_std": np.asarray(dataset_stats.get("action_std")),
        "qpos_mean": np.asarray(dataset_stats.get("qpos_mean")),
        "qpos_std": np.asarray(dataset_stats.get("qpos_std")),
        "qvel_mean": np.asarray(dataset_stats.get("qvel_mean")),
        "qvel_std": np.asarray(dataset_stats.get("qvel_std")),
    }
    validated_config["dataset_stats"] = stats

    return validated_config

class InferDiffusIn:
    def __init__(self, file_path, checkpoint_name=None, device="cuda" if torch.cuda.is_available() else "cpu", debug=False):

        config = parse_config(Path(file_path) / "config.yaml")
        self.diffusion_timesteps = config["training"].get("num_train_timesteps", 100)
        self.device = device
        self.pred_horizon = config["model"].get("pred_horizon", 8)
        self.execution_horizon = config["model"].get("execution_horizon", 4)
        self.obs_horizon = config["model"].get("observation_horizon", 2)
        self.stats = config.get("dataset_stats", None)
        self.debug = debug
        self.multiview = config["training"].get("multiview", False)
        self.file_path = file_path
        if checkpoint_name is None:
            self.checkpoint_name = "last_diffusion_model_checkpoint.pth"
        else:
            self.checkpoint_name = checkpoint_name


        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_timesteps,
            # the choice of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # load model
        if config["model"].get("vision_encoder", "OpenVision-vit-tiny") == "CLIP":
            self.vision_encoder = CLIPEncoder().to(self.device)
        else:
            self.vision_encoder = OpenVisionEncoder().to(self.device)
        self.ema_nets = DiffusionModel(
            state_dim=config["model"].get("state_dim", 14),
            obs_dim=config["model"].get("observation_dim", 206),
            action_dim=config["model"].get("action_dim", 14),
            obs_horizon=self.obs_horizon,
            vision_encoder=self.vision_encoder,
            device=self.device,
            multiview=self.multiview,
        )
    
        checkpoint_path = Path(file_path) / self.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.ema_nets.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from {checkpoint_path}")

    def _capture_views(self, ts):
        """Capture frames from the environment observations"""
        obs_imgs = ts.observation["images"]
        imgs = []
        for cam in CAMERA_KEYS:
            img = obs_imgs[cam]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(img)
        return imgs

    def _stack_images(self, imgs, horizontal=True):
        """Stack images horizontally or vertically"""
        h_min = min(im.shape[0] for im in imgs)
        w_min = min(im.shape[1] for im in imgs)
        imgs_cropped = [im[:h_min, :w_min, :] for im in imgs]
        if horizontal:
            stacked = np.concatenate(imgs_cropped, axis=1)
        else:
            stacked = np.concatenate(imgs_cropped, axis=0)
        return stacked

    def _get_frame_from_env(self, ts):
        """Get frame from environment observations"""
        imgs = self._capture_views(ts)
        stacked = self._stack_images(imgs)
        return stacked

    def eval(self, env, max_steps=500, render_offscreen=False, render_onscreen=False):  # default values taken from TRI example, should change
        """Evaluation Loop for Diffusion Model"""
#         in training:
        # |o|o|o|o|o|o|o|o|                 pred_horizon: 8
        # |h|h| | | | | | |                 obs_horizon: 2
        # given conditioning of length obs_horizon, we predict the action seq for the whole pred_horizon 
        # (with conditioning mask, not pictured) 

        # in eval:
        # |o|o|o|o|o|o|o|o|                 pred_horizon: 8
        # |h|h| | | | | | |                 obs_horizon: 2
        # | |a|a|a|a| | | |              execution_horizon: 4
        # given conditioning of length obs_horizon, we predict action seq for the whole pred_horizon, 
        # but only execute execution_horizon actions starting from the current obs (last idx in obs_horizon)
        print(f"Device: {self.device}")
        self.ema_nets.eval()

        # Reset environment with random peg and socket pose
        #peg_pose, socket_pose = act_utils.sample_insertion_pose()
        # fixed poses
        peg_pose, socket_pose = act_utils.get_insertion_pose()
        print("Peg Pose: ", peg_pose)
        print("Socket Pose: ", socket_pose)
        act_sim_env.BOX_POSE[0] = np.concatenate([peg_pose, socket_pose])
        ts = env.reset()

        # Init obs deque with initial observations
        # obs keys:'qpos', 'qvel', 'env_state', 'images'
        # keys in obs['images']: 'top', 'angle', 'vis'. Image shape: (480, 640, 3)
        obs = ts.observation
        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        frames = []
        if render_offscreen:
            frames.append(self._get_frame_from_env(ts))

        if render_onscreen:
            # Create a borderless, axis-free, full-figure render sized to the frame
            frame = self._get_frame_from_env(ts)
            h, w = frame.shape[:2]
            dpi = 100
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])   # fill entire figure
            ax.set_axis_off()
            plt_img = ax.imshow(frame, aspect="equal")
            plt.ion()
            print("Creating onscreen render")

        rewards = []
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval SimInsertion") as pbar:
            while not done:

                B = 1
                # stack the last obs_horizon number of observations
                if self.multiview:
                    image_top = np.stack([x["images"]["top"] for x in obs_deque])
                    image_angle = np.stack([x["images"]["angle"] for x in obs_deque])
                    image_vis = np.stack([x["images"]["vis"] for x in obs_deque])
                    images = np.stack([image_top, image_angle, image_vis], axis=0)
                    images = images.reshape((self.obs_horizon*3, *images.shape[-3:])) 
                else:
                    images = np.stack([x["images"]["top"] for x in obs_deque]) # (obs_horizon, h, w, c)
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
                        (B, self.pred_horizon, self.ema_nets.action_dim), device=self.device
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
                start = self.obs_horizon - 1
                end = start + self.execution_horizon
                action = action_pred[start:end, :]  # (execution_horizon, action_dim)

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

                    if render_offscreen:
                        frames.append(self._get_frame_from_env(ts))

                    if render_onscreen:
                        # Update onscreen render with current frame
                        frame = self._get_frame_from_env(ts)
                        plt_img.set_data(frame)
                        plt.pause(0.02)

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

        if render_offscreen:
            # save vis as gif
            save_path = f"{self.file_path}/eval_vis_{self.checkpoint_name}.gif"
            # NOTE: The original hz is 50, but we set it to a custom value here.
            imageio.mimsave(save_path, frames, fps=FPS, loop=0)  # loop=0 means infinite loop
            print(f"Saved GIF to {save_path}")

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
    
    parser.add_argument("--file_dir_path", type=str, required=True,
                        help="Path to the config.yaml file")
    parser.add_argument("--checkpoint_name", type=str, default=None,)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (default: cuda if available else cpu)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum number of steps to run in the environment")
    parser.add_argument("--render_onscreen", action="store_true", default=False,
                        help="Render the environment onscreen")
    parser.add_argument("--save_gif", action="store_true", default=False,
                        help="Render the environment offscreen and save as GIF")
    args = parser.parse_args()
    evaluator = InferDiffusIn(file_path=args.file_dir_path, checkpoint_name=args.checkpoint_name, device=args.device)
    evaluator.eval(env, max_steps=args.max_steps, render_offscreen=args.save_gif, render_onscreen=args.render_onscreen)