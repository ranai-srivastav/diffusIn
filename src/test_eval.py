import os
import sys
from pathlib import Path

sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
        str(Path("include/OpenVision").resolve()),
    ]
)

import h5py
import imageio
import numpy as np
from tqdm import tqdm

import act.sim_env as act_sim_env
import act.utils as act_utils


# Global configuration (edit HDF5_FILE_PATH to point to your file)
HDF5_FILE_PATH = Path("data/sim_insertion_scripted/episode_0.hdf5")
OUTPUT_DIR = Path("data/test_eval_vis").absolute()
TASK_NAME = "sim_insertion"
CAMERA_KEYS = ("top", "angle", "vis")  # available in env observations
FPS = 33


def _resolve_hdf5_path(default_path: Path) -> Path:
    if default_path.is_file():
        return default_path
    # Fallback: pick the first episode found in common dataset dirs
    candidates = list(Path("data").glob("sim_insertion_*/*.hdf5"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"HDF5 file not found. Set HDF5_FILE_PATH correctly. Tried default: {default_path}"
    )


def _stack_vertical(frames_list):
    return np.concatenate(frames_list, axis=0)


def main():
    hdf5_path = _resolve_hdf5_path(HDF5_FILE_PATH)
    print(f"Using HDF5: {hdf5_path}")

    # Set random peg and socket pose, then create/reset env
    peg_pose, socket_pose = act_utils.sample_insertion_pose()
    act_sim_env.BOX_POSE[0] = np.concatenate([peg_pose, socket_pose])
    env = act_sim_env.make_sim_env(TASK_NAME)
    ts = env.reset()

    # Load q_pos from HDF5
    with h5py.File(str(hdf5_path), "r") as f:
        if "/observations/qpos" not in f:
            raise KeyError("Dataset missing '/observations/qpos'")
        qpos_seq = np.array(f["/observations/qpos"])  # shape: (T, 14)

    frames = []

    # Step through q_pos and capture images
    for t in tqdm(range(qpos_seq.shape[0]), desc="Stepping episode"):
        action = qpos_seq[t]
        ts = env.step(action)

        obs_imgs = ts.observation["images"]
        imgs = []
        for cam in CAMERA_KEYS:
            img = obs_imgs[cam]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(img)

        # Ensure consistent sizes before stacking
        h_min = min(im.shape[0] for im in imgs)
        w_min = min(im.shape[1] for im in imgs)
        imgs_cropped = [im[:h_min, :w_min, :] for im in imgs]
        stacked = _stack_vertical(imgs_cropped)
        frames.append(stacked)

    # Save GIF
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = hdf5_path.stem
    out_path = OUTPUT_DIR / f"{out_name}.gif"
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print(f"Saved GIF to: {out_path}")


if __name__ == "__main__":
    main()


