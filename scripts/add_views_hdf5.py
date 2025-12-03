"""
Script to add angle and vis views to the HDF5 files with only top view.
"""

import os
import sys
from pathlib import Path


sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
    ]
)

import h5py
import numpy as np
from tqdm import tqdm

import act.sim_env as act_sim_env
import act.utils as act_utils


DATA_ROOT = Path("data").absolute()
INPUT_SUBDIRS = ["sim_insertion_human", "sim_insertion_scripted"]
OUTPUT_ROOT = Path("data_augmented").absolute()
TASK_NAME = "sim_insertion"


def find_hdf5_files():
    files = []
    for sub in INPUT_SUBDIRS:
        src_dir = DATA_ROOT / sub
        if not src_dir.exists():
            continue
        files.extend(sorted(src_dir.glob("*.hdf5")))
    return files


def ensure_output_path(src_path: Path) -> Path:
    # Preserve directory structure under OUTPUT_ROOT
    relative = src_path.relative_to(DATA_ROOT)
    out_path = OUTPUT_ROOT / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def needs_augmentation(dst_path: Path) -> bool:
    if not dst_path.exists():
        return True
    # If file exists, check if angle and vis are present
    try:
        with h5py.File(str(dst_path), "r") as f:
            has_angle = "/observations/images/angle" in f
            has_vis = "/observations/images/vis" in f
            return not (has_angle and has_vis)
    except Exception:
        return True


def augment_file(src_path: Path, dst_path: Path):
    # Read original episode
    with h5py.File(str(src_path), "r") as src:
        qpos = np.array(src["/observations/qpos"])  # (T, 14)
        qvel = np.array(src["/observations/qvel"])  # (T, 14)
        action = np.array(src["/action"])           # (T, 14)
        top_images = np.array(src["/observations/images/top"])  # (T, H, W, 3)
        is_sim = bool(src.attrs.get("sim", True))
        # Try to recover the original object poses (env_state at step 0) to keep outcomes identical
        env_state0 = None
        if "/observations/env_state" in src:
            try:
                env_state0 = np.array(src["/observations/env_state"][0])
            except Exception:
                print(f"Failed to find env_state in {src_path}")
                env_state0 = None

    T = qpos.shape[0]

    # Create a fresh environment and reset with the SAME object poses if available
    if env_state0 is not None:
        act_sim_env.BOX_POSE[0] = env_state0
    else:
        # Fallback to a random pose if not recorded in this file
        peg_pose, socket_pose = act_utils.sample_insertion_pose()
        act_sim_env.BOX_POSE[0] = np.concatenate([peg_pose, socket_pose])
    env = act_sim_env.make_sim_env(TASK_NAME)
    ts = env.reset()

    # Rollout using recorded qpos and collect angle/vis
    angle_frames = []
    vis_frames = []
    for t in tqdm(range(T), desc=f"Augment {src_path.stem}"):
        ts = env.step(qpos[t])
        imgs = ts.observation["images"]
        angle_frames.append(imgs["angle"])
        vis_frames.append(imgs["vis"])

    angle_frames = np.asarray(angle_frames, dtype=np.uint8)
    vis_frames = np.asarray(vis_frames, dtype=np.uint8)

    # Save to augmented path, preserving structure
    with h5py.File(str(dst_path), "w", rdcc_nbytes=1024 ** 2 * 2) as dst:
        dst.attrs["sim"] = is_sim
        obs = dst.create_group("observations")
        image = obs.create_group("images")

        # Create datasets with same lengths/shapes
        _ = image.create_dataset("top", data=top_images, dtype="uint8", chunks=(1,) + top_images.shape[1:])
        _ = image.create_dataset("angle", data=angle_frames, dtype="uint8", chunks=(1,) + angle_frames.shape[1:])
        _ = image.create_dataset("vis", data=vis_frames, dtype="uint8", chunks=(1,) + vis_frames.shape[1:])

        _ = obs.create_dataset("qpos", data=qpos)
        _ = obs.create_dataset("qvel", data=qvel)
        _ = dst.create_dataset("action", data=action)


def main():
    files = find_hdf5_files()
    if not files:
        print(f"No hdf5 files found in {INPUT_SUBDIRS} under {DATA_ROOT}")
        return

    for src_path in tqdm(files, desc="Processing files"):
        dst_path = ensure_output_path(src_path)
        if not needs_augmentation(dst_path):
            print(f"Skipping (already augmented): {dst_path}")
            continue
        print(f"Augmenting: {src_path} -> {dst_path}")
        augment_file(src_path, dst_path)


if __name__ == "__main__":
    main()


