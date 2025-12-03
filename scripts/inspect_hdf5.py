"""
Script to inspect the fields inside an HDF5 file and visualize the images.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# Global: set this to your file path
HDF5_FILE_PATH = Path("data/recorded/episode_3.hdf5")


def _print_hdf5_keys(h5: h5py.File) -> None:
    # Top-level keys
    print("Top-level keys:")
    for k in h5.keys():
        print(f"- {k}")

    print("\nAll paths:")
    def _visitor(name, obj):
        obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        print(f"{name} ({obj_type})")

    h5.visititems(_visitor)

def _print_image_group_keys(h5: h5py.File) -> None:
    key_path = "/observations/images"
    if key_path in h5:
        print(f"\nKeys under {key_path}:")
        img_group = h5[key_path]
        for k in img_group.keys():
            print(f"- {k}")
    else:
        print(f"\nGroup {key_path} not found.")

def visualize_episode(h5: h5py.File) -> None:
    img_root = "/observations/images"
    if img_root not in h5:
        print(f"{img_root} not found in file.")
        return

    # Get length of box pose from env_state
    env_state = h5["/observations/env_state"]
    box_pose = env_state[0]
    print(f"box_pose: {box_pose}")
    print(f"env_state length: {len(env_state)}")

    img_group = h5[img_root]
    # Prefer ordered cameras if present; else use whatever exists
    preferred_order = ["top", "angle", "vis"]
    cams = [c for c in preferred_order if c in img_group.keys()]
    print(f"cams: {cams}")
    if not cams:
        cams = list(img_group.keys())
    if len(cams) == 0:
        print("No camera datasets found under /observations/images.")
        return

    # Number of frames from the first camera
    T = img_group[cams[0]].shape[0]
    # Ensure all cameras have at least T frames
    for c in cams:
        if img_group[c].shape[0] != T:
            T = min(T, img_group[c].shape[0])

    # Determine common height/width for safe concatenation (crop to min)
    heights = []
    widths = []
    for c in cams:
        _, h, w, _ = img_group[c].shape
        heights.append(h)
        widths.append(w)
    min_h, min_w = min(heights), min(widths)

    # Initialize figure sized to concatenated width
    first_imgs = [np.asarray(img_group[c][0])[:min_h, :min_w, :] for c in cams]
    concat0 = np.concatenate(first_imgs, axis=1)
    h0, w0 = concat0.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w0 / dpi, h0 / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    plt_img = ax.imshow(concat0)
    plt.ion()
    plt.show(block=False)

    # Iterate and update
    for t in range(T):
        frame_imgs = []
        for c in cams:
            img = np.asarray(img_group[c][t])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            frame_imgs.append(img[:min_h, :min_w, :])
        concat = np.concatenate(frame_imgs, axis=1)
        plt_img.set_data(concat)
        fig.canvas.draw_idle()
        plt.pause(0.02)

def main():
    print(f"Reading: {HDF5_FILE_PATH}")
    with h5py.File(str(HDF5_FILE_PATH), "r") as f:
        _print_hdf5_keys(f)
        _print_image_group_keys(f)
        visualize_episode(f)


if __name__ == "__main__":
    main()


