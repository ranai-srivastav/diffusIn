#!/usr/bin/env python3
"""Minimal HDF5 image visualizer.

Examples:
  python data/visualize_hdf5.py data/episode_0.hdf5 --list
  python data/visualize_hdf5.py data/episode_0.hdf5 -d observations/images/top -i 0
  python data/visualize_hdf5.py data/episode_0.hdf5 -d observations/images/top --grid 4

Features:
  - List datasets.
  - Visualize a single frame by index.
  - Optional grid of first N frames.
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt


def find_datasets(h5file):
    out = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj.shape, str(obj.dtype)))
    h5file.visititems(visitor)
    return out


def normalize(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.integer):
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img.astype(np.float32) - mn) / (mx - mn)
        else:
            img = img.astype(np.float32)
    elif np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
    return img


def channel_last(arr):
    # Convert (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        return np.transpose(arr, (1, 2, 0))
    return arr


def frame_count(arr):
    # Decide if first dimension is a frame dimension
    if arr.ndim == 4:  # (N,H,W,C) or (N,C,H,W)
        return arr.shape[0]
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        return arr.shape[0]
    return 1


def extract_frame(arr, index):
    if arr.ndim == 4:
        frame = arr[index]
        frame = channel_last(frame)
        return frame
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        frame = arr[index]
        return frame
    # Single image already
    return arr


def show_single(frame, title, if_save:bool):
    frame = channel_last(frame)
    frame = normalize(frame)
    if frame.ndim == 2:
        plt.imshow(frame, cmap='gray')
    else:
        plt.imshow(frame)
        
    if if_save:
        plt.savefig(f"{title.replace('/','_')}.png")
    plt.title(title)
    plt.axis('off')


def show_grid(arr, n):
    n = min(n, frame_count(arr))
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.asarray(axes).reshape(-1)
    for i in range(rows*cols):
        ax = axes[i]
        if i < n:
            frame = extract_frame(arr, i)
            frame = channel_last(frame)
            frame = normalize(frame)
            if frame.ndim == 2:
                ax.imshow(frame, cmap='gray')
            else:
                ax.imshow(frame)
            ax.set_title(f'frame {i}')
        ax.axis('off')
    fig.tight_layout()


def main():
    ap = argparse.ArgumentParser(description='Visualize image datasets in an HDF5 file.')
    ap.add_argument('file', help='Path to HDF5 file')
    ap.add_argument('-d', '--dataset', help='Dataset path to visualize')
    ap.add_argument('-i', '--index', type=int, default=0, help='Frame index (for stacks)')
    ap.add_argument('--grid', type=int, help='Show a grid of first N frames')
    ap.add_argument('--list', action='store_true', help='List datasets and exit')
    ap.add_argument('--save', default='false', action='store_true', help='Save visualization to file instead of showing')
    args = ap.parse_args()

    with h5py.File(args.file, 'r') as f:
        if args.list or not args.dataset:
            dsets = find_datasets(f)
            print('Datasets:')
            for name, shape, dt in dsets:
                print(f' - {name}: shape={shape}, dtype={dt}')
            if args.list:
                return
            if not args.dataset:
                print('\nSpecify one with -d/--dataset to visualize.')
                return
        if args.dataset not in f:
            raise SystemExit(f'Dataset {args.dataset} not found.')
        arr = f[args.dataset][()]

    if args.grid:
        show_grid(arr, args.grid, args.save)
    else:
        frame = extract_frame(arr, args.index)
        show_single(frame, f'{args.dataset}[{args.index}]')
    plt.show()


if __name__ == '__main__':
    main()
