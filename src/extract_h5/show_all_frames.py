#!/usr/bin/env python3
"""Show all frames of an image stack sequentially in one window.
Usage:
  python data/show_all_frames.py data/episode_0.hdf5 observations/images/top
If dataset path is omitted, defaults to observations/images/top.
Press Ctrl+C to stop early.
"""
import sys, time
import h5py
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python data/show_all_frames.py <file> [dataset]")
    sys.exit(1)

fname = sys.argv[1]
dset_path = sys.argv[2] if len(sys.argv) > 2 else 'observations/images/top'

with h5py.File(fname, 'r') as f:
    if dset_path not in f:
        print(f'Dataset {dset_path} not found.')
        print('Available datasets:')
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(' -', name, obj.shape)
        f.visititems(visitor)
        sys.exit(1)
    data = f[dset_path][()]

# Expect shape (N,H,W,C) or (N,H,W)
if data.ndim < 3:
    print('Dataset is not an image stack (needs at least 3 dims).')
    sys.exit(1)
if data.ndim == 3:  # (N,H,W) grayscale
    cmap = 'gray'
else:
    cmap = None

fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap=cmap)
ax.set_title(f'{dset_path}[0]')
ax.axis('off')
plt.pause(0.1)

try:
    for i in range(1, data.shape[0]):
        im.set_data(data[i])
        ax.set_title(f'{dset_path}[{i}]')
        plt.pause(0.05)  # adjust delay as needed
        if i == 30:
            plt.imsave(f"Frame_{i}.png", data[i], cmap=cmap)
    plt.show()
except KeyboardInterrupt:
    print('\nStopped.')
