import os
import numpy as np
import glob
import h5py
import torch
from torch.utils.data import DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        """Dataset class that we will use to read with ACT MuJoCo sim

        Args:
            dataset_dir (str): Directory where the dataset files are stored
            camera_names (list): List of camera names to load images from, "top" only in our case
            norm_stats (dict): Normalization statistics for observations and actions
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None

        # get episode paths
        human_episodes = sorted(
            glob.glob(os.path.join(dataset_dir, "sim_insertion_human", "*.hdf5"))
        )
        scripted_episodes = sorted(
            glob.glob(os.path.join(dataset_dir, "sim_insertion_scripted", "*.hdf5"))
        )
        all_episode_paths = human_episodes + scripted_episodes
        self.episode_paths = [all_episode_paths[i] for i in episode_ids]

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_path = self.episode_paths[index]
        with h5py.File(episode_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            
            # TODO cast to list is unnecessary?
            qpos = np.array(list(root['/observations/qpos']))
            qvel = np.array(list(root['/observations/qvel']))
            images = np.array(list(root['/observations/images/top']))
            action = np.array(list(root['/action']))

        # construct observations
        image_data = torch.from_numpy(images)
        qpos_data = torch.from_numpy(qpos).float()
        qvel_data = torch.from_numpy(qvel).float()
        action_data = torch.from_numpy(action).float()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # Fixed: Properly indexing q_mean and q_std to separate qpos and qvel
        qpos_mean = self.norm_stats["qpos_mean"]
        qvel_mean = self.norm_stats["qvel_mean"]
        qpos_std = self.norm_stats["qpos_std"]
        qvel_std = self.norm_stats["qvel_std"]

        # normalize image and change dtype to float
        #TODO: Check if range between 0 to 1 is needed based on vision encoder
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - qpos_mean) / qpos_std
        qvel_data = (qvel_data - qvel_mean) / qvel_std

        values = {
            "image": image_data,
            "q_pos": qpos_data,
            "q_vel": qvel_data,
            "action": action_data
        }

        return values


def get_norm_stats(dataset_dir, episode_ids):
    all_qpos_data = []
    all_action_data = []
    all_qvel_data = []

    # get episode paths
    episode_paths = []
    human_episodes = sorted(
        glob.glob(os.path.join(dataset_dir, "sim_insertion_human", "*.hdf5"))
    )
    scripted_episodes = sorted(
        glob.glob(os.path.join(dataset_dir, "sim_insertion_scripted", "*.hdf5"))
    )
    episode_paths = human_episodes + scripted_episodes
    selected_episode_paths = [episode_paths[i] for i in episode_ids]
        
    for path in selected_episode_paths:
        with h5py.File(path, 'r') as root:
            qpos = np.array(root['/observations/qpos'])
            qvel = np.array(root['/observations/qvel'])
            action = np.array(root['/action'])

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_qvel_data.append(torch.from_numpy(qvel))
    
    # concatenate along time dimension to handle variable-length episodes
    all_qpos_data = torch.cat(all_qpos_data, dim=0)      # (sum_T, qpos_dim)
    all_action_data = torch.cat(all_action_data, dim=0)  # (sum_T, action_dim)
    all_qvel_data = torch.cat(all_qvel_data, dim=0)      # (sum_T, qvel_dim)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    
    # normalize qvel data
    qvel_mean = all_qvel_data.mean(dim=[0], keepdim=True)
    qvel_std = all_qvel_data.std(dim=[0], keepdim=True)
    qvel_std = torch.clip(qvel_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "qvel_mean": qvel_mean.numpy().squeeze(), "qvel_std": qvel_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    # BUGFIX: Data leakage. Need to look at train data only to get normalization stats
    norm_stats = get_norm_stats(dataset_dir, train_indices)

    # construct dataset and dataloader
    assert num_episodes > 1, "num_episodes must be greater than 1 to perform train/val split."
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
