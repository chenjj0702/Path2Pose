""" Define the dance dataset. """
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class LarvaDataset(Dataset):
    """ 数据集类 """
    def __init__(self, data_dict):
        self.pose = data_dict['pose']
        self.guide_path = data_dict['guide_path']

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, index):
        return self.pose[index], self.guide_path[index]


def move(X, points):
    assert len(X.shape) == 4
    if isinstance(points, int):
        path = X[:, :, [points], :]
    elif isinstance(points, list):
        path = X[:, :, points, :].mean(2, keepdims=True)
    else:
        raise EOFError
    pose = X - path
    path = path.squeeze()
    return pose, path


def load_data(f_name, base_points):
    data_load = np.load(f_name, allow_pickle=True)
    full_dict = data_load['full_dict'].item()  # (n,T,22,2)

    for k_, v_ in full_dict.items():  # 'train' and 'test'
        pose = v_.astype(np.float32)  # (n,T,22,2)
        _, real_path = move(pose, base_points)
        pose = pose.reshape((pose.shape[0], pose.shape[1], -1))

        if k_ == 'train':
            train_pose, train_path = pose, real_path
        elif k_ == 'test':
            test_pose, test_path = pose, real_path
        else:
            raise EOFError

    train_data = {'pose': train_pose, 'guide_path': train_path}
    test_data = {'pose': test_pose, 'guide_path': test_path}

    return train_data, test_data
