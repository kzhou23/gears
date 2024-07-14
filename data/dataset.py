import torch
import numpy as np
import os, glob
from torch.utils import data


class GRAB(data.Dataset):
    def __init__(self, path, split, window_size=30, step_size=15, num_points=4000):
        self.path = path
        self.split = split
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points

        seqs = glob.glob(os.path.join(path, split, '*.npy'))
        self.len = 0
        self.data = []
        for f in seqs:
            seq = np.load(f)
            seq_len = (len(seq) - window_size) // step_size + 1
            self.data.append((self.len, self.len+seq_len, seq))
            self.len += seq_len
        self.data.sort(key=lambda x: x[0])

    def __getitem__(self, index):
        for s in self.data:
            if index < s[1]:
                break
        start_idx = (index - s[0]) * self.step_size
        data = s[2][start_idx:start_idx+self.window_size]

        rhand_disp = data['f7'].reshape(self.window_size, -1)
        rhand_pos_traj = data['f8'].reshape(self.window_size, -1)
        rhand_ori_traj = data['f9'].reshape(self.window_size, -1)
        rhand_traj = np.concatenate([rhand_disp, rhand_pos_traj, rhand_ori_traj], axis=-1)

        obj_pc_crop = data['f13'].reshape(self.window_size, -1, 3)
        obj_normal_crop = data['f14'].reshape(self.window_size, -1, 3)
        obj_crop = np.concatenate([obj_pc_crop, obj_normal_crop], axis=-1)

        obj_pc = data['f15'].reshape(self.window_size, -1, 3)
        obj_normal = data['f16'].reshape(self.window_size, -1, 3)
        if self.num_points < obj_pc.shape[1]:
            samp_ind = np.arange(obj_pc.shape[1])
            np.random.shuffle(samp_ind)
            samp_ind = samp_ind[:self.num_points]
            obj_full = np.concatenate([obj_pc[:, samp_ind], obj_normal[:, samp_ind],], axis=-1)
        else:
            obj_full = np.concatenate([obj_pc, obj_normal], axis=-1)

        rhand_joints = data['f1']

        return rhand_traj, obj_crop, obj_full, rhand_joints

    def __len__(self):
        return self.len
