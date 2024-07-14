import torch
import numpy as np
from manopth.manolayer2 import ManoLayer2 as ManoLayer
from kornia.geometry.conversions import rotation_matrix_to_angle_axis
from torch_batch_svd import svd

def quat2mat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def ik_solver_mano(pred_joints, template_joints):
    mano_layer = ManoLayer(flat_hand_mean=True, side="right",
        mano_root='/BS/kzhou2/static00/smplx/mano/', use_pca=False, center_idx=0)
    mano_layer = mano_layer.to(pred_joints.device)
    batch_size = pred_joints.shape[0]

    mano_pose = torch.eye(3).repeat(batch_size, 16, 1, 1).to(pred_joints.device)
    mano_axisang = torch.zeros((batch_size, 16, 3), dtype=torch.float32, device=pred_joints.device)
    target_joints = (pred_joints[:, :21] - pred_joints[:, [0]]).clone().detach()
    target_shape = torch.zeros((batch_size, 10), dtype=torch.float32, device=pred_joints.device)

    R = torch.eye(3).view(1, 3, 3).repeat(batch_size, 1, 1).float().to(pred_joints.device)

    finger_list = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]    
    for group_idx, group in enumerate(finger_list):
        recon_joints = torch.zeros((batch_size, 5, 3), dtype=torch.float32, device=pred_joints.device)
        for joint_idx, joint in enumerate(group):
            if joint_idx < 2:
                continue

            vec_template = template_joints[:, group[joint_idx]] - \
                template_joints[:, group[joint_idx - 1]]

            R_pa = R.clone()
            for i in range(joint_idx - 2):
                R_pa = torch.matmul(R_pa, mano_pose[:, group_idx * 3 + i + 1])

            recon_joints[:, joint_idx - 1] = torch.matmul(R_pa,
                (template_joints[:, group[joint_idx - 1]] - \
                template_joints[:, group[joint_idx - 2]]).unsqueeze(-1)).squeeze(-1) + \
                recon_joints[:, joint_idx - 2]

            vec_target = torch.matmul(R_pa.transpose(1, 2), 
                (target_joints[:, group[joint_idx]] - \
                recon_joints[:, joint_idx - 1]).unsqueeze(-1)).squeeze(-1)

            temp_axis = torch.cross(vec_template, vec_target)
            temp_axis = temp_axis / (torch.norm(temp_axis, dim=-1, keepdim=True) + 1e-7)
            overall_angle = torch.acos(torch.clamp(torch.einsum('bk, bk->b',
                vec_template, vec_target).unsqueeze(-1) / \
                (torch.norm(vec_template, dim=-1, keepdim=True) + 1e-7) / \
                (torch.norm(vec_target, dim=-1, keepdim=True) + 1e-7), -1 + 1e-7, 1 - 1e-7))
            mano_axisang[:, group_idx * 3 + joint_idx - 2 + 1] = overall_angle * temp_axis
            local_R = batch_rodrigues(overall_angle * temp_axis).reshape(batch_size, 3, 3)
            mano_pose[:, group_idx * 3 + joint_idx - 2 + 1] = local_R

    _, joints, global_trans = mano_layer(mano_axisang.reshape((batch_size, -1)), target_shape)

    return joints, global_trans