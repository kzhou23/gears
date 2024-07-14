import numpy as np
import torch
import igl
from copy import copy
from scipy.spatial.transform import Rotation as R

MANO_TIPS = [744, 320, 443, 554, 671]

to_numpy = lambda tensor: tensor.detach().cpu().numpy()

to_torch = lambda array, dtype: torch.from_numpy(array).to(dtype)

def gram_schmidt(rots):
    v1 = rots[..., :3]
    v1 = v1 / torch.max(torch.sqrt(torch.sum(v1**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v2 = rots[..., 3:] - torch.sum(v1 * rots[..., 3:], dim=-1, keepdim=True) * v1
    v2 = v2 / torch.max(torch.sqrt(torch.sum(v2**2, dim=-1, keepdim=True)),
                        torch.tensor(1e-6, dtype=torch.float32, device=v1.device))
    v3 = v1.cross(v2)

    rots = torch.stack([v1, v2, v3], dim=2)
    
    return rots

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def parse_npz2(npz):
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def filter_npz(in_npz, rhand_data, lhand_data, object_data, mask, right=True):
    out_npz = {}
    out_npz['obj_name'] = in_npz['obj_name'].item()
    out_npz['n_comps'] = in_npz['n_comps'].item()
    out_npz['n_frames'] = mask.sum()
    if right:
        out_npz['rhand'] = {'params': {}}
        out_npz['rhand']['vtemp'] = in_npz['rhand'].item()['vtemp']
        for k in ('global_orient', 'hand_pose', 'transl', 'fullpose'):
            out_npz['rhand']['params'][k] = rhand_data[k][mask]
        out_npz['right'] = True
    else:
        out_npz['lhand'] = {'params': {}}
        out_npz['lhand']['vtemp'] = in_npz['lhand'].item()['vtemp']
        for k in ('global_orient', 'hand_pose', 'transl', 'fullpose'):
            out_npz['lhand']['params'][k] = lhand_data[k][mask]
        out_npz['right'] = False
    out_npz['object'] = {'params': {}}
    out_npz['object']['object_mesh'] = in_npz['object'].item()['object_mesh']
    for k in ('global_orient', 'transl'):
        out_npz['object']['params'][k] = object_data[k][mask]

    return out_npz

def filter_dict(in_dict, mask):
    out_dict = copy(in_dict)
    T = in_dict['n_frames']
    assert len(mask) == T
    for k, v in out_npz.items():
        if isinstance(v, dict):
            out_dict[k] = filter_dict(v, mask)
        elif isinstance(v, np.ndarray) and len(v.item()) == T:
            out_dict[k] = v.item[mask]
    return out_dict

def prepare_params(target, params, frame_mask=None, dtype=np.float32):
    for k in target.keys():
        if k in params.keys():
            if frame_mask is None:
                target[k] = params[k].astype(dtype)
            else:
                target[k] = params[k][frame_mask].astype(dtype)

def downsample_mask(num_frames, rate=2):
    ones = np.ones(num_frames//rate+1).astype(bool)
    zeros = [np.zeros(num_frames//rate+1).astype(bool) for _ in range(rate-1)]
    mask = np.vstack((ones, *zeros)).reshape((-1,), order='F')[:num_frames]
    return mask

def create_mano_grids(hand_joints, centers):
    T = len(hand_joints)
    wrist, mf = hand_joints[:, 0], hand_joints[:, 7]

    d1 = mf - wrist
    d1 = d1 / np.sqrt(np.sum(d1**2, axis=-1, keepdims=True))
    d2 = centers
    d2 = d2 - np.sum(d1*d2, axis=-1, keepdims=True) * d1
    d2 = d2 / np.sqrt(np.sum(d2**2, axis=-1, keepdims=True))
    grid_rots = np.concatenate([d1, d2, np.cross(d1, d2)], axis=1).reshape(-1, 3, 3)

    grid_centers = centers[:, None, :]

    l = np.linspace(-0.09, 0.09, 10)
    xc, yc, zc = np.meshgrid(l, l, l)
    cano_grid_points = np.stack([xc.flatten(), yc.flatten(), zc.flatten()], axis=1)
    cano_grid_points = np.tile(cano_grid_points, (T, 1, 1))

    grid_points = np.matmul(cano_grid_points, grid_rots.transpose(0, 2, 1)) + grid_centers
    return grid_points.reshape(-1, 3)

def create_mano_grids_corners(hand_joints, centers):
    T = len(hand_joints)
    wrist, mf = hand_joints[:, 0], hand_joints[:, 7]

    d1 = mf - wrist
    d1 = d1 / np.sqrt(np.sum(d1**2, axis=-1, keepdims=True))
    d2 = centers
    d2 = d2 - np.sum(d1*d2, axis=-1, keepdims=True) * d1
    d2 = d2 / np.sqrt(np.sum(d2**2, axis=-1, keepdims=True))
    grid_rots = np.concatenate([d1, d2, np.cross(d1, d2)], axis=1).reshape(-1, 3, 3)

    # T * 1 * 3
    grid_centers = centers[:, None, :]

    l = np.linspace(-0.09, 0.09, 2)
    xc, yc, zc = np.meshgrid(l, l, l)
    # T * 1000 * 3
    cano_grid_points = np.stack([xc.flatten(), yc.flatten(), zc.flatten()], axis=1)
    cano_grid_points = np.tile(cano_grid_points, (T, 1, 1))

    grid_points = np.matmul(cano_grid_points, grid_rots.transpose(0, 2, 1)) + grid_centers
    return grid_points.reshape(-1, 3)

def sphere_tracing(ray_origins, ray_dirs, obj_mesh, epsilon=1e-3, threshold=0.1, max_iter=100):
    dists = np.zeros(len(ray_origins))
    total_dists = np.zeros(len(ray_origins))

    valid_ray_mask = np.ones(len(ray_origins)).astype(np.bool)
    hit_mask = np.zeros(len(ray_origins)).astype(np.bool)

    for i in range(max_iter):
        if not np.any(valid_ray_mask):
            break
        cur_pos = ray_origins[valid_ray_mask] + total_dists[:, None][valid_ray_mask] * ray_dirs[valid_ray_mask]    
        dists, _, _ = igl.signed_distance(
            cur_pos,
            obj_mesh.vertices,
            obj_mesh.faces,
            return_normals=False)
        total_dists[valid_ray_mask] += np.abs(dists)
        hit_mask[valid_ray_mask] = dists < epsilon
        over_dist_mask = total_dists > threshold
        new_valid_ray_mask = ~(hit_mask | over_dist_mask)
        valid_ray_mask = new_valid_ray_mask
    total_dists[over_dist_mask] = threshold

    return total_dists

def cascaded_mask_ind(m1, m2):
    m1_ind = np.nonzero(m1)
    m2_ind = np.nonzero(m1 & m2)
    return np.in1d(m1_ind, m2_ind).nonzero()[0]

def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def homoify(points):
    points_dim = points.shape[:-1] + (1,)
    ones = points.new_ones(points_dim)

    return torch.cat([points, ones], dim=-1)


def dehomoify(points):
    return points[..., :-1] / points[..., -1:]