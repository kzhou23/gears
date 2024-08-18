import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob, argparse
import smplx
import trimesh
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation as R
from pysdf import SDF
from tqdm import tqdm
from utils.utils import MANO_TIPS, to_numpy, to_torch, parse_npz, parse_npz2, prepare_params, \
    downsample_mask, filter_npz

class GRAB_Preprocessing:
    def __init__(self, cfg):
        self.cfg = cfg
        self.framerate = cfg['framerate']
        self.grab_path = cfg['grab_path']
        self.model_path = cfg['model_path']
        self.out_path = cfg['out_path']
        self.opt = cfg['opt']

        print('Starting data preprocessing !')

        self.objname2id = {}
        obj_meshes = sorted(os.listdir(os.path.join(self.grab_path, '../tools/object_meshes/contact_meshes')))
        for i, fn in enumerate(obj_meshes):
            obj_name = fn.split('.')[0]
            self.objname2id[obj_name] = i

        self.splits = cfg['splits']
        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')
        self.split_seqs = {'test': [], 'val': [], 'train': []}

        self.split_sequences()
        self.process_sequences()

    def split_sequences(self):
        for sequence in self.all_seqs:
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)

    def preprocess_sequences(self):
        for split in self.split_seqs.keys():
            save_path = os.path.join(self.out_path, split+'_pre')
            os.makedirs(save_path, exist_ok=True)
            print('Preprocessing data for %s split.' % (split))

            k = 0
            for i, sequence in enumerate(tqdm(sorted(self.split_seqs[split]))):
                npz = np.load(sequence, allow_pickle=True)
                seq_data = parse_npz2(npz)
                obj_name = seq_data.obj_name
                n_comps  = seq_data.n_comps
                n_frames = seq_data.n_frames

                ds_mask = downsample_mask(n_frames, rate=120//self.framerate)
                right_contact_mask = self.filter_contact_frames(seq_data, right=True)
                left_contact_mask = self.filter_contact_frames(seq_data, right=False)

                rhand_data = {'global_orient': [], 'hand_pose': [], 'transl': [],
                    'fullpose': []}
                lhand_data = {'global_orient': [], 'hand_pose': [], 'transl': [],
                    'fullpose': []}
                object_data = {'global_orient': [], 'transl': [], 'id': []}
                    
                prepare_params(rhand_data, seq_data.rhand.params, ds_mask)
                prepare_params(lhand_data, seq_data.lhand.params, ds_mask)
                prepare_params(object_data, seq_data.object.params, ds_mask)

                n_frames = ds_mask.sum()

                rhand_temp_path = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
                rhand_vtemp = np.array(Mesh(filename=rhand_temp_path).v)
                lhand_temp_path = os.path.join(grab_path, '..', seq_data.lhand.vtemp)
                lhand_vtemp = np.array(Mesh(filename=lhand_temp_path).v)

                rh_m = smplx.create(model_path=cfg['model_path'],
                                    model_type='mano',
                                    is_rhand=True,
                                    v_template=rhand_vtemp,
                                    num_pca_comps=n_comps,
                                    flat_hand_mean=True,
                                    batch_size=n_frames)
                lh_m = smplx.create(model_path=cfg['model_path'],
                                    model_type='mano',
                                    is_rhand=False,
                                    v_template=lhand_vtemp,
                                    num_pca_comps=n_comps,
                                    flat_hand_mean=True,
                                    batch_size=n_frames)

                rh_output = rh_m(
                    hand_pose=to_torch(rhand_data['hand_pose'], torch.float32),
                    global_orient=to_torch(rhand_data['global_orient'], torch.float32),
                    transl=to_torch(rhand_data['transl'], torch.float32),
                )
                lh_output = lh_m(
                    hand_pose=to_torch(lhand_data['hand_pose'], torch.float32),
                    global_orient=to_torch(lhand_data['global_orient'], torch.float32),
                    transl=to_torch(lhand_data['transl'], torch.float32),
                )
                rwrists = to_numpy(rh_output.joints)[:, 0]
                lwrists = to_numpy(lh_output.joints)[:, 0]
                
                mesh_path = os.path.join(self.grab_path, '..', seq_data.object.object_mesh)
                temp_mesh = trimesh.load(mesh_path, process=False)
                rots = R.from_rotvec(np.array(object_data['global_orient']))
                transl = np.array(object_data['transl'])
                rwrists_cano = np.matmul((rwrists-transl)[:, None], rots.inv().as_matrix()).squeeze(1)
                lwrists_cano = np.matmul((lwrists-transl)[:, None], rots.inv().as_matrix()).squeeze(1)
                sdf = SDF(temp_mesh.vertices, temp_mesh.faces)
                right_dist_mask = np.abs(sdf(rwrists_cano)) < self.cfg['dist_thres']
                left_dist_mask = np.abs(sdf(lwrists_cano)) < self.cfg['dist_thres']

                frame_mask = right_contact_mask[np.nonzero(ds_mask)[0]] == 1
                frame_mask &= right_dist_mask
                intervals = [0] + list(np.nonzero(np.diff(frame_mask))[0] + 1)
                intervals.append(intervals[-1]+1)
                for i in range(len(intervals)-1):
                    if frame_mask[intervals[i]] == 1 and intervals[i+1] - intervals[i] >= 30:
                        m = np.zeros(n_frames, dtype=bool)
                        m[intervals[i]:intervals[i+1]] = 1
                        out_npz = filter_npz(npz, rhand_data, lhand_data, object_data,
                            m, right=True)
                        np.savez(os.path.join(save_path, '{}.npz'.format(k)), **out_npz)
                        k += 1

                frame_mask = left_contact_mask[np.nonzero(ds_mask)[0]] == 1
                frame_mask &= left_dist_mask
                intervals = [0] + list(np.nonzero(np.diff(frame_mask))[0] + 1)
                intervals.append(intervals[-1]+1)
                for i in range(len(intervals)-1):
                    if frame_mask[intervals[i]] == 1 and intervals[i+1] - intervals[i] >= 30:
                        m = np.zeros(n_frames, dtype=bool)
                        m[intervals[i]:intervals[i+1]] = 1
                        out_npz = filter_npz(npz, rhand_data, lhand_data, object_data,
                            m, right=False)
                        np.savez(os.path.join(save_path, '{}.npz'.format(k)), **out_npz)
                        k += 1


    def process_sequences(self):
        for split in self.split_seqs.keys():
            if split not in ('test', 'val'):
                continue
            save_path = os.path.join(self.out_path, split)
            os.makedirs(save_path, exist_ok=True)
            print('Processing data for %s split.' % (split))

            for i, sequence in enumerate(tqdm(sorted(
                glob.glob(os.path.join(self.out_path, split+'_pre', '*.npz'))))):
                seq_data = parse_npz(sequence)
                obj_name = seq_data.obj_name
                n_comps  = seq_data.n_comps
                n_frames = seq_data.n_frames

                rhand_data = {'verts': None, 'joints': None, 'global_orient': [],
                    'hand_pose': [], 'transl': [], 'fullpose': []}
                object_data = {'global_orient': [], 'transl': [], 'id': []}
                    
                if seq_data.right:
                    prepare_params(rhand_data, seq_data.rhand.params)
                else:
                    prepare_params(rhand_data, seq_data.lhand.params)
                prepare_params(object_data, seq_data.object.params)
                object_data['id'].extend([self.objname2id[obj_name]]*n_frames)

                if seq_data.right:
                    rhand_temp_path = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
                    rhand_vtemp = np.array(Mesh(filename=rhand_temp_path).v)

                    rh_m = smplx.create(model_path=cfg['model_path'],
                                        model_type='mano',
                                        is_rhand=True,
                                        v_template=rhand_vtemp,
                                        num_pca_comps=n_comps,
                                        flat_hand_mean=True,
                                        batch_size=n_frames)
                else:
                    lhand_temp_path = os.path.join(grab_path, '..', seq_data.lhand.vtemp)
                    lhand_vtemp = np.array(Mesh(filename=lhand_temp_path).v)

                    rh_m = smplx.create(model_path=cfg['model_path'],
                                        model_type='mano',
                                        is_rhand=False,
                                        v_template=lhand_vtemp,
                                        num_pca_comps=n_comps,
                                        flat_hand_mean=True,
                                        batch_size=n_frames)
            

                rh_output = rh_m(
                    hand_pose=to_torch(rhand_data['hand_pose'], torch.float32),
                    global_orient=to_torch(rhand_data['global_orient'], torch.float32),
                    transl=to_torch(rhand_data['transl'], torch.float32),
                )
                rhand_data['verts'] = to_numpy(rh_output.vertices)
                rhand_data['joints'] = np.concatenate([
                    to_numpy(rh_output.joints), rhand_data['verts'][:, MANO_TIPS]
                ], axis=1)

                if not seq_data.right:
                    rhand_data['verts'][..., 0] = -rhand_data['verts'][..., 0]
                    rhand_data['joints'][..., 0] = -rhand_data['joints'][..., 0]
                    rhand_data['transl'][..., 0] = -rhand_data['transl'][..., 0]
                    M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])[None, ...]
                    rhand_data['global_orient'] = np.matmul(np.matmul(M, 
                        R.from_rotvec(np.array(rhand_data['global_orient'])).as_matrix()), M)
                    rhand_data['global_orient'] = R.from_matrix(rhand_data['global_orient']).as_rotvec()
                    object_data['transl'][..., 0] = -object_data['transl'][..., 0]
                    object_data['global_orient'] = np.matmul(np.matmul(M,
                        R.from_rotvec(np.array(object_data['global_orient'])).as_matrix()), M)
                    object_data['global_orient'] = R.from_matrix(object_data['global_orient']).as_rotvec()

                self.compute_sensors(seq_data, rhand_data, object_data, seq_data.right)

                np_data = self.concat_data(rhand_data, object_data, seq_data.right)

                np.save(os.path.join(save_path, '{}.npy'.format(i)), np_data)

    def compute_sensors(self, seq_data, rhand_data, object_data, right=True):
        T = len(object_data['id'])
        mesh_path = os.path.join(self.grab_path, '..', seq_data.object.object_mesh)
        temp_mesh = trimesh.load(mesh_path, process=False)
        if not right:
            temp_mesh.vertices[:, 0] = -temp_mesh.vertices[:, 0]
            temp_mesh.faces = temp_mesh.faces[:, [2, 1, 0]]
        rots = R.from_rotvec(np.array(object_data['global_orient']))
        transl = np.array(object_data['transl'])

        step = self.framerate // 10
        past_fut_ind = [list(range(i-step*10, i+step*11, step)) for i in range(T)]
        past_fut_ind = [max(0, min(T-1, i)) for l in past_fut_ind for i in l]
        past_fut_wrists = rhand_data['joints'][:, 0][past_fut_ind]
        past_fut_ori = R.from_rotvec(rhand_data['global_orient'][past_fut_ind])
        past_fut_transl = transl[past_fut_ind]
        past_fut_rots = rots[past_fut_ind]

        past_fut_wrists_cano = np.matmul((past_fut_wrists-past_fut_transl)[:, None],
            past_fut_rots.inv().as_matrix()).squeeze(1)

        sdf = SDF(temp_mesh.vertices, temp_mesh.faces)
        closest_points = temp_mesh.vertices[sdf.nn(past_fut_wrists_cano)]
        signed_dists = sdf(past_fut_wrists_cano)
        dists = np.abs(signed_dists[10::21])
        rhand_data['dist'] = dists    

        displacements = np.matmul(closest_points[:, None], past_fut_rots.as_matrix()).squeeze(1) + \
            past_fut_transl - past_fut_wrists
        displacements = np.matmul(displacements[:, None], R.from_rotvec(np.repeat(rhand_data['global_orient'], 21, axis=0)).as_matrix()).squeeze(1)
        rhand_data['disp'] = displacements
        rhand_data['traj'] = past_fut_wrists - \
            np.repeat(rhand_data['joints'][:, 0], 21, axis=0)
        rhand_data['traj'] = np.matmul(rhand_data['traj'][:, None],
            R.from_rotvec(np.repeat(rhand_data['global_orient'], 21, axis=0)).as_matrix()).squeeze(1)
        rhand_data['ori_traj'] = (past_fut_ori * \
            R.from_rotvec(np.repeat(rhand_data['global_orient'], 21, axis=0))
            ).as_matrix().reshape(-1, 9)[:, :6]

        obj_crop_pc = []
        obj_crop_normal = []
        for i in range(T):
            obj_mesh = trimesh.Trimesh(
                vertices=np.matmul(temp_mesh.vertices, rots[i].as_matrix()) + transl[i][None, :],
                faces=temp_mesh.faces,
                process=False)
            bb_center = np.array([-0.12, -0.12, 0], dtype=np.float32)
            bb_center = np.matmul(bb_center[None, :],
                R.from_rotvec(rhand_data['global_orient'][i]).as_matrix().transpose(1, 0))
            bb_center = bb_center + rhand_data['joints'][i, 0][None, :]
            bb_center = bb_center.squeeze(axis=0)

            box_transform = np.eye(4)
            box_transform[:3, 3] = bb_center
            box_transform[:3, :3] = R.from_rotvec(rhand_data['global_orient'][i]).as_matrix()
            box = trimesh.creation.box(extents=[0.24, 0.24, 0.24], transform=box_transform)

            obj_crop_mesh = obj_mesh.slice_plane(box.facets_origin, -box.facets_normal)
            if dists[i] <= 0.2 and len(obj_crop_mesh.vertices) > 0:
                verts_sampled, face_ind = obj_crop_mesh.sample(2000, return_index=True)
                vn_sampled = obj_crop_mesh.face_normals[face_ind]

                obj_pc = np.matmul(
                    verts_sampled - rhand_data['joints'][i, 0][None, :],
                    R.from_rotvec(rhand_data['global_orient'][i]).as_matrix())
                vn_sampled = np.matmul(vn_sampled, R.from_rotvec(rhand_data['global_orient'][i]).as_matrix())
            else:
                obj_pc = np.zeros((2000, 3))
                vn_sampled = np.zeros((2000, 3))
            obj_crop_pc.append(obj_pc)
            obj_crop_normal.append(vn_sampled)
        object_data['pc_crop'] = np.stack(obj_crop_pc)
        object_data['normal_crop'] = np.stack(obj_crop_normal)

        verts_sampled, face_ind = temp_mesh.sample(4000, return_index=True)
        vn_sampled = temp_mesh.face_normals[face_ind]
        
        object_data['pc'] = np.matmul(verts_sampled[None, :], rots.as_matrix()) + \
            transl[:, None, :]
        object_data['normal'] = np.matmul(vn_sampled[None, :], rots.as_matrix())

        object_data['vel'] = np.matmul(
            object_data['pc'][1:] - object_data['pc'][:-1],
            R.from_rotvec(rhand_data['global_orient']).as_matrix()[:-1])
        object_data['vel'] = np.concatenate([object_data['vel'],
            np.zeros((1, object_data['vel'].shape[1], 3))], axis=0)

        object_data['pc'] = np.matmul(
            object_data['pc'] - rhand_data['joints'][:, 0][:, None, :],
            R.from_rotvec(rhand_data['global_orient']).as_matrix()) 
        object_data['normal'] = np.matmul(object_data['normal'],
            R.from_rotvec(rhand_data['global_orient']).as_matrix())

        rhand_data['joints'] = np.matmul(
            rhand_data['joints'] - rhand_data['joints'][:, 0][:, None, :],
            R.from_rotvec(rhand_data['global_orient']).as_matrix())         

    def concat_data(self, rhand_data, object_data, right):
        rhand_verts = rhand_data['verts'].reshape(-1, 2334)
        rhand_joints = rhand_data['joints'].reshape(-1, 63)
        rhand_global_orient = rhand_data['global_orient']
        rhand_pose = rhand_data['hand_pose']
        rhand_fullpose = rhand_data['fullpose']
        rhand_transl = rhand_data['transl']
        rhand_dist = rhand_data['dist']
        rhand_disp = rhand_data['disp'].reshape(-1, 63)
        rhand_traj = rhand_data['traj'].reshape(-1, 63)
        rhand_ori_traj = rhand_data['ori_traj'].reshape(-1, 126)

        object_global_orient = object_data['global_orient']
        object_transl = object_data['transl']
        object_id = object_data['id']
        object_pc_crop = object_data['pc_crop'].reshape(-1, 6000)
        object_normal_crop = object_data['normal_crop'].reshape(-1, 6000)
        object_pc = object_data['pc'].reshape(-1, 12000)
        object_normal = object_data['normal'].reshape(-1, 12000)

        object_vel = object_data['vel'].reshape(-1, 12000)

        right = np.array([right]*len(rhand_verts))

        np_dtype = np.dtype('(2334)f4, (63)f4, (3)f4, (24)f4, (45)f4, (3)f4, f4, (63)f4, (63)f4,' \
            '(126)f4, (3)f4, (3)f4, i4, (6000)f4, (6000)f4, (12000)f4, (12000)f4, ?, (12000)f4', align=True)
        np_data = list(zip(rhand_verts, rhand_joints, rhand_global_orient, rhand_pose,
            rhand_fullpose, rhand_transl, rhand_dist, rhand_disp, rhand_traj, rhand_ori_traj,
            object_global_orient, object_transl, object_id, object_pc_crop, object_normal_crop,
            object_pc, object_normal, right, object_vel))
        np_data = np.array(np_data, dtype=np_dtype)

        return np_data

    def filter_contact_frames(self, seq_data, right=True):
        obj_contact = seq_data['contact']['object']

        if right == True:
            rh_frame_mask = ~(((obj_contact == 21) | ((obj_contact >= 26) & (obj_contact <= 40))).any(axis=1))
            return rh_frame_mask
        else:
            lh_frame_mask = ~(((obj_contact == 22) | ((obj_contact >= 41) & (obj_contact <= 55))).any(axis=1))
            return lh_frame_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    grab_path = 'GRAB/grab'
    out_path = 'grab_processed_gears'
    model_path = 'smplx'

    grab_splits = { 'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                    'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                    'train': []}

    cfg = {
        'hand': 'right',

        'splits':grab_splits,

        'grab_path': grab_path,
        'out_path': out_path,

        'model_path': model_path,

        'framerate': 30,

        'opt': opt,

        'dist_thres': 0.5
    }

    GRAB_Preprocessing(cfg)