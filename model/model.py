import sys
sys.path.append('.')
sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D
from pytorch3d.ops import ball_query
from utils.ik_solver import ik_solver_mano
from utils.utils import homoify, dehomoify
from manopth.manolayer2 import ManoLayer2

class CustomLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
       x = (x - x.mean(dim=self.dim, keepdim=True)) / torch.sqrt(x.var(dim=self.dim, keepdim=True)+self.eps)
       return x

class ObjectEncoder(nn.Module):
    def __init__(self, obj_feat_dim, conv_size):
        super(ObjectEncoder, self).__init__()
        conv_layers = []
        for i in range(len(conv_size)-1):
            conv_layers.append(nn.Conv1d(conv_size[i], conv_size[i+1], 1))
            conv_layers.append(CustomLayerNorm(dim=1))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        self.latent_fc = nn.Linear(conv_size[-1], obj_feat_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        assert x.size(1) == 6, "Wrong input dimension"
        x = self.conv_layers(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.latent_fc(x)
        return x


class HandInitializer(nn.Module):
    def __init__(self, hand_feat_dim, obj_feat_dim, init_hid_dims, num_joints=20):
        super(HandInitializer, self).__init__()
        init_hid_dims = [hand_feat_dim+obj_feat_dim] + list(init_hid_dims)
        init_layers = []
        for i in range(len(init_hid_dims)-1):
            init_layers.append(nn.Linear(init_hid_dims[i], init_hid_dims[i+1]))
            init_layers.append(CustomLayerNorm(-1))
            init_layers.append(nn.ReLU())
            init_layers.append(nn.Dropout(0.1))
        init_layers.append(nn.Linear(init_hid_dims[-1], num_joints*3))
        self.layers = nn.Sequential(*init_layers)

    def forward(self, x):
        return self.layers(x)


class JointFeatExtractor(nn.Module):
    def __init__(self, latent_dim, conv_size, num_joints=20):
        super(JointFeatExtractor, self).__init__()

        self.num_joints = num_joints

        conv_layers = []
        for i in range(len(conv_size)-1):
            conv_layers.append(nn.Conv1d(conv_size[i], conv_size[i+1], 1))
            conv_layers.append(CustomLayerNorm(dim=1))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        self.latent_conv = nn.Conv1d(conv_size[-1], latent_dim, 1)

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)
        x = torch.cat([x, mask.view(x.size(0), 1, x.size(2)).float()], dim=1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1), self.num_joints, -1)
        x = (x * mask.unsqueeze(1)).sum(dim=-1)
        local_sum = mask.sum(dim=-1).unsqueeze(1)
        zero_mask = local_sum == 0
        local_sum[zero_mask] = 1
        x = x / (local_sum + 1e-6)
        x[zero_mask.repeat(1, x.size(1), 1)] = 0

        x = self.latent_conv(x)
        return x


class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, return_weights=False):
        if isinstance(sublayer, nn.MultiheadAttention):
            attn_output, _ = sublayer(x, x, x)
            return self.norm(x + self.dropout(attn_output))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


class SimpleFullyConnected(nn.Module):
    def __init__(self, dims):
        super(SimpleFullyConnected, self).__init__()
        assert len(dims) >= 2, 'Need more than two layers for a fully connected network'
        
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                layers.append(nn.LayerNorm((dims[i+1],)))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim, dropout=0):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = PositionalEncoding2D(embed_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x + self.pe(x)
        x = x.permute(1, 2, 0, 3).contiguous()
        return self.dropout(x)


class STTransformer(nn.Module):
    def __init__(self, window_size, embed_dim, fc_latent_dim, fc_output_dim, dropout,
        num_heads, num_blocks):
        super(STTransformer, self).__init__()
        self.num_blocks = num_blocks
        self.sublayers = nn.ModuleList()

        self.pe = PositionalEncoder(window_size, embed_dim)

        for i in range(num_blocks):
            self.sublayers.append(nn.MultiheadAttention(embed_dim, num_heads))
            self.sublayers.append(SimpleFullyConnected([embed_dim, fc_latent_dim, embed_dim]))
            self.sublayers.append(nn.MultiheadAttention(embed_dim, num_heads))
            self.sublayers.append(SimpleFullyConnected([embed_dim, fc_latent_dim, embed_dim]))

        self.normadds = nn.ModuleList(
            [SublayerConnection(embed_dim, dropout) for _ in range(len(self.sublayers))]
        )

        self.output = nn.Linear(embed_dim, fc_output_dim)

    def forward(self, x):
        # x: T * 20 * B * F
        x = self.pe(x)

        temporal_shape = x.size()
        spatial_shape = (x.size(1), x.size(0), x.size(2), x.size(3))
        for i in range(self.num_blocks):
            x = x.view(x.size(0), -1, x.size(3))
            x = self.normadds[4*i](x, self.sublayers[4*i])
            x = self.normadds[4*i+1](x, self.sublayers[4*i+1])

            x = x.view(*temporal_shape).permute(1, 0, 2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(3))
            x = self.normadds[4*i+2](x, self.sublayers[4*i+2])
            x = self.normadds[4*i+3](x, self.sublayers[4*i+3])

            x = x.view(*spatial_shape).permute(1, 0, 2, 3).contiguous()
        
        x = self.output(x).permute(2, 0, 1, 3).contiguous()

        return x


class HandSynthesizerCano(nn.Module):
    def __init__(self, obj_encoder, hand_initializer, joint_embedder, joint_feat_extractor,
        st_transformer, window_size):
        super(HandSynthesizerCano, self).__init__()
        self.obj_encoder = obj_encoder
        self.hand_initializer = hand_initializer
        self.joint_embedder = joint_embedder
        self.joint_feat_extractor = joint_feat_extractor
        self.st_transformer = st_transformer
        self.window_size = window_size

        mano_layer = ManoLayer2(flat_hand_mean=True, side="right",
            mano_root='/BS/kzhou2/static00/smplx/mano/', use_pca=False, center_idx=0)
        mano_axisang = torch.zeros((1, 48), dtype=torch.float32)
        target_shape = torch.zeros((1, 10), dtype=torch.float32)
        _, template_joints, _, = mano_layer(mano_axisang, target_shape)
        self.template_joints = template_joints


    def forward(self, rhand_traj, obj_crop, obj_full):
        batch_size = rhand_traj.size(0)

        obj_crop_feat = self.obj_encoder(obj_crop.view(batch_size*self.window_size, -1, 6))
        obj_crop_feat = obj_crop_feat.view(batch_size, self.window_size, -1)

        feat = torch.cat([rhand_traj, obj_crop_feat], dim=-1)
        rhand_joint_init = self.hand_initializer(feat.view(batch_size*self.window_size, -1))
        rhand_joint_init = rhand_joint_init.view(-1, 20, 3)
        rhand_joint_init_output = rhand_joint_init.view(batch_size, -1, 20, 3)

        rhand_joint_init = torch.cat([torch.zeros(batch_size*self.window_size, 1, 3,
            dtype=torch.float, device=rhand_joint_init.device),
            rhand_joint_init], dim=1)
        
        rhand_joint_init, global_trans = ik_solver_mano(rhand_joint_init,
            self.template_joints.repeat(rhand_joint_init.size(0), 1, 1).to(rhand_joint_init.device))
        rhand_joint_init = rhand_joint_init[:, 1:]
        global_trans = global_trans[:, 1:]

        joint_embed = self.joint_embedder(rhand_joint_init).permute(0, 2, 1).contiguous()

        obj_full = obj_full.view(batch_size*self.window_size, -1, 6)

        ind = ball_query(rhand_joint_init, obj_full[..., :3], K=300, 
            radius=0.025, return_nn=False).idx
        sample_mask = ind != -1
        ind = ind.view(batch_size*self.window_size, -1)
        batch_ind = torch.arange(batch_size*self.window_size).repeat_interleave(ind.size(1))
        sample_obj_points = obj_full[batch_ind.view(*ind.size()), ind]

        sample_obj_points = sample_obj_points.view(batch_size*self.window_size, 20, -1, 6)

        inv_global_trans = torch.inverse(global_trans)

        sample_obj_points_homo = homoify(sample_obj_points[..., :3])
        inv_sample_obj_points_homo = torch.matmul(inv_global_trans, 
            sample_obj_points_homo.transpose(2, 3)).transpose(2, 3)
        cano_sample_obj_points = dehomoify(inv_sample_obj_points_homo)
        cano_sample_obj_normals = torch.matmul(inv_global_trans[..., :3, :3], 
            sample_obj_points[..., 3:6].transpose(2, 3)).transpose(2, 3)

        sample_obj_points = torch.cat([cano_sample_obj_points, cano_sample_obj_normals],
            dim=-1)
        sample_obj_points = sample_obj_points.view(batch_size*self.window_size, -1, 6)

        joint_feat = self.joint_feat_extractor(sample_obj_points, sample_mask)

        joint_feat = torch.cat([joint_embed, joint_feat], dim=1)
        joint_feat = joint_feat.view(batch_size, self.window_size, joint_feat.size(1), -1)

        joint_feat = joint_feat.permute(1, 3, 0, 2).contiguous()
        joint_disp_cano = self.st_transformer(joint_feat)

        joint_disp = torch.matmul(global_trans[..., :3, :3], 
            joint_disp_cano.view(-1, 20, 3, 1)).squeeze(3)
        joint_disp = joint_disp.view(*joint_disp_cano.size())

        joint_disp = joint_disp - rhand_joint_init_output.detach() + \
            rhand_joint_init.view(*joint_disp.size()).detach()

        return rhand_joint_init_output, joint_disp
