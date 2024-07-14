import numpy as np
import torch
import os, argparse
from torch import nn, optim
from data.dataset import GRAB
from model.model import *
from train_fn import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epoch', default=500, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_gpu', default=3, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_worker', default=8, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume_ckpt', default=70, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_printoptions(precision=6)

    obj_feat_dim = 64
    hand_feat_dim = 21 * 12
    joint_latent_dim = 128
    fc_latent_dim = 512
    fc_output_dim = 3
    window_size = 30
    dropout = 0.1
    num_heads = 4
    num_blocks = 2
    obj_enc_conv_size = [6, 64, 128]
    init_hid_dims = [128, 256, 128]
    joint_fc_dims = [3, 64, 128]
    joint_feat_extractor_conv = [7, 64, 128, 128]

    device = torch.device('cuda:0')

    obj_encoder = ObjectEncoder(obj_feat_dim, obj_enc_conv_size)
    hand_initializer = HandInitializer(hand_feat_dim, obj_feat_dim, init_hid_dims)
    joint_embedder = SimpleFullyConnected(joint_fc_dims)
    joint_feat_extractor = JointFeatExtractor(joint_latent_dim, joint_feat_extractor_conv)
    st_transformer = STTransformer(window_size, joint_latent_dim*2, fc_latent_dim,
        fc_output_dim, dropout, num_heads, num_blocks)

    device_ids = list(range(args.num_gpu))
    hand_synthesizer = nn.DataParallel(HandSynthesizerCano(obj_encoder, hand_initializer,
        joint_embedder, joint_feat_extractor, st_transformer, window_size),
        device_ids=device_ids).to(device)

    if args.resume:
        resume_path = os.path.join(args.ckpt_path, '{}.pth'.format(args.resume_ckpt))
        ckpt = torch.load(resume_path, map_location=device)
        hand_synthesizer.module.load_state_dict(ckpt['model'])

    # setup optimizer
    opt = optim.Adam(list(hand_synthesizer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, eta_min=5e-5)

    if args.resume:
        opt.load_state_dict(ckpt['opt'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])

    # prepare data
    train_set = GRAB(args.data_path, 'train', step_size=15)
    vald_set = GRAB(args.data_path, 'val', step_size=15)
    test_set = GRAB(args.data_path, 'test', step_size=15)
    ###################
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_worker, drop_last=True)
    vald_loader = torch.utils.data.DataLoader(vald_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_worker, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_worker, drop_last=True)

    if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    train_model(hand_synthesizer, train_loader, vald_loader, test_loader,
        opt, lr_scheduler, device, args)
