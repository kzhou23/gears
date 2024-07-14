import numpy as np
import torch
import os, time
from torch import nn, optim
from torch.nn import functional as F

mse_loss = nn.MSELoss()

def model_loss(model, data):
    rhand_traj, obj_crop, obj_full, rhand_joint_gt = data
    rhand_joint_gt = rhand_joint_gt.view(rhand_joint_gt.size(0), rhand_joint_gt.size(1), -1, 3)
    rhand_joint_gt = rhand_joint_gt[:, :, 1:]
    rhand_joint_init_pred, disp_pred = model(rhand_traj, obj_crop, obj_full)
    init_joint_loss = mse_loss(rhand_joint_init_pred, rhand_joint_gt)
    joint_disp_loss = mse_loss(rhand_joint_init_pred.detach()+disp_pred, rhand_joint_gt)
    return init_joint_loss, joint_disp_loss


def train_fn_iter(model, data, opt):
    opt.zero_grad()

    init_joint_loss, joint_disp_loss = model_loss(model, data)
    loss = 0.5 * init_joint_loss + 0.5 * joint_disp_loss
    loss.backward()
    opt.step()

    return init_joint_loss.item(), joint_disp_loss.item()


def eval_fn_iter(model, data):
    with torch.no_grad():
        init_joint_loss, joint_disp_loss = model_loss(model, data)
    
    return init_joint_loss.item(), joint_disp_loss.item()


def train_model(model, train_loader, vald_loader, test_loader, opt, lr_scheduler, device, args):
    num_epochs = args.num_epoch
    save_path = args.ckpt_path

    if args.resume:
        print('Training resumed', flush=True)
    else:
        print('Training started', flush=True)

    start_epoch = 1 if not args.resume else args.resume_ckpt+1

    for epoch in range(start_epoch, num_epochs+1):
        t1 = time.time()

        model.train()

        train_loss_dict = {'init_joint_loss': 0, 'joint_disp_loss': 0}
        vald_loss_dict = {'init_joint_loss': 0, 'joint_disp_loss': 0}
        test_loss_dict = {'init_joint_loss': 0, 'joint_disp_loss': 0}

        if not args.eval_only:
            for data in train_loader:
                data = [d.to(device) for d in data]
                init_joint_loss, joint_disp_loss = train_fn_iter(model, data, opt)
                train_loss_dict['init_joint_loss'] += init_joint_loss * 100
                train_loss_dict['joint_disp_loss'] += joint_disp_loss * 100

            lr_scheduler.step(epoch-1)
            t2 = time.time()
            print(t2-t1, flush=True)

        # validation
        model.eval()

        for data in vald_loader:
            data = [d.to(device) for d in data]
            init_joint_loss, joint_disp_loss = eval_fn_iter(model, data)
            vald_loss_dict['init_joint_loss'] += init_joint_loss * 100
            vald_loss_dict['joint_disp_loss'] += joint_disp_loss * 100

        for data in test_loader:
            data = [d.to(device) for d in data]
            init_joint_loss, joint_disp_loss = eval_fn_iter(model, data)
            test_loss_dict['init_joint_loss'] += init_joint_loss * 100
            test_loss_dict['joint_disp_loss'] += joint_disp_loss * 100


        print('====> Epoch {}/{}: Training'.format(epoch, num_epochs), flush=True)
        if not args.eval_only:    
            for term in train_loss_dict:
                print('\t{}: {:.5f}'.format(term, train_loss_dict[term] / len(train_loader)), flush=True)

        print('                   Validation', flush=True)

        for term in vald_loss_dict:
            print('\t{}: {:.5f}'.format(term, vald_loss_dict[term] / len(vald_loader)), flush=True)

        print('                   Test', flush=True)

        for term in test_loss_dict:
            print('\t{}: {:.5f}'.format(term, test_loss_dict[term] / len(test_loader)), flush=True)
     

        checkpoint = {
            'model': model.module.state_dict(),
            'opt': opt.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }   
        torch.save(checkpoint, os.path.join(save_path, '{}.pth'.format(epoch)))
