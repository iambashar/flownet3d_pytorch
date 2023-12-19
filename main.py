from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from data import ModelNet40, SceneflowDataset, SceneflowDataset_Kitti
from model import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import util
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import cv2

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    num_examples = 0
    for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
        pc1, pc2, color1, color2, flow, mask1 = None, None, None, None, None, None
        if args.dataset == 'SceneflowDataset':
            pc1, pc2, color1, color2, flow, mask1 = data
            flow = flow.cuda().transpose(2,1).contiguous()
            mask1 = mask1.cuda().float()
        elif args.dataset == 'Kitti':
            pc, co = data
            pc1 = pc[:, 0, :, :]
            pc2 = pc[:, 1, :, :]
            color1 = co[:, 0, :, :]
            color2 = co[:, 1, :, :]
            pc1 = pc1.float()
            pc2 = pc2.float()
            color1 = color1.float()
            color2 = color2.float()
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()


        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
        # loss = torch.mean(mask1 * torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(flow_pred.detach().cpu().numpy(), flow, mask1)
        total_epe += epe_3d * batch_size
        total_acc3d += acc_3d * batch_size
        total_acc3d_2+=acc_3d_2*batch_size
        # print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))

        # total_loss += loss.item() * batch_size
        

    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples


def _train_one_epoch(args, net, train_loader, opt):   # FlowNet3D
    net.train()
    num_examples = 0
    total_loss = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda().transpose(2,1).contiguous()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)
        loss = torch.mean(mask1 * torch.sum((flow_pred - flow) ** 2, 1) / 2.0)
        loss.backward()

        opt.step()
        total_loss += loss.item() * batch_size

        # if (i+1) % 100 == 0:
        #     print("batch: %d, mean loss: %f" % (i, total_loss / 100 / batch_size))
        #     total_loss = 0
    return total_loss * 1.0 / num_examples

def pcd_vis (grouped_xyz):
    grouped_xyz_vis = grouped_xyz.cpu().detach().numpy()
    grouped_xyz_vis = grouped_xyz_vis[0]
    grouped_xyz_vis = grouped_xyz_vis.transpose(1, 0)
    grouped_xyz_vis = grouped_xyz_vis[:, :3]
    grouped_xyz_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grouped_xyz_vis))
    grouped_xyz_vis = np.asarray(grouped_xyz_vis.points)
    return grouped_xyz_vis

def train_one_epoch(args, net, train_loader, opt):   # Just Go with Flow: Cycle-loss + KNN loss
    net.train()
    num_examples = 0
    total_loss = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        pc1, pc2, color1, color2, flow, mask1 = None, None, None, None, None, None
        if args.dataset == 'SceneflowDataset':
            pc1, pc2, color1, color2, flow, mask1 = data
            flow = flow.cuda().transpose(2,1).contiguous()
            mask1 = mask1.cuda().float()
        elif args.dataset == 'Kitti':
            pc, co = data
            pc1 = pc[:, 0, :, :]
            pc2 = pc[:, 1, :, :]
            color1 = co[:, 0, :, :]
            color2 = co[:, 1, :, :]
            pc1 = pc1.float()
            pc2 = pc2.float()
            color1 = color1.float()
            color2 = color2.float()


        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        forward_flow_pred = net(pc1, pc2, color1, color2)

        forward_pc1 = pc1 + forward_flow_pred

        _, knn_idx = util.knn_point(1, pc2.view(batch_size, -1, 3), forward_pc1.view(batch_size, -1, 3))
        knn_idx = knn_idx[:, :, 0]


        grouped_xyz = util.index_points(pc2.view(batch_size, -1, 3), knn_idx)
        grouped_xyz = grouped_xyz.view(batch_size, 3, -1)              

        loss_nn = torch.mean(torch.sum((grouped_xyz - forward_pc1) ** 2, 1) / 2.0)

        anchor_point = (grouped_xyz + forward_pc1) * 0.5

        backward_flow_pred = net(anchor_point, pc1, color2, color1)

        backward_pc2 = anchor_point + backward_flow_pred

        # Display using matplotlib:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        grouped_xyz_vis = pcd_vis(grouped_xyz)
        pc1_vis = pcd_vis(pc1)
        forward_pc1_vis = pcd_vis(forward_pc1)
        pc2_vis = pcd_vis(pc2)
        # anchor_point_vis = pcd_vis(anchor_point)
        # backward_pc2_vis = pcd_vis(backward_pc2)
        # ax.scatter3D(backward_pc2_vis[:, 0], backward_pc2_vis[:, 1], backward_pc2_vis[:, 2], c='r', marker='o')
        # ax.scatter3D(anchor_point_vis[:, 0], anchor_point_vis[:, 1], anchor_point_vis[:, 2], c='r', marker='o')
        # ax.scatter3D(grouped_xyz_vis[:, 0], grouped_xyz_vis[:, 1], grouped_xyz_vis[:, 2], c='r', marker='o')
        # ax.scatter3D(pc1_vis[:, 0], pc1_vis[:, 1], pc1_vis[:, 2], c='b', marker='o')
        # ax.scatter3D(pc2_vis[:, 0], pc2_vis[:, 1], pc2_vis[:, 2], c='r', marker='o')
        # ax.scatter3D(forward_pc1_vis[:, 0], forward_pc1_vis[:, 1], forward_pc1_vis[:, 2], c='g', marker='o')

        gt_flow = pc1 + flow
        gt_flow_vis = pcd_vis(gt_flow)

        pcd_np = np.asarray(pc1_vis)
        pc2_np = np.asarray(pc2_vis)
        gt_flow_np = np.asarray(gt_flow_vis)
        forward_pc1_np = np.asarray(forward_pc1_vis)
        ax.quiver(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2], gt_flow_np[:, 0], gt_flow_np[:, 1], gt_flow_np[:, 2], length=0.1, color = 'r')
        # ax.quiver(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2], pc2_np[:, 0], pc2_np[:, 1], pc2_np[:, 2], length=0.1)
        ax.quiver(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2], forward_pc1_np[:, 0], forward_pc1_np[:, 1], forward_pc1_np[:, 2], length=0.1, color = 'g')

        # ax.set_title("Scene Flow Vectors")
        plt.show()
        sys.exit()

        loss_cycle = torch.mean(torch.sum((pc1 - backward_pc2) ** 2, 1) / 2.0)

        loss = loss_nn + loss_cycle

        loss.backward()

        opt.step()
        total_loss += loss.item() * batch_size

        # if (i+1) % 100 == 0:
        #     print("batch: %d, mean loss: %f" % (i, total_loss / 100 / batch_size))
        #     total_loss = 0
    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, boardio, textio):

    test_loss, epe, acc, acc_2 = test_one_epoch(args, net, test_loader)

    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f'%(test_loss, epe, acc, acc_2))


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma = 0.7)

    best_test_loss = np.inf
    for epoch in range(args.epochs):
        textio.cprint('==epoch: %d, learning rate: %f=='%(epoch, opt.param_groups[0]['lr']))
        train_loss = train_one_epoch(args, net, train_loader, opt)
        textio.cprint('mean train EPE loss: %f'%train_loss)
        boardio.add_scalar('train/loss', train_loss, epoch)
        boardio.add_scalar('train/lr', opt.param_groups[0]['lr'], epoch)

        test_loss, epe, acc, acc_2 = test_one_epoch(args, net, test_loader)
        textio.cprint('mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f'%(test_loss, epe, acc, acc_2))
        boardio.add_scalar('test/loss', test_loss, epoch)
        boardio.add_scalar('test/epe', epe, epoch)
        boardio.add_scalar('test/acc', acc, epoch)
        boardio.add_scalar('test/acc_2', acc_2, epoch)

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
        
        scheduler.step()
        # if torch.cuda.device_count() > 1:
        #     torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        # else:
        #     torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        # gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'],
                        help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--dataset', type=str, default='SceneflowDataset',
                        choices=['SceneflowDataset', 'Kitti'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='Data/data_processed_maxcut_35_20k_2k_8192', metavar='N',
                        help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to train ')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')                        

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='runs/' + args.exp_name)
    # boardio = []
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    elif args.dataset == 'SceneflowDataset':
        train_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, partition='train'),
            batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    elif args.dataset == 'Kitti':
        train_loader = DataLoader(
            SceneflowDataset_Kitti(npoints=args.num_points, train=True),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(
            SceneflowDataset_Kitti(npoints=args.num_points, train=False),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    else:
        raise Exception("not implemented")

    if args.model == 'flownet':
        net = FlowNet3D(args).cuda()
        net.apply(weights_init)
        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')

    if args.resume:
        print("Resume training from checkpoint...")
        net.load_state_dict(torch.load(args.model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()