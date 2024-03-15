"""
@author: XiangLan
@file: train_mnet.py
@time: 2022/12/24 11:00
@desc: train the model for motion analysis based liveness detection
"""
import sys
sys.path.append('../../..')

import argparse
import os
import random
import numpy as np
import torch
import tensorboardX
from torch import nn

from src.bioauth_ml.face_liveness.dataloader.read_3d_data import UCF101, get_default_video_loader
from src.bioauth_ml.face_liveness.dataloader.spatial_transforms import Compose, Normalize, Scale, ToTensor
from src.bioauth_ml.face_liveness.dataloader.temporal_transforms import LoopPadding, TemporalRandomCrop
from src.bioauth_ml.face_liveness.dataloader.target_transforms import ClassLabel
from src.bioauth_ml.face_liveness.model.FASNet3D import MNet
from src.bioauth_ml.face_liveness.tools.utility import AverageMeter, calculate_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ucf101', help='dataset type')
parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str,
                    help='Root directory path of data')
parser.add_argument('--video_path', default='./datasets/motion_analysis/image_data/', type=str,
                    help='Directory path of Videos')
parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
parser.add_argument('--n_val_samples', default=3, type=int,
                    help='Number of validation samples for each activity')
parser.add_argument('--annotation_path', default='./datasets/motion_analysis/annotation/'
                                                 'ucf101_01.json', type=str, help='Annotation file path')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--sample_size', default=128, type=int, help='Height and width of inputs')
parser.add_argument('--log_interval', default=10, type=int,
                    help='Log interval for showing training loss')
parser.add_argument('--save_interval', default=2, type=int, help='Model saving interval')
parser.add_argument('--model', default='MNet', type=str, help='(MNet | MNet_attn |')
parser.add_argument('--n_classes', default=2, type=int, help='Number of classes '
                    '(activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
parser.add_argument('--lr_rate', default=1e-3, type=float,
                    help='Initial learning rate (divided by 10 while training by lr scheduler)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
parser.add_argument('--no_mean_norm', default=False, action='store_true',
                    help='If true, inputs are not normalized by mean.')
parser.add_argument('--mean_dataset', default='activitynet', type=str,
                    help='dataset for mean values of mean subtraction (activitynet | kinetics)')
parser.add_argument('--use_cuda', default=True, action='store_true', help='If true, use GPU.')
parser.add_argument('--nesterov', default=False, action='store_true', help='Nesterov momentum')
parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
parser.add_argument('--lr_patience', default=10, type=int,
                    help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
parser.add_argument('--resume_path', default="../face_liveness/checkpoints/anti_spoof_models_v2"
                                             "/fasnet_3d/org_1_150x150_MNet.pth", type=str, help='Resume training')
parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of threads for multi-thread loading')
parser.add_argument('--norm_value', default=1, type=int,
                    help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
parser.add_argument('--std_norm', default=False, action='store_true',
                    help='If true, inputs are normalized by standard deviation.')


def get_mean(norm_value=255, dataset='activitynet'):
    """ get mean value of the source image """
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    if dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]
    return None


def get_std(norm_value=255):
    """ get std value of the source image """
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    """ get dataloader for training (UCF101 dataloader) """
    training_data = None
    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    """ get dataloader for validation (UCF101 dataloader) """
    validation_data = None
    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    """ train the model every epochs """
    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print(f'Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(data_loader.dataset)} '
                  f'({100. * (batch_idx + 1) / len(data_loader):.0f}%)]\tLoss: {avg_loss:.6f}')
            train_loss = 0.0

    print(f'Train set ({len(data_loader.dataset)} samples): '
          f'Average loss: {losses.avg:.4f}\tAcc: {accuracies.avg * 100:.4f}%')

    return losses.avg, accuracies.avg


def val_epoch(model, data_loader, criterion, device):
    """ validate the model every epochs """
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    print(f'Validation set ({len(data_loader.dataset)} samples): Average loss: '
          f'{losses.avg:.4f}\tAcc: {accuracies.avg * 100:.4f}%')
    return losses.avg, accuracies.avg


def get_mnet_score(video_path):
    """ get score with the mnet model """
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = MNet()
    model = model.to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(args.resume_path)
    else:
        checkpoint = torch.load(args.resume_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(args.mean, [1, 1, 1])
    else:
        norm_method = Normalize(args.mean, args.std)

    spatial_transform = Compose([
        Scale((args.sample_size, args.sample_size)),
        # CenterCrop(opt.sample_size),
        ToTensor(args.norm_value), norm_method
    ])

    frame_loader = get_default_video_loader
    default_loader = frame_loader()
    input_data = default_loader(video_path, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    input_data = [spatial_transform(img) for img in input_data]
    input_data = torch.stack(input_data, 0)
    input_data = torch.unsqueeze(input_data, 0)

    output = model(input_data)
    output = torch.softmax(output, dim=-1)
    predict = output.to("cpu").detach().numpy()
    return predict[0][0]


def resume_model(opt, model, optimizer):
    """ resume model with the pretrained model """
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model Restored from Epoch {checkpoint['epoch']}")
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def get_loaders(opt):
    """ make dataloaders for train and validation sets """
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        # crop_method,
        Scale((opt.sample_size, opt.sample_size)),
        # RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)

    # validation loader
    spatial_transform = Compose([
        Scale((opt.sample_size, opt.sample_size)),
        # CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    target_transform = ClassLabel()
    temporal_transform = LoopPadding(16)
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True)
    return train_loader, val_loader


def main():
    """" main process """
    args = parser.parse_args()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # CUDA for PyTorch
    device = torch.device(f"cuda:{args.gpu}" if args.use_cuda else "cpu")

    # tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

    # define model
    model = MNet()
    model = model.to(device)
    # get data loaders
    train_loader, val_loader = get_loaders(args)

    # define optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr_rate, weight_decay=args.weight_decay)

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    criterion = nn.CrossEntropyLoss()

    # resume model
    if args.resume_path:
        start_epoch = resume_model(args, model, optimizer)
    else:
        start_epoch = 1

    # start training
    for epoch in range(start_epoch, args.n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, args.log_interval, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        # saving weights to checkpoint
        if epoch % args.save_interval == 0:
            # scheduler.step(val_loss)
            # write summary
            summary_writer.add_scalar('losses/train_loss', train_loss, global_step=epoch)
            summary_writer.add_scalar('losses/val_loss', val_loss, global_step=epoch)
            summary_writer.add_scalar('acc/train_acc', train_acc * 100, global_step=epoch)
            summary_writer.add_scalar('acc/val_acc', val_acc * 100, global_step=epoch)

            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, os.path.join('logs', f'{args.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
            print(f"Epoch {epoch} model saved!")


if __name__ == "__main__":
    main()
