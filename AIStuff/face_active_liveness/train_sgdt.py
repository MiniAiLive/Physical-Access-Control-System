import sys
sys.path.append('..')

import os
import random
import argparse
import shutil
import numpy as np
from PIL import Image

import torchvision.transforms as standard_transforms
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from src.data_io import dataset_sgdt as dataset
from src.model_lib.SGDTSingleNet import Net

train_live_rgb_dir = './datasets/rgb_image/1.2_144x144/0'
train_fake_rgb_dir = './datasets/rgb_image/1.2_144x144/1'

test_live_rgb_dir = './datasets/rgb_image/1.2_144x144/0'
test_fake_rgb_dir = './datasets/rgb_image/1.2_144x144/1'

parser = argparse.ArgumentParser(description='PyTorch Liveness Training')
parser.add_argument('-s', '--scale', default=1.0, type=float,
                    metavar='N', help='net scale')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='./saved_logs/weights_sgdt/ft1_54_checkpoint_99.878.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DepthFocalLoss(nn.Module):
    def __init__(self, gamma=1, eps=1e-7):
        super(DepthFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = (loss) ** self.gamma
        return loss.mean()


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net(args.scale)
    # net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    print(net)
    print("start load train data")
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    target_transform = standard_transforms.Compose([
        standard_transforms.Resize((32, 32)),
        standard_transforms.ToTensor()
    ])

    train_set = dataset.Dataset('train', train_live_rgb_dir, None, train_fake_rgb_dir,
                                random_transform=random_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
                              random_transform=random_input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    criterion_class = FocalLoss()
    criterion_depth = DepthFocalLoss()
    optimizer = torch.optim.Adam(net.parameters())

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # g_err_rate = checkpoint['best_err_rate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(device, net, val_loader, args.arch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(device, net, train_loader, criterion_depth, criterion_class, optimizer, epoch)
        acc = validate(device, net, val_loader)

        filename = f'ft1_{epoch+1}_checkpoint_{acc:0.3f}.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename)


def validate(device, net, val_loader, depth_dir='./depth_predict'):
    try:
        shutil.rmtree(depth_dir)
    except:
        pass
    try:
        os.makedirs(depth_dir)
    except:
        pass
    toImage = standard_transforms.ToPILImage(mode='L')
    net.eval()

    corrects = 0
    for i, data in enumerate(val_loader):
        input, label = data
        input = input.cuda(device)
        class_ret = net(input)

        # class_ret = class_ret.detach().cpu()
        # image = toImage(out_depth)
        class_output = torch.softmax(class_ret, dim=-1)
        _, predicted = torch.max(class_output.data, 1)

        label = label.to('cpu').detach().numpy()
        predicted = predicted.to('cpu').detach().numpy()
        corrects += (label == predicted).sum()

    valid_acc = 100*corrects/len(val_loader.dataset)
    print(f"Accuracy on valid dataset: {valid_acc:0.3f}")

    return valid_acc


def conv_loss(device, out_depth, label_depth, criterion_depth):
    loss0 = criterion_depth(out_depth, label_depth)
    filters1 = torch.tensor([[[[-1, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters2 = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters3 = torch.tensor([[[[0, 0, -1], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters4 = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters5 = torch.tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters6 = torch.tensor([[[[0, 0, 0], [0, 1, 0], [-1, 0, 0]]]], dtype=torch.float).cuda(device)
    filters7 = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=torch.float).cuda(device)
    filters8 = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, -1]]]], dtype=torch.float).cuda(device)

    loss1 = criterion_depth(nn.functional.conv2d(out_depth, filters1, padding=1),
                            nn.functional.conv2d(label_depth, filters1, padding=1))
    loss2 = criterion_depth(nn.functional.conv2d(out_depth, filters2, padding=1),
                            nn.functional.conv2d(label_depth, filters2, padding=1))
    loss3 = criterion_depth(nn.functional.conv2d(out_depth, filters3, padding=1),
                            nn.functional.conv2d(label_depth, filters3, padding=1))
    loss4 = criterion_depth(nn.functional.conv2d(out_depth, filters4, padding=1),
                            nn.functional.conv2d(label_depth, filters4, padding=1))
    loss5 = criterion_depth(nn.functional.conv2d(out_depth, filters5, padding=1),
                            nn.functional.conv2d(label_depth, filters5, padding=1))
    loss6 = criterion_depth(nn.functional.conv2d(out_depth, filters6, padding=1),
                            nn.functional.conv2d(label_depth, filters6, padding=1))
    loss7 = criterion_depth(nn.functional.conv2d(out_depth, filters7, padding=1),
                            nn.functional.conv2d(label_depth, filters7, padding=1))
    loss8 = criterion_depth(nn.functional.conv2d(out_depth, filters8, padding=1),
                            nn.functional.conv2d(label_depth, filters8, padding=1))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    return loss


def train(device, net, train_loader, criterion_depth, criterion_class, optimizer, epoch):
    losses_class = AverageMeter()
    net.train()
    for i, data in enumerate(train_loader):
        input, label = data
        input = input.cuda(device)
        label = label.cuda(device)
        class_ret = net(input)

        loss_class = criterion_class(class_ret, label)
        losses_class.update(loss_class.data, input.size(0))
        loss = loss_class

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("epoch:{} batch:{} class loss:{:f} class avg loss:{:f}".format(
                epoch, i, loss_class.data.cpu().numpy(), losses_class.avg.cpu().numpy()))


def get_sgdt_score(image):
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net(args.scale)
    net = net.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])

    net.eval()
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1., 1.)),
        standard_transforms.ToTensor(),
        normalize
    ])

    input = random_input_transform(Image.fromarray(image))
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    output = net(input)
    output = torch.softmax(output, dim=-1)
    predict = output.to("cpu").detach().numpy()

    return predict[0][1]


if __name__ == '__main__':
    main(parser.parse_args())