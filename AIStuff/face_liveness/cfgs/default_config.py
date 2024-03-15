import os
from datetime import datetime

import torch
from easydict import EasyDict
from src.bioauth_ml.face_liveness.tools.utility import get_kernel


def get_width_height(patch_info):
    """ get the width and height from the filename """
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def get_default_config():
    """ set the default configuration """
    conf = EasyDict()

    conf.lr = 1e-1
    # [9, 13, 15]
    conf.milestones = [10, 15, 22]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 25
    conf.momentum = 0.9
    conf.batch_size = 64

    # model
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataloader
    conf.train_root_path = '../datasets/rgb_image'

    # save file path
    conf.snapshot_dir_path = './logs/snapshot'

    # log path
    conf.log_path = './logs/jobs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 30

    return conf


def update_config(args, conf):
    """ update the configuration """
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = f"cuda:{conf.devices[0]}" if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = f'Anti_Spoofing_{args.patch_info}'
    log_path = f'{conf.log_path}/{job_name}/{current_time}'
    snapshot_dir = f'{conf.snapshot_dir_path}/{job_name}'

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
