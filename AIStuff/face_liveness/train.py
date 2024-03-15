"""
@author: XiangLan   
@file: train.py
@time: 2022/12/12 10:05
@desc: include the training and inference module for face liveness detection
"""

import sys
sys.path.append('..')

from argparse import ArgumentParser, Namespace
import safitty
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_model import LightningFASNet, LightningOpticalFASNet


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--configs", default="./cfgs/train_config.yml", required=True)
    parser.add_argument("--opt", action='store_true', required=True)
    parser.add_argument("--prefix", default="2.7_128x128_MiniFASNet", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    if not args.opt:
        model = LightningFASNet(hparams=configs)
    else:
        model = LightningOpticalFASNet(hparams=configs)

    logger = loggers.TensorBoardLogger('./logs/')
    checkpoint_callback = ModelCheckpoint(
        # monitor='val_loss',
        dirpath='./lightning_logs/',
        filename=f'{args.prefix}' + '{epoch:02d}'
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(callbacks=[checkpoint_callback, lr_monitor], gpus=1, max_epochs=50, logger=logger)
    trainer.fit(model)
