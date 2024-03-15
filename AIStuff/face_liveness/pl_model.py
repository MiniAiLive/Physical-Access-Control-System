"""
@author: Xiang
@file: pl_model.py
@time: 2022/12/16 20:22
@desc: include the training and inference module for face liveness detection
"""
import sys
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.dataset import LivenessDataset, OpticalLivenessDataset
from model.FASNetA import FASNetA
from model.FASNetB import FASNetB
from model.FASNetC import FASNetCV1
from model.FeatherNet import FeatherNetA
from model.EfficientNet import EfficientNet
from model.FishNet import Fishnet150
from model.FaceBagNet import FaceBagNet_model_A
from model.OpticalFASNet import optical_fasnetv1
from model.MultiFTNet import MultiFTNet
from tools.utility import get_kernel


class LightningFASNet(pl.LightningModule):
    """ Pytorch Lightning module for face liveness detection """

    def __init__(self, hparams):
        """ initialize the variables """
        super().__init__()
        self.save_hyperparameters(hparams)

        if hparams.net == "FASNetA":
            self.net = FASNetA()
        elif hparams.net == "FASNetB":
            self.net = FASNetB()
        elif hparams.net == "FASNetCV1":
            self.net = FASNetCV1()
        elif hparams.net == "FeatherNetA":
            self.net = FeatherNetA()
        elif hparams.net == "EfficientNet":
            self.net = EfficientNet()
        elif hparams.net == "FishNet":
            self.net = Fishnet150()
        elif hparams.net == "FaceBagNet":
            self.net = FaceBagNet_model_A()
        elif hparams.net == "MiniFASNet":
            self.net = MultiFTNet(embedding_size=128, conv6_kernel=get_kernel(128, 128), num_classes=3, img_channel=3)
        else:
            print("Unable to find model.")
            sys.exit(1)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.ft_criterion = nn.MSELoss()

        self.root_path = hparams.root_dir
        self.train_live_dir = hparams.train_live_dir
        self.train_fake_dir = hparams.train_fake_dir
        self.train_fake_3d_dir = hparams.train_fake_3d_dir

        self.test_live_dir = hparams.test_live_dir
        self.test_fake_dir = hparams.test_fake_dir
        self.test_fake_3d_dir = hparams.test_fake_3d_dir

        self.ft_width = hparams.ft_width
        self.ft_height = hparams.ft_height

    def forward(self, x):
        """ pass model for training """
        return self.net(x)

    def infer(self, x):
        """ pass model for inference """
        outs, _ = self.net(x)
        return outs[-1]

    def calc_losses(self, cls, ft_input, target):
        """ calculate the loss during the training process """
        cls_loss = self.cls_criterion(cls, target)
        if ft_input[0] is None:
            ft_loss = 0
        else:
            ft_loss = self.ft_criterion(ft_input[0], ft_input[1])

        return cls_loss, ft_loss

    def training_step(self, batch, batch_idx):
        """ train the model """
        raw, ft, label = batch
        raw = raw.cuda(self.device)
        label = label.cuda(self.device)
        class_ret, feature_map = self.net(raw)

        cls_loss, ft_loss = self.calc_losses(class_ret, (feature_map, ft), label)
        total_loss = 0.5 * cls_loss + 0.5 * ft_loss
        self.log("train/classification_loss", cls_loss)
        self.log("train/fourier_loss", ft_loss)
        self.log("train/total_loss", total_loss)
        return total_loss

    def training_epoch_end(self, outputs):
        """ get the loss at the end every epochs and save it into the tensorboard """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train/train_avg_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        """ validate the model """
        raw, ft, label = batch

        raw = raw.cuda(self.device)
        class_ret = self.net(raw)

        cls_loss, ft_loss = self.calc_losses(class_ret, (None, ft), label)
        total_loss = 0.5 * cls_loss + 0.5 * ft_loss
        self.log("valid/classification_loss", cls_loss)
        self.log("valid/fourier_loss", ft_loss)
        self.log("valid/total_loss", total_loss)
        return total_loss

    def validation_epoch_end(self, outputs):
        """ get the loss at the end every epochs and save it into the tensorboard """
        avg_loss = torch.stack([x for x in outputs]).mean()
        # _ = np.hstack([output["target"] for output in outputs])
        # scores = np.vstack([output["score"] for output in outputs])[:, 1]
        # metrics_, best_thr, acc = eval_from_scores(scores, targets)
        # acer, apcer, npcer = metrics_
        # roc_auc = metrics.roc_auc_score(targets, scores)
        self.log("valid/val_avg_loss", avg_loss)
        # tensorboard_logs = {
        #     "val_loss": avg_loss,
        #     "val_roc_auc": roc_auc,
        #     "val_acer": acer,
        #     "val_apcer": apcer,
        #     "val_npcer": npcer,
        #     "val_acc": acc,
        #     "val_thr": best_thr,
        # }
        return avg_loss

    def configure_optimizers(self):
        """ configure the optimizer for the training """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        """ load training dataloader """
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        train_set = LivenessDataset('train', self.train_live_dir, self.train_fake_dir, self.train_fake_3d_dir,
                                    random_transform=train_transform, target_transform=target_transform,
                                    ft_width=self.hparams.ft_width, ft_height=self.hparams.ft_height)
        train_loader = DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True
        )

        return train_loader

    def val_dataloader(self):
        """ load validation dataloader """
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        val_set = LivenessDataset('test', self.test_live_dir, self.test_fake_dir, self.test_fake_3d_dir,
                                  random_transform=train_transform, target_transform=target_transform,
                                  ft_width=self.hparams.ft_width, ft_height=self.hparams.ft_height)
        val_loader = DataLoader(val_set, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False)

        return val_loader

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class LightningOpticalFASNet(pl.LightningModule):
    """ Pytorch Lightning module for face liveness detection """

    def __init__(self, hparams):
        """ initialize the variables """
        super().__init__()
        self.save_hyperparameters(hparams)

        net_param = {"num_classes": 2, "width_mult": 1.0}
        self.net = optical_fasnetv1(**net_param)

        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

        self.root_path = hparams.root_dir
        self.train_live_dir = hparams.train_live_dir
        self.train_fake_dir = hparams.train_fake_dir

        self.test_live_dir = hparams.test_live_dir
        self.test_fake_dir = hparams.test_fake_dir

    def forward(self, x):
        """ pass model for training """
        return self.net(x)

    def infer(self, x):
        """ pass model for inference """
        outs, _ = self.net(x)
        return outs[-1]

    def calc_losses(self, cls, target):
        """ calculate the loss during the training process """
        cls_loss = self.cls_criterion(cls, target)
        return cls_loss

    def training_step(self, batch, batch_idx):
        """ train the model """
        raw, label = batch
        raw = raw.cuda(self.device)
        label = label.cuda(self.device)
        class_ret = self.net(raw)

        cls_loss = self.calc_losses(class_ret, label)
        total_loss = cls_loss
        self.log("train/classification_loss", cls_loss)
        self.log("train/total_loss", total_loss)
        return total_loss

    def training_epoch_end(self, outputs):
        """ get the loss at the end every epochs and save it into the tensorboard """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train/train_avg_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        """ validate the model """
        raw, label = batch
        raw = raw.cuda(self.device)
        class_ret = self.net(raw)

        cls_loss = self.calc_losses(class_ret, label)
        total_loss = cls_loss
        self.log("valid/classification_loss", cls_loss)
        self.log("valid/total_loss", total_loss)
        return total_loss

    def validation_epoch_end(self, outputs):
        """ get the loss at the end every epochs and save it into the tensorboard """
        avg_loss = torch.stack([x for x in outputs]).mean()
        # _ = np.hstack([output["target"] for output in outputs])
        # scores = np.vstack([output["score"] for output in outputs])[:, 1]
        # metrics_, best_thr, acc = eval_from_scores(scores, targets)
        # acer, apcer, npcer = metrics_
        # roc_auc = metrics.roc_auc_score(targets, scores)
        self.log("valid/val_avg_loss", avg_loss)
        # tensorboard_logs = {
        #     "val_loss": avg_loss,
        #     "val_roc_auc": roc_auc,
        #     "val_acer": acer,
        #     "val_apcer": apcer,
        #     "val_npcer": npcer,
        #     "val_acc": acc,
        #     "val_thr": best_thr,
        # }
        return avg_loss

    def configure_optimizers(self):
        """ configure the optimizer for the training """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        """ load training dataloader """
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_set = OpticalLivenessDataset('train', self.train_live_dir, self.train_fake_dir,
                                           random_transform=train_transform)
        train_loader = DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True
        )

        return train_loader

    def val_dataloader(self):
        """ load validation dataloader """
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_set = OpticalLivenessDataset('test', self.test_live_dir, self.test_fake_dir,
                                         random_transform=valid_transform)
        val_loader = DataLoader(val_set, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False)

        return val_loader

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
