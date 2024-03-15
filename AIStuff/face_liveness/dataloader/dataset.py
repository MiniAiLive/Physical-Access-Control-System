import os
import cv2
import torch
from PIL import Image
from torch.utils import data
from .dataset_folder import generate_FT
from datasets.prepare_dataset import get_file_names


def make_dataset(rgb_dir, depth_dir=None):
    items = []
    for file in os.listdir(rgb_dir):
        if file.endswith('.bmp') or file.endswith('.jpg') or file.endswith('.png'):
            if depth_dir is not None:
                depth_file = file[:-4] + '_depth.jpg'
                depth_file = os.path.join(depth_dir, depth_file)
                if os.path.exists(depth_file):
                    items.append((os.path.join(rgb_dir, file), depth_file))
            else:
                items.append((os.path.join(rgb_dir, file), ''))
    return items


class LivenessDataset(data.Dataset):
    def __init__(self, mode, live_rgb, fake_rgb, fake_3d_rgb, random_transform=None, target_transform=None,
                 ft_width=10, ft_height=10):
        super().__init__()
        self.live_path = live_rgb
        self.fake_path = fake_rgb
        self.fake3d_path = fake_3d_rgb

        self.live_imgs = get_file_names(self.live_path)
        self.fake_imgs = get_file_names(self.fake_path)
        self.fake_3d_imgs = get_file_names(self.fake3d_path)

        self.live_len = len(self.live_imgs)
        self.fake_len = len(self.fake_imgs)
        self.fake_3d_len = len(self.fake_3d_imgs)
        self.mode = mode

        max_len = max([self.live_len, self.fake_len, self.fake_3d_len])

        live_n, live_d = max_len // self.live_len, max_len % self.live_len
        fake_n, fake_d = max_len // self.fake_len, max_len % self.fake_len
        fake3d_n, fake3d_d = max_len // self.fake_3d_len, max_len % self.fake_3d_len

        self.total_files = live_n * self.live_imgs + self.live_imgs[:live_d] + \
                           fake_n * self.fake_imgs + self.fake_imgs[:fake_d] + \
                           fake3d_n * self.fake_3d_imgs + self.fake_3d_imgs[:fake3d_d]

        print(mode, ': live image size:', len(self.live_imgs))
        print(mode, ': fake image size:', len(self.fake_imgs))
        print(mode, ': fake 3d image size:', len(self.fake_3d_imgs))
        print(mode, ': total size:', len(self.total_files))

        if len(self.live_imgs) == 0 or len(self.fake_imgs) == 0 or len(self.fake_3d_imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.ft_width = ft_width
        self.ft_height = ft_height

        self.random_transform = random_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.total_files[index]
        img = cv2.imread(img_path)

        ft_img = generate_FT(img)
        ft_img = cv2.resize(ft_img, (self.ft_width, self.ft_height))
        ft_img = torch.from_numpy(ft_img).float()
        ft_img = torch.unsqueeze(ft_img, 0)

        if self.random_transform is not None:
            img = self.random_transform(img)

        if self.live_path in img_path:
            label = 0
        elif self.fake_path in img_path:
            label = 1
        elif self.fake3d_path in img_path:
            label = 2
        else:
            raise RuntimeError("Not found label.")

        # if self.target_transform is not None:
        #     depth = self.target_transform(depth)
        return img, ft_img, label

    def __len__(self):
        return len(self.total_files)


class OpticalLivenessDataset(data.Dataset):
    def __init__(self, mode, live_rgb, fake_rgb, random_transform=None):
        super().__init__()
        self.live_path = live_rgb
        self.fake_path = fake_rgb

        self.live_imgs = get_file_names(self.live_path)
        self.fake_imgs = get_file_names(self.fake_path)

        self.live_len = len(self.live_imgs)
        self.fake_len = len(self.fake_imgs)
        self.mode = mode

        max_len = max([self.live_len, self.fake_len])

        live_n, live_d = max_len // self.live_len, max_len % self.live_len
        fake_n, fake_d = max_len // self.fake_len, max_len % self.fake_len

        self.total_files = live_n * self.live_imgs + self.live_imgs[:live_d] + \
                           fake_n * self.fake_imgs + self.fake_imgs[:fake_d]

        print(f'{mode}: live image size: {len(self.live_imgs)}')
        print(f'{mode}: fake image size: {len(self.fake_imgs)}')
        print(f'{mode}: total size: {len(self.total_files)}')

        if len(self.live_imgs) == 0 or len(self.fake_imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.random_transform = random_transform

    def __getitem__(self, index):
        img_path = self.total_files[index]
        img = cv2.imread(img_path)

        if self.random_transform is not None:
            img = self.random_transform(img)

        if self.live_path in img_path:
            label = 0
        elif self.fake_path in img_path:
            label = 1
        else:
            raise RuntimeError("Not found label.")

        return img, label

    def __len__(self):
        return len(self.total_files)
