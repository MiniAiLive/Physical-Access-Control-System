import cv2
import torch
import numpy as np
from torchvision import datasets
from face_liveness.datasets.prepare_dataset import get_file_names


def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except:
                print(f"Error Occured: ", path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for _, img in enumerate(fimg):
        if maxx < max(img):
            maxx = max(img)
        if minn > min(img):
            minn = min(img)
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg


class DatasetEvalFolder:
    def __init__(self, live_rgb, fake_rgb, fake_3d_rgb, target_transform=None):
        self.live_imgs = get_file_names(live_rgb)
        self.fake_imgs = get_file_names(fake_rgb)
        self.fake_3d_imgs = get_file_names(fake_3d_rgb)

        self.live_len = len(self.live_imgs)
        self.fake_len = len(self.fake_imgs)
        self.fake_3d_len = len(self.fake_3d_imgs)

        print('live image size:', len(self.live_imgs))
        print('fake image size:', len(self.fake_imgs))
        print('fake 3d image size:', len(self.fake_3d_imgs))

        if len(self.live_imgs) == 0 or len(self.fake_imgs) == 0 or len(self.fake_3d_imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.target_transform = target_transform
        self.max = len(self.live_imgs) + len(self.fake_imgs) + len(self.fake_3d_imgs)
        self.batch_size = 64

    def __getitem__(self, index):
        if index < self.live_len:
            img_path = self.live_imgs[index]
            img = cv2.imread(img_path)
            img = img.astype(np.float32).transpose(2, 1, 0)
            # image_data = np.expand_dims(img, 0)
            # img = self.target_transform(img)
            label = 0
        elif index < self.live_len + self.fake_len:
            img_path = self.fake_imgs[index - self.live_len - self.fake_3d_len]
            img = cv2.imread(img_path)
            img = img.astype(np.float32).transpose(2, 1, 0)
            # image_data = np.expand_dims(img, 0)
            # img = self.target_transform(img)
            label = 1
        else:
            img_path = self.fake_3d_imgs[index - self.live_len - self.fake_len]
            img = cv2.imread(img_path)
            img = img.astype(np.float32).transpose(2, 1, 0)
            # image_data = np.expand_dims(img, 0)
            # img = self.target_transform(img)
            label = 2

        return img, label

    def __len__(self):
        return len(self.live_imgs) + len(self.fake_imgs) + len(self.fake_3d_imgs)
