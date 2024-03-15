import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def img_loader(path):
    try:
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        return img
    except IOError:
        print('Cannot load image ' + path)
        return None


class LFW(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):
        super().__init__()
        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // 600
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for _, i in enumerate(imglist):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)


class LFWDataset:
    def __init__(self, root, file_list, loader=img_loader):
        super().__init__()
        self.root = root
        self.file_list = file_list
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // 600
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

        self.idx = 0
        self.len = len(self.nameLs)

    def __next__(self, index):
        if self.idx > self.len:
            raise StopIteration

        img_l = self.loader(os.path.join(self.root, self.nameLs[self.idx]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[self.idx]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        outputs = []
        image_mean = np.array([127.5, 127.5, 127.5])
        for _, image in enumerate(imglist):
            img = (image - image_mean) / 127.5
            img = img.astype(np.float32).transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

            outputs.append(img)

        self.idx += 1
        return outputs

    def __iter__(self):
        return self


if __name__ == '__main__':
    root = 'D:/data/lfw_align_112'
    file_list = 'D:/data/pairs.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]+
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = LFW(root, file_list, transform=transform)
    #dataloader = LFW(root, file_list)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        for d in data:
            print(d[0].shape)
