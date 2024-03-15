import os
from PIL import Image
from torch.utils.data import Dataset
from src.bioauth_ml.face_liveness.datasets.prepare_dataset import get_file_names


class MyDataset(Dataset):
    """ define the dataloader to train the deepfake model """
    def __init__(self, data_path, transform=None, target_transform=None):
        super().__init__()
        total_files = get_file_names(data_path)
        imgs = []
        for path in total_files:
            label_str = os.path.basename(path)
            label = 0 if label_str == str("real") else 1
            imgs.append((path, label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
