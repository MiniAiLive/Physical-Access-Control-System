import sys

from torch.utils.data import DataLoader
from torchvision import transforms
from src.bioauth_ml.face_liveness.dataloader.dataset_folder import DatasetFolderFT, DatasetEvalFolder
from src.bioauth_ml.face_liveness.dataloader import transform


def get_train_loader(conf):
    train_transform = transform.Compose([
        transform.ToPILImage(),
        transform.RandomResizedCrop(size=tuple(conf.input_size),
                                    scale=(0.9, 1.1)),
        transform.ColorJitter(brightness=0.4,
                              contrast=0.4, saturation=0.4, hue=0.1),
        transform.RandomRotation(10),
        transform.RandomHorizontalFlip(),
        transform.ToTensor()
    ])
    root_path = f'{conf.train_root_path}/{conf.patch_info}'
    trainset = DatasetFolderFT(root_path, train_transform, None, conf.ft_width, conf.ft_height)

    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0)
    return train_loader


def get_test_loader():
    train_transform = transform.Compose([
        transforms.ToTensor()
    ])

    train_live_dir = "/src/bioauth_ml/face_liveness/datasets/rgb_image/2.7_128x128/0"
    train_fake_dir = "/src/bioauth_ml/face_liveness/datasets/rgb_image/2.7_128x128/1"
    train_fake_3d_dir = "/src/bioauth_ml/face_liveness/datasets/rgb_image/2.7_128x128/2"

    trainset = DatasetEvalFolder(train_live_dir, train_fake_dir, train_fake_3d_dir, train_transform)

    # train_loader = DataLoader(
    #     trainset,
    #     batch_size=64,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=0)
    return trainset


if __name__ == '__main__':
    loader = get_test_loader()
    for data in loader:
        print(">>>>>>>>>>>>: ", data)
        sys.exit(1)
