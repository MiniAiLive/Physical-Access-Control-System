import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# CASIA-SURF training dataloader and our private dataloader
# depth_dir_train_file = os.getcwd() +'/data/2depth_train.txt'
# label_dir_train_file = os.getcwd() + '/data/2label_train.txt'

# CASIA-SURF training dataloader and our private dataloader
depth_dir_train_file = os.getcwd() +'/datasets/filelist/color_train_4.txt'
label_dir_train_file = os.getcwd() + '/datasets/filelist/label_train_4.txt'

# for IR train
# depth_dir_train_file = os.getcwd() +'/data/ir_final_train.txt'
# label_dir_train_file = os.getcwd() +'/data/label_ir_train.txt'



# CASIA-SURF Val data
depth_dir_val_file = os.getcwd() +'/datasets/filelist/color_train_4.txt'
label_dir_val_file = os.getcwd() +'/datasets/filelist/label_train_4.txt' #val-label 100%


# depth_dir_val_file = os.getcwd() +'/data/ir_val.txt'
# label_dir_val_file = os.getcwd() +'/data/label_val.txt' #val-label 100%

# # CASIA-SURF Test data
depth_dir_test_file = os.getcwd() +'/datasets/filelist/color_train_4.txt'
label_dir_test_file = os.getcwd() +'/datasets/filelist/label_train_4.txt'


# depth_dir_test_file = os.getcwd() +'/data/ir_test.txt'
# label_dir_test_file = os.getcwd() +'/data/label_test.txt'

class CASIA(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None, phase_test=False):
        super().__init__()
        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform

        try:
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()

            with open(depth_dir_val_file, 'r') as f:
                self.depth_dir_val = f.read().splitlines()
            with open(label_dir_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(depth_dir_test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_dir_test_file, 'r') as f:
                    self.label_dir_test = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            sys.exit()

    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test
                label_dir = self.label_dir_test
#                 label = int(label_dir[idx])
                label = np.random.randint(0, 2, 1)
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)

        depth = Image.open(depth_dir[idx])
        depth = depth.convert('RGB')

        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth, label
        else:
            return depth, label
