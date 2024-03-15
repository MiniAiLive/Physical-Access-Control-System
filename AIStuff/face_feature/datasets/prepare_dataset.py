import sys

import cv2

sys.path.append('../..')

import os
import argparse
from tqdm import tqdm
from face_liveness.datasets.prepare_dataset import get_file_names
from face_detect.test import get_bbox
from face_pose.test import get_pose
from feature_api import align

parser = argparse.ArgumentParser(description='split or merge')
parser.add_argument('--file_name', default='./glink360k/train.rec', help='source file name')
parser.add_argument('--start_no', type=int, default=1, help='start number for merge')
parser.add_argument('--end_no', type=int, default=34, help='end number for merge')
parser.add_argument('--remove', default=False, help='Flag for Remove')
parser.add_argument('--split_size', type=int, default=80000000, help='split file size')
parser.add_argument('--db_path', default='/datasets/public2/upload/faces_emore_images', help='source file name')
parser.add_argument('--label_file', default='/datasets/public2/upload/faces_emore/faces_emore.list', help='source file name')

args = parser.parse_args()


def merge_files(args):
    """ merge the split files in Azure """
    with open(args.file_name, 'ab') as f:
        for i in range(args.start_no, args.end_no + 1):
            fn = args.file_name + str(i) + '.rar'
            with open(fn, 'rb') as chunk_file:
                f.write(chunk_file.read())
            if args.remove:
                os.remove(fn)
            print(fn)

    print('ok')


def split_files(args):
    file_number = 1
    with open(args.file_name, 'rb') as f:
        chunk = f.read(args.split_size)
        while chunk:
            with open(args.file_name + str(file_number) + '.rar', 'wb') as chunk_file:
                chunk_file.write(chunk)
            file_number += 1
            chunk = f.read(args.split_size)

    print('ok')


def generate_train_label_file(args):
    label_list = []
    file_list = get_file_names(args.db_path)
    class_idx = -1
    dir_list = []
    for file_path in tqdm(file_list):
        dirname = os.path.basename(os.path.dirname(file_path))
        if dirname not in dir_list:
            dir_list.append(dirname)
            class_idx += 1

        label_list.append(f'{file_path}  {class_idx}\n')

    with open(args.label_file, 'w') as f:
        f.writelines(label_list)


def align_files(args):
    """ align face images from the indian dataset and use it as the training dataset for feature extraction """
    file_list = get_file_names(args.db_path)
    for path in tqdm(file_list):
        image = cv2.imread(path)

        face_bbox = get_bbox(image)
        if face_bbox is None:
            continue

        yaw, pitch, roll = get_pose(image, face_bbox)
        if abs(yaw.item()) > 25 or abs(pitch.item()) > 25 or abs(roll.item()) > 25:
            continue

        face_image = align(image, output_size=(112, 112))
        dst_path = path.replace('indian_images', 'indian_align_images')
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

        if face_image is not None:
            cv2.imwrite(dst_path, face_image)


def rename_umd(args):
    """ rename folder name for umd dataset """
    folders = os.listdir(args.db_path)
    for folder in folders:
        os.rename(os.path.join(args.db_path, folder), f'{args.db_path}/umd{folder}')


if __name__ == '__main__':
    # merge_files(args)
    # split_files(args)
    # align_files(args)
    # generate_train_label_file(args)
    rename_umd(args)
