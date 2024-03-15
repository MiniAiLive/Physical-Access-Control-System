import os
import sys
sys.path.append('../..')

import argparse
import warnings
import cv2
import numpy as np
from tqdm import tqdm
from face_liveness.datasets.generate_patches import CropImage
from face_detect.test import get_bbox
warnings.filterwarnings('ignore')


def get_file_names(dir_path):
    file_list = os.listdir(dir_path)
    total_file_list = []

    for node in file_list:
        full_path = os.path.join(dir_path, node)
        if os.path.isdir(full_path):
            total_file_list = total_file_list + get_file_names(full_path)
        else:
            _, file_extension = os.path.splitext(full_path)
            if file_extension in ['.jpg', '.png', '.bmp', '.mp4', '.avi']:
                total_file_list.append(full_path)

    return total_file_list


def get_label_from_casia_antispoof(file_name):
    real_label = ['1', '2', 'HR_1']

    file_title = os.path.splitext(file_name)[0]
    end = file_title.rfind('_')
    video_filename = file_title[:end]
    if video_filename in real_label:
        return 0

    return 1


def get_label_from_lcc_fasd(file_name):
    label = os.path.basename(os.path.dirname(file_name))
    if label == 'real':
        return 0

    return 1


def get_label_from_humanode_fas(path):
    sub_folders = path.split(os.sep)
    if 'real' in sub_folders or 'Live' in sub_folders:
        return 0

    return 1


def get_label_from_celeba_spoof(file_name):
    label = os.path.basename(os.path.dirname(file_name))
    if label == 'live':
        return 0

    return 1


def get_label_from_raw(file_name):
    if str('ClientRaw') in file_name:
        return 0
    return 1


def get_label_from_Gionee(file_name):
    if str('real') in file_name:
        return 0
    return 1


def generate_data_v1(image, args):
    image_cropper = CropImage()
    image_bbox = get_bbox(image)
    if image_bbox is None:
        return None

    param = {
        "org_img": image, "bbox": image_bbox, "scale": args.scale, "out_w": args.input_size, "out_h": args.input_size, "crop": True,
    }
    if args.scale is None:
        param["crop"] = False

    align_image = image_cropper.crop(**param)
    if args.format == str('origin'):
        align_image = cv2.resize(image, (args.input_size, args.input_size))

    return align_image


def generate_data_v2(image, args):
    # image_bbox = get_bbox(image)

    # count = len(image_bbox)
    align_image = np.zeros([args.input_size, args.input_size, args.input_channel], dtype=np.uint8)
    # for idx in range(count):
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     landmark = get_face_landmark(gray_image, image_bbox[idx])
    #     landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    #     align_vertical(image, image.shape[1], image.shape[0], align_image, args.input_size, args.input_size, args.input_channel,
    #                    landmark_vec, args.input_size/4, args.input_size/2, args.input_size/2)

    return align_image


def generate_filelist(args):
    label = os.path.basename(args.image_dir)
    image_list = get_file_names(args.image_dir)

    f_filelist = open(os.path.join(args.dst_dir, 'color_train_3.txt'), 'a')
    f_label = open(os.path.join(args.dst_dir, 'label_train_3.txt'), 'a')

    for path in tqdm(image_list):
        f_filelist.write(path + '\n')
        f_label.write(label + '\n')


def main(args):
    image_list = get_file_names(args.image_dir)
    for path in tqdm(image_list):
        dir_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)

        if args.dataset == '3d':
            label = 2
        elif args.dataset == 'CASIA_faceAntisp':
            label = get_label_from_casia_antispoof(file_name)
        elif args.dataset == 'fake_web':
            label = 1
        elif args.dataset == 'LCC_FASD':
            label = get_label_from_lcc_fasd(path)
        elif args.dataset == 'raw':
            label = get_label_from_raw(path)
        elif args.dataset == 'CelebA_Spoof':
            label = get_label_from_celeba_spoof(path)
        elif args.dataset in ['superface-dataloader', 'CASIA-FaceV5', 'replayattack']:
            label = 0
        elif args.dataset == 'Gionee':
            label = get_label_from_Gionee(path)
        elif args.dataset == 'Humanode_FAS':
            label = get_label_from_humanode_fas(path)
        else:
            print('Unable to find the proper label')
            sys.exit(1)

        image = cv2.imread(path)
        align_image = generate_data_v1(image, args)
        # align_image = generate_data_v2(image, args)
        if align_image is None:
            continue

        sub_path = path.replace(args.image_dir, "")
        save_path = os.path.join(args.dst_dir, str(label), args.dataset)
        full_path = save_path + sub_path

        dir_name = os.path.dirname(full_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        cv2.imwrite(full_path, align_image)


def parse_args():
    parser = argparse.ArgumentParser(description='Data Preparation for Training')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CASIA-FaceV5', help="[3d, CASIA_faceAntisp, fake_web, \
                             LCC_FASD, raw, CelebA_Spoof, superface-dataloader, Gionee, CASIA-FaceV5, replayattack]")
    parser.add_argument('--format', type=str, default='align', help="[origin, align]")
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--dst_dir', type=str, default='')
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--scale', type=float, default=2.7)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # generate_filelist(args)
