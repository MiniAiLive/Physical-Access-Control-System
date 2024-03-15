"""
@author: Xiang
@file: eval.py
@time: 2022/12/13 15:06
@desc: evaluate the model for liveness detection
"""
import os
import shutil
import sys
sys.path.append('..')

import argparse
import time
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from face_detect.test import get_bbox
from face_liveness.datasets.prepare_dataset import (
    get_label_from_casia_antispoof,
    get_label_from_celeba_spoof,
    get_file_names,
    get_label_from_humanode_fas
)
from face_liveness.datasets.generate_patches import CropImage
from face_liveness.tools.utility import parse_model_name
from face_liveness.test import LivenessEstimator
from face_pose.test import get_pose

warnings.filterwarnings('ignore')

face_cropper = CropImage()


def is_centered_face(image, face_bbox):
    """ check if face is in the center of the camera """
    height, width, _ = image.shape

    face_center_x = int(face_bbox[0] + face_bbox[2] / 2)
    face_center_y = int(face_bbox[1] + face_bbox[3] / 2)

    ratio_x, ratio_y = 0.3, 0.2
    if int(ratio_x * width) < face_center_x < int((1 - ratio_x) * width) and \
            int(ratio_y * height) < face_center_y < int((1 - ratio_y) * height):
        return True

    return False


def is_fronted_face(image, face_bbox):
    """ check if it's front face """
    max_pan, max_yaw = 15, 15
    pan, yaw, _ = get_pose(image, face_bbox)
    if -max_pan < pan < max_pan and -max_yaw < yaw < max_yaw:
        return True

    return False


def is_normal_distance(image, face_bbox):
    """ check if face is far away from the camera """
    height, width, _ = image.shape

    max_size = height if height < width else width
    face_max_ratio = 0.4
    face_min_ratio = 0.2
    face_size = (face_bbox[2] + face_bbox[3]) / 2

    if face_min_ratio * max_size > face_size:
        return 1
    elif face_size > (1 - face_max_ratio) * max_size:
        return 2
    return 0


def evaluate(hparams):
    """ evaluate the model """
    total_files = get_file_names(hparams.image_dir)

    liveness_estimator = LivenessEstimator("checkpoints/anti_spoof_models_v1", hparams.device_id)

    for file_name in tqdm(total_files):
        image = cv2.imread(file_name)

        image_bbox = get_bbox(image)
        if image_bbox is None:
            continue

        liveness_scores = np.zeros(8)

        if "1_128x128_MiniFASNetV2SE.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 1.0, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            liveness_scores[0] = liveness_estimator.predict(img, "1_128x128_MiniFASNetV2SE.pth")[0][0]
        if "2.7_128x128_MiniFASNetV2SE.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            liveness_scores[1] = liveness_estimator.predict(img, "2.7_128x128_MiniFASNetV2SE.pth")[0][0]
        if "4_0_0_128x128_MiniFASNetV2SE.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 4.0, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            liveness_scores[2] = liveness_estimator.predict(img, "4_0_0_128x128_MiniFASNetV2SE.pth")[0][0]
        if "org_1_128x128_MiniFASNetV2SE.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 1.0, "out_w": 128, "out_h": 128,
                     "crop": False}
            img = face_cropper.crop(**param)
            liveness_scores[3] = liveness_estimator.predict(img, "org_1_128x128_MiniFASNetV2SE.pth")[0][0]
        if "2.7_128x128_FASNetA.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            liveness_scores[4] = liveness_estimator.predict(input_image, "2.7_128x128_FASNetA.pth")[0][0]
        if "2.7_128x128_FASNetB.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            liveness_scores[5] = liveness_estimator.predict(img, "2.7_128x128_FASNetB.pth")[0][0]
        if "2.7_128x128_FASNetCV1.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            liveness_scores[6] = liveness_estimator.predict(input_image, "2.7_128x128_FASNetCV1.pth")[0][0]
        if "2.7_128x128_fnet.pth" in hparams.model_list:
            param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128,
                     "crop": True}
            img = face_cropper.crop(**param)
            input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            liveness_scores[7] = liveness_estimator.predict(input_image, "2.7_128x128_fnet.pth")[0][0]
        if hparams.save:
            data = pd.DataFrame({'path': [file_name], 'minifasnet1': liveness_scores[0], 'fasnetb': liveness_scores[1],
                                 'fasnetcv1': liveness_scores[2], 'minifasnet2.7': liveness_scores[3],
                                 'minifasnet4': liveness_scores[4], 'minifasnet_org': liveness_scores[5],
                                 'fnet': liveness_scores[6], 'fasneta': liveness_scores[7]})

            data.to_csv('liveness_eval.csv', mode='a', header=False)


def get_face_liveness(image):
    """ get the score for the deployment """
    liveness_estimator = LivenessEstimator(os.path.join(os.path.dirname(__file__), "checkpoints/anti_spoof_models_v1"), 0)
    image_cropper = CropImage()

    image_bbox = get_bbox(image)
    # if image_bbox is None:
    #     return default.face_is_not_detected
    #
    # if not is_centered_face(image, image_bbox):
    #     return default.face_is_not_centered
    #
    # if not is_fronted_face(image, image_bbox):
    #     return default.face_is_not_fronted
    #
    # dist_state = is_normal_distance(image, image_bbox)
    # if dist_state == 1:
    #     return default.face_is_far
    # if dist_state == 2:
    #     return default.face_is_near

    predicted_list = []
    test_speed = 0

    # sum the prediction from single model's result
    for model_name in os.listdir(os.path.join(os.path.dirname(__file__), "checkpoints/anti_spoof_models_v1")):
        h_input, w_input, _, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        img = image_cropper.crop(**param)

        start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = liveness_estimator.predict(img, model_name)

        if scale == 2.7:
            score = liveness_estimator.predict(img, "2.7_128x128_FASNetB.pth")[0][0]
            predicted_list.append(score)
            score = liveness_estimator.predict(img, "2.7_128x128_FASNetCV1.pth")[0][0]
            predicted_list.append(score)

        predicted_list.append(pred[0][0])
        test_speed += time.time() - start

    fnet_param = {"org_img": image, "bbox": image_bbox, "scale": 2.7, "out_w": 128, "out_h": 128, "crop": True}
    input_image = image_cropper.crop(**fnet_param)

    test_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    fnet_score = liveness_estimator.predict(input_image, "2.7_128x128_fnet.pth")[0][0]
    fasnet_score = liveness_estimator.predict(test_image, "2.7_128x128_FASNetA.pth")[0][0]
    # pred_label = np.argmax(prediction)
    predicted_list.append(fnet_score)
    predicted_list.append(fasnet_score)
    score = (predicted_list[3] + predicted_list[4]) / 2
    score = predicted_list[3]
    predicted_list.append(score)
    print("Live Scores: ", predicted_list)

    if score > 0.5:
        return True
    return False


def get_total_fake_live_count(dataframe):
    """ get tatal count for the evaluation """
    live_count, fake_count = 0, 0
    for idx in dataframe.index:
        if args.dataset == str('CelebA_Spoof'):
            label = get_label_from_celeba_spoof(dataframe['path'][idx])
        elif args.dataset == str('CASIA_faceAntisp'):
            label = get_label_from_casia_antispoof(dataframe['path'][idx])
        elif args.dataset == str('Humanode_FAS'):
            label = get_label_from_humanode_fas(dataframe['path'][idx])
        else:
            sys.exit(1)

        if label == 0:
            live_count += 1.0
        else:
            fake_count += 1.0

    return live_count, fake_count


def eval_csv(hparams):
    """ evaluate the accuracy with csv file """
    dataframe = pd.read_csv(hparams.csv_file)

    acc_tmp_info = {'minifasnet1': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'fasnetb': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'fasnetcv1': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'minifasnet2.7': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'minifasnet4': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'minifasnet_org': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'fnet': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'fasneta': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0},
                    'total': {'max_live_count': 0.0, 'max_fake_count': 0.0, 'max_t': 0.0}}

    acc_info = {'minifasnet1': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'fasnetb': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'fasnetcv1': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'minifasnet2.7': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'minifasnet4': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'minifasnet_org': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'fnet': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'fasneta': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0},
                'total': {'max_a': 0.0, 'max_t': 0.0, 'max_live_a': 0.0, 'max_fake_a': 0.0}}

    live_count, fake_count = get_total_fake_live_count(dataframe)

    for index in tqdm(range(100)):
        threshold = index / 100.0
        for idx in dataframe.index:
            if hparams.dataset == str('CelebA_Spoof'):
                label = get_label_from_celeba_spoof(dataframe['path'][idx])
            elif hparams.dataset == str('CASIA_faceAntisp'):
                label = get_label_from_casia_antispoof(dataframe['path'][idx])
            elif hparams.dataset == str('Humanode_FAS'):
                label = get_label_from_humanode_fas(dataframe['path'][idx])
            else:
                sys.exit(1)

            dataframe['total'][idx] = (dataframe['minifasnet1'][idx] + dataframe['fasnetcv1'][idx] + dataframe['minifasnet4'][idx] + dataframe['fnet'][idx] + + dataframe['fasneta'][idx])/5

            for key in acc_tmp_info.keys():
                pred = 0 if dataframe[key][idx] > threshold else 1
                if pred == label:
                    if label == 0:
                        acc_tmp_info[key]['max_live_count'] += 1.0
                    else:
                        acc_tmp_info[key]['max_fake_count'] += 1.0

                elif hparams.log_dir is not None:
                    img = cv2.imread(dataframe['path'][idx])
                    filename = os.path.basename(dataframe['path'][idx])
                    save_dir = os.path.join(hparams.log_dir, str(label))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    cv2.imwrite(os.path.join(save_dir, filename), img)

        for key in acc_tmp_info.keys():
            if live_count != 0 and fake_count != 0:
                acc = (acc_tmp_info[key]['max_live_count']/live_count + acc_tmp_info[key]['max_fake_count']/fake_count)/2
            elif live_count == 0:
                acc = acc_tmp_info[key]['max_fake_count']/fake_count
            elif fake_count == 0:
                acc = acc_tmp_info[key]['max_live_count'] / live_count
            else:
                sys.exit()

            if acc > acc_info[key]['max_a']:
                acc_info[key]['max_a'] = acc
                acc_info[key]['max_t'] = threshold
                acc_info[key]['max_live_a'] = acc_tmp_info[key]['max_live_count']/live_count
                acc_info[key]['max_fake_a'] = acc_tmp_info[key]['max_fake_count']/fake_count

            acc_tmp_info[key]['max_live_count'] = 0
            acc_tmp_info[key]['max_fake_count'] = 0

    fig, ax = plt.subplots(figsize=(16, 9))
    model_list = []
    acc_list = []
    for key in acc_info.keys():
        model_list.append(f"{key}_{acc_info[key]['max_t']:.03}")
        acc_list.append(acc_info[key]['max_a'])
    ax.barh(model_list, acc_list)

    print("Final Result: ", acc_info)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=5)

    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    ax.invert_yaxis()

    for i in ax.patches:
        plt.text(i.get_width(), i.get_y() + 0.2, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='grey')

    ax.set_title('Evaluation in Passive Liveness Detection', loc='left')
    fig.text(0.9, 0.05, os.path.basename(hparams.csv_file), fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)
    fig.text(0.9, 0.03, f'Live: {live_count}, Fake: {fake_count}', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)

    plt.show()


def filter_db(args):
    total_files = get_file_names(args.image_dir)
    for idx in range(0, len(total_files), 20):
        src_path = total_files[idx]
        dst_path = src_path.replace(args.image_dir, args.dst_dir)
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

        shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    description = "Evaluation for liveness detection"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--mode", type=str, default='csv', help="[csv, dir, image, dataset]")
    parser.add_argument('--dataset', type=str, default='Humanode_FAS', help="[Humanode_FAS]")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold for the evaluation")
    parser.add_argument("--model_list", type=list, default=["2.7_128x128_MiniFASNetV2SE.pth"], help="liveness model")
    parser.add_argument("--csv_file", type=str, default="", help="csv file to save score value")
    parser.add_argument("--image_dir", type=str, default="", help="source image directory for the evaluation")
    parser.add_argument("--dst_dir", type=str, default="", help="destination directory")
    parser.add_argument("--log_dir", type=str, default=None, help="save log")
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()

    if args.mode == str('dir'):
        evaluate(args)
    elif args.mode == str('csv'):
        eval_csv(args)
    elif args.mode == str('image'):
        image = cv2.imread('./datasets/images/test/yu.jpg')
        get_face_liveness(image)
    elif args.mode == str('dataset'):
        filter_db(args)
    else:
        print('Please set up the evaluation mode.')
