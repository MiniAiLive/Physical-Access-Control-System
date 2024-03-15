"""
@author: XiangLan
@file: run_motion_detection.py
@time: 2023/1/3 09:30
@desc: inference module for motion analysis based liveness detection
"""
import sys

sys.path.append('..')
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms
from tqdm import tqdm
from face_detect.test import get_bbox
from landmark_api import get_face_landmark
from face_liveness.datasets.generate_patches import CropImage
from face_liveness.datasets.prepare_dataset import get_file_names
from face_liveness.train_mnet import get_mnet_score
from face_liveness.model.OpticalFASNet import optical_fasnetv1
from face_pose.test import get_pose

face_cropper = CropImage()


def load_optical_fasnetv1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = os.path.join(
        os.path.dirname(__file__),
        "../face_liveness/checkpoints/anti_spoof_models_v2/optical_fasnetv1/1.7_128_128_OpticalNet_epoch=49.pth"
    )

    net_param = {"num_classes": 2, "width_mult": 1.0}
    model = optical_fasnetv1(**net_param)

    checkpoint = torch.load(pretrained_path, map_location=device)

    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        name_key = key[4:]
        new_state_dict[name_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_optical_flow(args):
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    correct_video_count = 0
    model = load_optical_fasnetv1()
    video_files = get_file_names(args.video_dir)

    for path in tqdm(video_files):
        cap = cv2.VideoCapture(path)
        _, first_frame = cap.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        count, process_total_frames, correct_frames = 0, 0, 0
        while cap.isOpened() and count < 2000:
            count += 1
            _, frame = cap.read()
            if frame is None:
                continue

            face_bbox = get_bbox(frame)
            if face_bbox is None:
                continue

            yaw, pitch, roll = get_pose(frame, face_bbox)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            param = {
                "org_img": rgb, "bbox": face_bbox, "scale": args.scale, "out_w": args.input_size,
                "out_h": args.input_size, "crop": True,
            }
            if args.scale is None:
                param["crop"] = False

            align_image = face_cropper.crop(**param)
            img = valid_transform(align_image)
            img = img.unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                result = F.softmax(output).cpu().numpy()
                cv2.putText(frame, f'score = {result[0][0]:.02f}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            if abs(yaw.item()) < 15 and abs(pitch.item()) < 15 and abs(roll.item()) < 15:
                label = os.path.dirname(path)
                predict = 0 if result[0][0] > 0.5 else 1
                process_total_frames += 1
                if label == predict:
                    correct_frames += 1

            cv2.imshow("Motion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_gray = gray

        if correct_frames > int(0.8 * process_total_frames):
            correct_video_count += 1

        cap.release()
        cv2.destroyAllWindows()

    print(f'Accuracy = {correct_video_count / len(video_files):0.02f}')


def extract_optical_flow_images(args):
    """ extract optical flow images from superface-dataset """
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_optical_fasnetv1()
    image_files = get_file_names(args.image_dir)
    current_id = None
    count = 0

    for path in tqdm(image_files):
        id = os.path.dirname(os.path.dirname(path))
        if id != current_id:
            current_id = id
            first_frame = cv2.imread(path)
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(first_frame)
            mask[..., 1] = 255
            continue

        frame = cv2.imread(path)

        face_bbox = get_bbox(frame)
        if face_bbox is None:
            continue

        yaw, pitch, roll = get_pose(frame, face_bbox)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        param = {
            "org_img": rgb, "bbox": face_bbox, "scale": args.scale, "out_w": args.input_size,
            "out_h": args.input_size, "crop": True,
        }
        if args.scale is None:
            param["crop"] = False

        align_image = face_cropper.crop(**param)

        if args.test_model:
            # gray_align = cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY)
            img = valid_transform(align_image)
            img = img.unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                result = F.softmax(output).cpu().numpy()
                cv2.putText(frame, f'score = {result[0][0]:.02f}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Motion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save_img and abs(yaw.item()) < 15 and abs(pitch.item()) < 15 and abs(roll.item()) < 15:
            dst_path = path.replace(args.image_dir, args.dst_dir)
            dst_dir = os.path.dirname(os.path.dirname(dst_path))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            dst_path = os.path.join(dst_dir, f"{count}.png")
            count += 1

            cv2.imshow("input", align_image)
            cv2.imwrite(dst_path, align_image)

        prev_gray = gray


def detect_motion_v1(args):
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_optical_fasnetv1()
    video_files = get_file_names(args.video_dir)
    for path in tqdm(video_files):
        cap = cv2.VideoCapture(0)
        _, first_frame = cap.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        count = 0
        while cap.isOpened() and count < 2000:
            count += 1
            _, frame = cap.read()
            if frame is None:
                continue

            face_bbox = get_bbox(frame)
            if face_bbox is None:
                continue

            yaw, pitch, roll = get_pose(frame, face_bbox)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            param = {
                "org_img": rgb, "bbox": face_bbox, "scale": args.scale, "out_w": args.input_size,
                "out_h": args.input_size, "crop": True,
            }
            if args.scale is None:
                param["crop"] = False

            align_image = face_cropper.crop(**param)

            if args.test_model:
                # gray_align = cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY)
                img = valid_transform(align_image)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    output = model(img)
                    result = F.softmax(output).cpu().numpy()
                    cv2.putText(frame, f'score = {result[0][0]:.02f}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Motion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.save_img and abs(yaw.item()) < 15 and abs(pitch.item()) < 15 and abs(roll.item()) < 15:
                dst_path = path.replace(args.video_dir, args.dst_dir)
                dst_dir = dst_path.replace('.mp4', '').replace('.avi', '')
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                dst_path = os.path.join(dst_dir, f"{count}.png")

                cv2.imshow("input", align_image)
                cv2.imwrite(dst_path, align_image)

            prev_gray = gray

        cap.release()
        cv2.destroyAllWindows()


def detect_motion_v2():
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_path, 'frames')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    cap = cv2.VideoCapture(0)
    color = (0, 255, 0)
    first_frame_trigger = True
    mask_frame_trigger = True

    frame_index = 0
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            continue

        if mask_frame_trigger:
            mask = np.zeros_like(frame)
            mask_frame_trigger = False

        if first_frame_trigger:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            first_frame_trigger = False
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        face_bbox = get_bbox(frame)
        if face_bbox is None:
            continue
        points = get_face_landmark(gray, face_bbox)
        draw_points = points.detach().numpy().reshape(-1, 1, 2)
        prev_pt = np.array([draw_points[30], draw_points[39], draw_points[45]])
        # for i, point in enumerate(draw_points):
        #     frame = cv2.putText(frame, str(i), (int(point[0][0]), int(point[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        next_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params)
        good_old = prev_pt[status == 1].astype(int)
        good_new = next_pt[status == 1].astype(int)

        euclidean_list = []
        for _, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            euclidean_list.append(np.sqrt((a - c) ** 2 + (b - d) ** 2))
            # frame = cv2.circle(frame, (a, b), 3, color, -1)

        euclidean_dist = sum(euclidean_list) / len(euclidean_list)
        degree = 100 * euclidean_dist / face_bbox[2]
        output = cv2.add(frame, mask)
        if degree > 5:
            param = {"org_img": frame, "bbox": face_bbox, "scale": 2.7, "out_w": 150, "out_h": 150, "crop": True}
            face_image = face_cropper.crop(**param)
            cv2.imwrite(f"{save_path}/image_{frame_index:05}.jpg", face_image)
            cv2.imwrite(f"{save_path}/origin_{frame_index:05}.jpg", frame)
            prev_gray = gray.copy()
            frame_index += 1
        prev_pt = good_new.reshape(-1, 1, 2)
        cv2.imshow("Motion Detection", output)
        if cv2.waitKey(10) & 0xFF == ord('q') or frame_index == 15:
            break

    cap.release()
    cv2.destroyAllWindows()

    score = get_mnet_score(save_path)
    print("MNet Score = ", score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video_dir", type=str, help="")
    parser.add_argument("--image_dir", type=str, help="")
    parser.add_argument("--dst_dir", type=str, help="")
    parser.add_argument("--save_img", action='store_true', help="")
    parser.add_argument("--test_model", action='store_true', help="")
    parser.add_argument("--scale", type=float, default=1.4, help="")
    parser.add_argument("--input_size", type=int, default=128, help="")
    args = parser.parse_args()

    detect_motion_v1(args)
    # detect_motion_v2()
    # extract_optical_flow_images(args)
