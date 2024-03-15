"""
@author: XiangLan
@file: run_passive_liveness.py
@time: 2023/1/10 09:30
@desc: inference module for passive based liveness detection
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from src.bioauth_ml.face_detect.test import get_bbox
from src.bioauth_ml.face_liveness.datasets.generate_patches import CropImage
from src.bioauth_ml.face_liveness.tools.utility import parse_model_name
from src.bioauth_ml.face_eyeblink.test import get_eye_image, get_eye_bbox, predict_eye_blink
from src.bioauth_ml.face_expression.test import get_face_image, predict_expression
from src.bioauth_ml.face_pose.test import get_pose

face_cropper = CropImage()


def get_liveness_by_frame():
    model_test = None #AntiSpoofPredict(0)
    image_cropper = CropImage()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image_bbox = get_bbox(frame)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir("../face_liveness/checkpoints/anti_spoof_models_v1"):
            h_input, w_input, _, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
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
            prediction += model_test.predict(img, os.path.join("../face_liveness/checkpoints/anti_spoof_models_v1", model_name))
            test_speed += time.time() - start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            print(f"Frame is Real Face. Score: {value:.2f}.")
            result_text = f"RealFace Score: {value:.2f}"
            color = (255, 0, 0)
        else:
            print(f"Frame is Fake Face. Score: {value:.2f}.")
            result_text = f"FakeFace Score: {value:.2f}"
            color = (0, 0, 255)
        print(f"Prediction cost {test_speed:.2f} s")
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def get_eyeblink_by_frame(frame, face_bbox, points, counter, total):
    # for i in range(int(len(points)/2)):
    #     cv2.circle(frame, (int(points[2*i]), int(points[2*i+1])), 1, (0, 0, 255), 2)

    left_bbox, right_bbox = get_eye_bbox(points, frame.shape)
    left_img, right_img, _, _ = get_eye_image(frame, left_bbox, right_bbox)

    left_result = predict_eye_blink(left_img)
    right_result = predict_eye_blink(right_img)

    if left_result == "Close" and right_result == "Close":
        counter += 1
    else:
        if counter >= 1:
            total += 1

    return counter, total


def get_expression_by_frame(frame, face_bbox):
    face = get_face_image(frame, face_bbox)
    # print('[get_expression_by_frame]: ', face)
    if face.size == 0:
        return None

    expression_result = predict_expression(face)
    return expression_result


def get_pose_by_frame(frame, face_bbox, question):
    """ get pose rotation angles(yaw, pitch, roll) with bounding box """
    max_rx, max_ry, _ = 15, 15, 15
    yaw, pitch, roll = get_pose(frame, face_bbox)

    print("[get_pose_by_frame] yaw, pitch, roll:", yaw, pitch, roll)
    result = "front"
    if yaw.item() < -max_ry and question == str("turn face right"):
        result = "right"
    elif yaw.item() > max_ry and question == str("turn face left"):
        result = "left"
    elif pitch.item() > max_rx and question == str("turn face up"):
        result = "up"
    elif pitch.item() < -max_rx and question == str("turn face down"):
        result = "down"

    return result


def get_passive_result(frame, image_bbox, points, question, counter, total):
    pose = None
    expression = None
    if question in ["turn face right", "turn face left", "turn face up", "turn face down"]:
        pose = get_pose_by_frame(frame, image_bbox, question)
    elif question in ["smile", "surprise", "angry"]:
        expression = get_expression_by_frame(frame, image_bbox)
    else:
        pose, expression = None, None

    counter, total = get_eyeblink_by_frame(frame, image_bbox, points, counter, total)

    boxes_face = [image_bbox]
    box_orientation = [[129, 132, 384, 387]]
    emotion = [expression]
    orientation = [pose]
    # print("[get_passive_result]: ", boxes_face, emotion, orientation, total, counter)
    output = {
        'box_face_frontal': boxes_face,
        'box_orientation': box_orientation,
        'emotion': emotion,
        'orientation': orientation,
        'total_blinks': total,
        'count_continuous_blinks': counter
    }
    return output


def main(input_file):
    image = cv2.imread(input_file)
    get_passive_result(image, 0, 0, 0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Image File.')
    parser.add_argument("--model_dir", type=str, default="./checkpoints/anti_spoof_models_v1", help="model used to test")
    parser.add_argument('--input_file', type=str, default='E:/19.Database/FaceSDK/TestDB/Prob-MBGC/04212d.bmp', help="path for image file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.input_file)
