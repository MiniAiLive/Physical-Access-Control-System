"""
@author: Xiang Lan
@file: run_active_liveness.py
@time: 2023/1/4 09:30
@desc: include scenarios for active liveness detection
"""

import sys
sys.path.append('../../..')

import os
import random
import cv2
import numpy as np

from src.bioauth_ml.face_pipeline.questions import question_bank, challenge_result
from src.bioauth_ml.face_pipeline.run_passive_liveness import get_passive_result
from src.bioauth_ml.landmark_api import get_face_landmark
from src.bioauth_ml.face_liveness.train_mnet import get_mnet_score
from src.bioauth_ml.face_liveness.datasets.generate_patches import CropImage
from src.bioauth_ml.face_detect.test import get_bbox
from src.bioauth_ml.face_utils.digit_recognition.test import main

face_cropper = CropImage()


def show_image(cam, text, color=(0, 0, 255)):
    """ show image and print message """
    _, image = cam.read()
    if image is None:
        return None

    # image = imutils.resize(image, width=720)
    # im = cv2.flip(im, 1)
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return image


def main_pipeline1():
    cv2.namedWindow('Scenario 1 in Active Liveness Detection')
    camera = cv2.VideoCapture(0)

    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_path, 'frames')
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # parameters
    first_frame_trigger = True
    counter, total = 0, 0
    counter_ok_questions = 0
    counter_ok_consecutives = 0
    limit_consecutives = 3
    limit_questions = 6
    counter_try = 0
    limit_try = 50
    frame_index = 0

    for _ in range(0, limit_questions):
        index_question = random.randint(0, 5)
        question = question_bank(index_question)

        image = show_image(camera, question)
        if image is None:
            continue

        cv2.imshow('Scenario 1 in Active Liveness Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for try_index in range(limit_try):
            _, frame = camera.read()
            # configure motion analyser
            if first_frame_trigger:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                first_frame_trigger = False
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_bbox = get_bbox(frame)
            if face_bbox is None:
                continue
            points = get_face_landmark(gray, face_bbox)
            draw_points = points.detach().numpy().reshape(-1, 1, 2)
            prev_pt = np.array([draw_points[30], draw_points[39], draw_points[45]])

            next_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params)
            good_old = prev_pt[status == 1].astype(int)
            good_new = next_pt[status == 1].astype(int)

            euclidean_list = []
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                euclidean_list.append(np.sqrt((a - c) ** 2 + (b - d) ** 2))

            euclidean_dist = sum(euclidean_list) / len(euclidean_list)
            degree = 100 * euclidean_dist / face_bbox[2]
            if degree > 5:
                param = {"org_img": frame, "bbox": face_bbox, "scale": 2.7, "out_w": 150, "out_h": 150, "crop": True}
                face_image = face_cropper.crop(**param)
                cv2.imwrite(f"{save_path}/image_{frame_index:05}.jpg", face_image)
                cv2.imwrite(f"{save_path}/origin_{frame_index:05}.jpg", frame)
                prev_gray = gray.copy()
                frame_index += 1

            total_0 = total
            out_model = get_passive_result(image, face_bbox, points, question, counter, total_0)
            if out_model is None:
                continue

            total = out_model['total_blinks']
            counter = out_model['count_continuous_blinks']
            dif_blink = total - total_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0

            challenge_res = challenge_result(question, out_model, blinks_up)

            image = show_image(camera, question)
            cv2.imshow('Scenario 1 in Active Liveness Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if challenge_res == "pass":
                image = show_image(camera, question + " : ok")
                cv2.imshow('Scenario 1 in Active Liveness Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                counter_ok_consecutives += 1
                if counter_ok_consecutives == limit_consecutives:
                    counter_ok_questions += 1
                    counter_try = 0
                    counter_ok_consecutives = 0
                    break
                else:
                    continue

            elif challenge_res == "fail":
                counter_try += 1
                show_image(camera, question + " : fail")
            elif try_index == limit_try - 1:
                break

        if counter_ok_questions == limit_questions:
            while True:
                image = show_image(camera, "LIVENESS SUCCESSFUL", color=(0, 255, 0))
                cv2.imshow('Scenario 1 in Active Liveness Detection', image)

                score = get_mnet_score(save_path)
                print("MNet Score = ", score)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif try_index == limit_try - 1:
            while True:
                image = show_image(camera, "LIVENESS FAIL")
                cv2.imshow('Scenario 1 in Active Liveness Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

        else:
            continue


def main_pipeline2():
    cv2.namedWindow("Scenario 2 in Active Liveness Detection")
    camera = cv2.VideoCapture(0)
    _, frame = camera.read()
    mask = np.zeros_like(frame)
    limit_try = 500
    old_point, new_point = None, None

    draw_number = random.randint(0, 100)
    for _ in range(limit_try):
        _, image = camera.read()
        if image is None:
            continue

        face_bbox = get_bbox(image)
        if face_bbox is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        points = get_face_landmark(gray, face_bbox)
        draw_points = points.detach().numpy().reshape(-1, 1, 2)

        if old_point is None:
            old_point = draw_points[30]
            continue
        else:
            new_point = draw_points[30]
            cv2.line(mask, (new_point[0][0], new_point[0][1]), (old_point[0][0], old_point[0][1]), (255, 255, 255), 4)
            old_point = new_point

        image = show_image(camera, f"Please draw number: {draw_number}")
        _ = cv2.add(image, mask)
        cv2.imshow("Scenario 2 in Active Liveness Detection", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    digit_image = cv2.bitwise_not(mask)
    cv2.imshow("digit image in Active Liveness Detection", digit_image)
    main(digit_image)


def get_active_liveness_status(image, question, blink_count, total_count):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_bbox = face_detector.get_bbox(image)
    if face_bbox is None:
        return None

    points = get_face_landmark(gray, face_bbox)
    out_model = get_passive_result(image, face_bbox, points, question, blink_count, total_count)
    return out_model


if __name__ == '__main__':
    main_pipeline1()
