import os
import sys
import json
import cv2
from tqdm import tqdm
from src.bioauth_ml.face_detect.test import get_bbox
sys.path.append('../../..')


def extract():
    train_frame_folder = './datasets/train_sample_videos'
    with open(os.path.join(train_frame_folder, 'metadata.json'), 'r') as file:
        data = json.load(file)

    list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]

    for vid in tqdm(list_of_train_data):
        count = 0
        cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
        frameRate = cap.get(5)
        while cap.isOpened():
            frameId = cap.get(1)
            ret, frame = cap.read()
            if ret is not True:
                break

            if frameId % ((int(frameRate)+1)*1) == 0:
                face_rects = get_bbox(frame)
                if face_rects is not None:
                    x1, y1, x2, y2 = face_rects[0], face_rects[1], face_rects[0] + face_rects[2], face_rects[1] + face_rects[3]
                    crop_img = frame[y1:y2, x1:x2]
                    try:
                        if data[vid]['label'] == 'REAL':
                            cv2.imwrite('datasets/rgb_image/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                        elif data[vid]['label'] == 'FAKE':
                            cv2.imwrite('datasets/rgb_image/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                        count += 1
                    except:
                        print("failed video file: ", vid)


if __name__ == '__main__':
    extract()
