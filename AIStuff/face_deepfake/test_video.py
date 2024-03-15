import sys
import os
import argparse
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from model.models import model_selection
from dataloader.transform import xception_default_data_transforms
from src.bioauth_ml.face_detect.test import get_bbox
sys.path.append('../../..')


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face[0], face[1], face[0] + face[2], face[1] + face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        size_bb = max(size_bb, minsize)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print(f'Starting: {video_path}')

    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda:
        model = model.cuda()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        height, width = image.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(os.path.join(output_path, video_fn), fourcc, fps, (height, width)[::-1])

        face = get_bbox(image)
        if len(face):

            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
            x, y, w, h = face[0], face[1], face[2], face[3]

            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = [f'{float(x):.2f}' for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30), font_face, font_scale, color, thickness, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        cv2.imshow('test', image)
        cv2.waitKey(33)
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print(f'Finished! Output saved under {output_path}')
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = os.path.join(video_path, video)
            test_full_image_network(**vars(args))
