import sys
import argparse
import cv2
import torch
from PIL import Image
from model.models import model_selection
from dataloader.transform import xception_default_data_transforms
from src.bioauth_ml.face_detect.test import get_bbox
sys.path.append('../../..')


def main():
    args = parse.parse_args()
    model_path = args.model_path

    torch.backends.cudnn.benchmark = True

    # model = torchvision.models.densenet121(num_classes=2)
    model = model_selection(modelname='resnet18', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()
    with torch.no_grad():

        image = cv2.imread(args.image_file)
        face_bbox = get_bbox(image)
        if len(face_bbox):
            x1, y1, x2, y2 = face_bbox[0], face_bbox[1], face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]
            crop_img = image[y1:y2, x1:x2]

            image = xception_default_data_transforms['test'](Image.fromarray(crop_img))
            image = image.unsqueeze(0)
            image = image.cuda()

            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            print("result: ", preds)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--image_file', type=str, default='./datasets/test_images/hu.jpg')
    parse.add_argument('--model_path', type=str, default='./checkpoints/output/fs_resnet18/fs_resnet18.pkl')
    main()
