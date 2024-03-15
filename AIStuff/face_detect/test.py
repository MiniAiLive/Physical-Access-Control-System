import os
import sys
sys.path.append(os.path.dirname(__file__))

import cv2
import math
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data.config import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

trained_model = os.path.join(os.path.dirname(__file__), './checkpoints/FaceBoxesProd.pth')
save_folder = 'eval'
dataset = 'Custom'
confidence_threshold = 0.2
top_k = 5000
nms_threshold = 0.3
keep_top_k = 750
show_image = True
vis_thres = 0.5


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=device)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = FaceBoxes(phase='test', size=None, num_classes=2)
net = load_model(net, trained_model, device)
net.eval()
cudnn.benchmark = True
net = net.to(device)


def get_bbox(orig_image):
    # testing scale
    resize = 0.5

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    img_raw = orig_image
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #keep = py_cpu_nms(dets, nms_threshold)
    keep = nms(dets, nms_threshold, force_cpu=True)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    _t['misc'].toc()

    boxes, scores = [], []
    for k in range(dets.shape[0]):
        xmin = dets[k, 0]
        ymin = dets[k, 1]
        xmax = dets[k, 2]
        ymax = dets[k, 3]
        ymin += 0.2 * (ymax - ymin + 1)
        score = dets[k, 4]
        boxes.append([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)])
        scores.append(score)

    max_score = 0.0
    final_box = None
    for i, score in enumerate(scores):
        if max_score < score:
            max_score = score
            final_box = boxes[i]

    return final_box


class Detection:
    def __init__(self):
        src_dir = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(src_dir, "checkpoints")):
            os.makedirs(os.path.join(src_dir, "checkpoints"))

        caffemodel = os.path.join(src_dir, "checkpoints/Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(src_dir, "checkpoints/deploy.prototxt")

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height

        if right == left or bottom == top:
            return None

        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

    def check_face(self):
        pass


if __name__ == '__main__':
    image = cv2.imread('multi47_60.jpg')

    box = get_bbox(image)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    image1 = cv2.imread('multi47_60.jpg')
    detector = Detection()
    box = detector.get_bbox(image1)
    cv2.rectangle(image1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

    cv2.imshow('result', image)
    cv2.imshow('result1', image1)
    cv2.waitKey(0)
