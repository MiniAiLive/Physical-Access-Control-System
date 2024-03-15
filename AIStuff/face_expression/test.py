import os
import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
from torchsummary import summary
import src.bioauth_ml.face_expression.model.fer as model
from src.bioauth_ml.face_expression.model.rmm import RMN
from src.bioauth_ml.face_expression.visualize.grad_cam import BackPropagation, GradCAM,GuidedBackPropagation


rmm_net = RMN()
shape = (48,48)
classes = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]


def get_face_image(frame, face_bbox):
    size = int(face_bbox[2] * 1.2)
    left = int(face_bbox[0] - 0.1 * size)
    top = int(face_bbox[1] - 0.1 * size)
    face = frame[top:top+size, left:left+size]
    return face


def get_gradient_image(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return np.uint8(gradient)


def get_gradcam_image(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return np.uint8(gcam)


def predict_expression(image):
    pred_label, _, _ = rmm_net.detect_emotion_for_single_face_image(image)
    # print("[predict_expression]: ", prob*100, classes[actual_emotion.data], probs.data[:,0] * 100)
    return pred_label


def guided_backprop(images, model_name):

    for _, image in enumerate(images):
        target, raw_image = None, None #preprocess(image['path'])
        image['image'] = target
        image['raw_image'] = raw_image

    net = model.FERNet(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('checkpoints', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    summary(net, (1, shape[0], shape[1]))

    result_images = []
    for _, image in enumerate(images):
        img = torch.stack([image['image']])
        bp = BackPropagation(model=net)
        probs, ids = bp.forward(img)
        gcam = GradCAM(model=net)
        _ = gcam.forward(img)

        gbp = GuidedBackPropagation(model=net)
        _ = gbp.forward(img)

        # Guided Backpropagation
        actual_emotion = ids[:,0]
        gbp.backward(ids=actual_emotion.reshape(1,1))
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=actual_emotion.reshape(1,1))
        regions = gcam.generate(target_layer='last_conv')

        # Get Images
        label_image = np.zeros((shape[0],65, 3), np.uint8)
        cv2.putText(label_image, classes[actual_emotion.data], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        prob_image = np.zeros((shape[0],60,3), np.uint8)
        cv2.putText(prob_image, f'{probs.data[:,0] * 100:.02}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        guided_bpg_image = get_gradient_image(gradients[0])
        guided_bpg_image = cv2.merge((guided_bpg_image, guided_bpg_image, guided_bpg_image))

        grad_cam_image = get_gradcam_image(gcam=regions[0, 0],raw_image=image['raw_image'])

        guided_gradcam_image = get_gradient_image(torch.mul(regions, gradients)[0])
        guided_gradcam_image = cv2.merge((guided_gradcam_image, guided_gradcam_image, guided_gradcam_image))

        img = cv2.hconcat([image['raw_image'],label_image,prob_image,guided_bpg_image,grad_cam_image,guided_gradcam_image])
        result_images.append(img)
        print(image['path'],classes[actual_emotion.data], probs.data[:,0] * 100)

    cv2.imwrite('../test/guided_gradcam.jpg',cv2.resize(cv2.vconcat(result_images), None, fx=2,fy=2))


def convert_to_onnx():
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(rmm_net.emo_model, dummy_input, "checkpoints/expression.onnx", verbose=True, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
    convert_to_onnx()
