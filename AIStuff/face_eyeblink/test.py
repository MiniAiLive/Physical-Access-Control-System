import os
import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import src.bioauth_ml.face_eyeblink.model.feb as model
from src.bioauth_ml.face_expression.visualize.grad_cam import BackPropagation, GradCAM, GuidedBackPropagation

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
shape = (24, 24)
classes = [
    'Close',
    'Open',
]

def preprocess(image_path):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        print('no face found')
        face = cv2.resize(image, shape)
        return None, face
    else:
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        (x, y, w, h) = image_path['left']
        left_eye = face[y:y + h, x:x + w]
        left_eye = cv2.resize(left_eye, shape)
        (x, y, w, h) = image_path['right']
        right_eye = face[y:y + h, x:x + w]
        right_eye = cv2.resize(right_eye, shape)
        return transform_test(Image.fromarray(left_eye).convert('L')), \
               transform_test(Image.fromarray(right_eye).convert('L')), \
               left_eye, right_eye, cv2.resize(face, (48,48))


def get_eye_image(frame, left_bbox, right_bbox):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    left_eye = frame[left_bbox[1]:left_bbox[1] + left_bbox[3], left_bbox[0]:left_bbox[0] + left_bbox[2]]
    if left_eye.size == 0:
        return None

    left_eye = cv2.resize(left_eye, shape)
    right_eye = frame[right_bbox[1]:right_bbox[1] + right_bbox[3], right_bbox[0]:right_bbox[0] + right_bbox[2]]
    if right_eye.size == 0:
        return None

    right_eye = cv2.resize(right_eye, shape)

    return transform_test(Image.fromarray(left_eye).convert('L')), \
                   transform_test(Image.fromarray(right_eye).convert('L')), \
                   left_eye, right_eye


def get_eye_bbox(landmark, size):
    height, width, _ = size
    padding_rate = 1.6
    left_eye_center_x = int((landmark[74] + landmark[76] + landmark[80] + landmark[82]) / 4)
    left_eye_center_y = int((landmark[75] + landmark[77] + landmark[81] + landmark[83]) / 4)
    left_eye_size = int((landmark[78] - landmark[72]) * padding_rate)
    left_corner_x = int(left_eye_center_x - left_eye_size / 2)
    left_corner_x = 0 if left_corner_x < 0 else left_corner_x
    left_corner_y = int(left_eye_center_y - left_eye_size / 2)
    left_corner_y = 0 if left_corner_y < 0 else left_corner_y
    left_eye_size = width - left_corner_x - 1 if left_corner_x + left_eye_size >= width else left_eye_size
    left_eye_size = height - left_corner_y - 1 if left_corner_y + left_eye_size >= height else left_eye_size

    right_eye_center_x = int((landmark[86] + landmark[88] + landmark[92] + landmark[94]) / 4)
    right_eye_center_y = int((landmark[87] + landmark[89] + landmark[93] + landmark[95]) / 4)
    right_eye_size = int((landmark[90] - landmark[84]) * padding_rate)
    right_corner_x = int(right_eye_center_x - right_eye_size / 2)
    right_corner_x = 0 if right_corner_x < 0 else right_corner_x
    right_corner_y = int(right_eye_center_y - right_eye_size / 2)
    right_corner_y = 0 if right_corner_y < 0 else right_corner_y
    right_eye_size = width - right_corner_x - 1 if right_corner_x + right_eye_size >= width else right_eye_size
    right_eye_size = height - right_corner_y - 1 if right_corner_y + right_eye_size >= height else right_eye_size

    return [left_corner_x, left_corner_y, left_eye_size, left_eye_size], [right_corner_x, right_corner_y, right_eye_size, right_eye_size]


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


def predict_eye_blink(image):
    net = model.FEBNet(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/eyeblink.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()

    img = torch.stack([image])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)

    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    print("[predict_eye_blink]: ", prob*100, classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]


def guided_backprop_eye(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    gcam = GradCAM(model=net)
    _ = gcam.forward(img)

    gbp = GuidedBackPropagation(model=net)
    _ = gbp.forward(img)

    # Guided Backpropagation
    actual_status = ids[:, 0]
    gbp.backward(ids=actual_status.reshape(1, 1))
    gradients = gbp.generate()

    # Grad-CAM
    gcam.backward(ids=actual_status.reshape(1, 1))
    regions = gcam.generate(target_layer='last_conv')

    # Get Images
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    prob_image = np.zeros((shape[0], 60, 3), np.uint8)
    cv2.putText(prob_image, f'{prob * 100:.01}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

    guided_bpg_image = get_gradient_image(gradients[0])
    guided_bpg_image = cv2.merge((guided_bpg_image, guided_bpg_image, guided_bpg_image))

    grad_cam_image = get_gradcam_image(gcam=regions[0, 0], raw_image=image[name + '_raw'])
    guided_gradcam_image = get_gradient_image(torch.mul(regions, gradients)[0])
    guided_gradcam_image = cv2.merge((guided_gradcam_image, guided_gradcam_image, guided_gradcam_image))
    print(image['path'],classes[actual_status.data], probs.data[:,0] * 100)

    return cv2.hconcat(
        [image[name + '_raw'], prob_image, guided_bpg_image, grad_cam_image, guided_gradcam_image])


def guided_backprop(images, model_name):
    net = model.FEBNet(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('checkpoints', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    summary(net, (1, shape[0], shape[1]))

    result_images = []
    for _, image in enumerate(images):
        left = guided_backprop_eye(image, 'left', net)
        right = guided_backprop_eye(image, 'right', net)
        eyes = cv2.vconcat([left,right])
        image = cv2.hconcat([image['face'],eyes])
        result_images.append(image)

    cv2.imwrite('../test/guided_gradcam.jpg',cv2.resize(cv2.vconcat(result_images), None, fx=2,fy=2))


def convert_to_onnx():
    net = model.FEBNet(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/model_11_96_0.1256.t7'),
                            map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()

    dummy_input = torch.randn(1, 1, 24, 24)
    torch.onnx.export(net, dummy_input, "checkpoints/eye.onnx", verbose=True, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
    convert_to_onnx()
