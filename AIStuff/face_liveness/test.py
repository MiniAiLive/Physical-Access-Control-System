"""
@author: XiangLan
@file: test.py
@time: 2022/12/20 10:20
@desc: test the model for face liveness detection
"""

import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision import transforms
from face_liveness.model.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from face_liveness.model.FASNetA import FASNetA
from face_liveness.model.FASNetB import FASNetB
from face_liveness.model.FASNetC import FASNetCV1
from face_liveness.model.FeatherNet import FeatherNetA
from face_liveness.tools.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class LivenessEstimator():
    """ define the all FAS models and load them """
    def __init__(self, model_dir, device_id):
        super(LivenessEstimator, self).__init__()
        self.device = torch.device(f"cuda:{device_id}"
                                   if torch.cuda.is_available() else "cpu")

        self._load_model_all(model_dir)

    def _load_model_all(self, model_dir):
        """ load all model for minifasnet """
        pretrained_path = os.path.join(os.path.dirname(__file__),
                                       "checkpoints/anti_spoof_models_v1/2.7_128x128_MiniFASNetV2SE.pth")

        params = {
            'embedding_size': 128,
            'conv6_kernel': (8, 8),
            'drop_p': 0.75,
            'num_classes': 3,
            'img_channel': 3
        }

        self.model_1 = MiniFASNetV2SE(**params).to(self.device)

        if os.path.isfile(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=self.device)

            new_state_dict = OrderedDict()
            for key, value in state_dict["state_dict"].items():
                name_key = key[4:]
                if name_key.find('model.') >= 0:
                    name_key = name_key[6:]
                    new_state_dict[name_key] = value

            self.model_1.load_state_dict(new_state_dict)

        model_2_name = '4_0_0_128x128_MiniFASNetV2SE.pth'
        h_input, w_input, model_type, _ = parse_model_name(model_2_name)
        self.kernel_size = get_kernel(h_input, w_input, )

        state_dict = torch.load(os.path.join(model_dir, model_2_name), map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                if name_key.find('model.') >= 0 and model_type == str('MiniFASNetV2SE'):
                    name_key = name_key[6:]

                new_state_dict[name_key] = value

            if model_type == str('MiniFASNetV2SE'):
                params = {
                    'embedding_size': 128,
                    'conv6_kernel': (8, 8),
                    'drop_p': 0.75,
                    'num_classes': 3,
                    'img_channel': 3
                }

                self.model_2 = MiniFASNetV2SE(**params).to(self.device)

            self.model_2.load_state_dict(new_state_dict, strict=False)
        else:
            self.model_2.load_state_dict(state_dict)

        model_3_name = 'org_1_128x128_MiniFASNetV2SE.pth'
        h_input, w_input, model_type, _ = parse_model_name(model_3_name)
        self.kernel_size = get_kernel(h_input, w_input, )
        # self.model_3 = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        state_dict = torch.load(os.path.join(model_dir, model_3_name), map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                if name_key.find('model.') >= 0 and model_type == str('MiniFASNetV2SE'):
                    name_key = name_key[6:]

                new_state_dict[name_key] = value

            if model_type == str('MiniFASNetV2SE'):
                params = {
                    'embedding_size': 128,
                    'conv6_kernel': (8, 8),
                    'drop_p': 0.75,
                    'num_classes': 3,
                    'img_channel': 3
                }

                self.model_3 = MiniFASNetV2SE(**params).to(self.device)

            self.model_3.load_state_dict(new_state_dict, strict=False)
        else:
            self.model_3.load_state_dict(state_dict)

        model_4_name = '1_128x128_MiniFASNetV2SE.pth'
        h_input, w_input, model_type, _ = parse_model_name(model_4_name)
        self.kernel_size = get_kernel(h_input, w_input, )
        # self.model_3 = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        state_dict = torch.load(os.path.join(model_dir, model_4_name), map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                if name_key.find('model.') >= 0 and model_type == str('MiniFASNetV2SE'):
                    name_key = name_key[6:]

                new_state_dict[name_key] = value

            if model_type == str('MiniFASNetV2SE'):
                params = {
                    'embedding_size': 128,
                    'conv6_kernel': (8, 8),
                    'drop_p': 0.75,
                    'num_classes': 3,
                    'img_channel': 3
                }

                self.model_4 = MiniFASNetV2SE(**params).to(self.device)

            self.model_4.load_state_dict(new_state_dict, strict=False)
        else:
            self.model_4.load_state_dict(state_dict)

        pretrained_path = os.path.join(os.path.dirname(__file__),
                                       "checkpoints/anti_spoof_models_v2/fasnet_c/2.7_128x128_FASNetCV1.pth")
        self.fasnetcv1_model = FASNetCV1()
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            new_state_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                name_key = key[4:]
                new_state_dict[name_key] = value

            self.fasnetcv1_model.to(self.device)
            self.fasnetcv1_model.load_state_dict(new_state_dict)

        pretrained_path = os.path.join(os.path.dirname(__file__),
                                       "checkpoints/anti_spoof_models_v2/fasnet_a/2.7_128x128_FASNetA.pth")
        self.fasneta_model = FASNetA()
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            new_state_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                name_key = key[4:]
                new_state_dict[name_key] = value

            self.fasneta_model.to(self.device)
            self.fasneta_model.load_state_dict(new_state_dict, strict=False)

        pretrained_path = os.path.join(os.path.dirname(__file__),
                                       "checkpoints/anti_spoof_models_v2/fasnet_b/2.7_128x128_FASNetB.pth")
        self.fasnetb_model = FASNetB()
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            new_state_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                name_key = key[4:]
                new_state_dict[name_key] = value

            self.fasnetb_model.to(self.device)
            self.fasnetb_model.load_state_dict(new_state_dict, strict=False)

        pretrained_path = os.path.join(os.path.dirname(__file__),
                                       "checkpoints/anti_spoof_models_v2/fnet/2.7_128x128_fnet.pth")
        self.fnet_model = FeatherNetA()
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            new_state_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                name_key = key[4:]
                new_state_dict[name_key] = value

            self.fnet_model.to(self.device)
            self.fnet_model.load_state_dict(new_state_dict)

    def _load_model(self, model_path):
        """ load only MiniFASNetV2SE model """
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                if name_key.find('model.') >= 0 and model_type == str('MiniFASNetV2SE'):
                    name_key = name_key[6:]

                new_state_dict[name_key] = value

            if model_type == str('MiniFASNetV2SE'):
                params = {
                    'embedding_size': 128,
                    'conv6_kernel': (8, 8),
                    'drop_p': 0.75,
                    'num_classes': 3,
                    'img_channel': 3
                }

                self.model = MiniFASNetV2SE(**params)

            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

    def predict(self, img, model_name):
        """ get the score of the ensemble networks """
        # transform = trans.Compose([
        #     trans.ToTensor(),
        # ])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = transform(img)
        img = img.unsqueeze(0).to(self.device)
        if model_name == str('2.7_128x128_MiniFASNetV2SE.pth'):
            self.model_1.eval()
            with torch.no_grad():
                result = self.model_1.forward(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('4_0_0_128x128_MiniFASNetV2SE.pth'):
            self.model_2.eval()
            with torch.no_grad():
                result = self.model_2.forward(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('org_1_128x128_MiniFASNetV2SE.pth'):
            self.model_3.eval()
            with torch.no_grad():
                result = self.model_3.forward(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('1_128x128_MiniFASNetV2SE.pth'):
            self.model_4.eval()
            with torch.no_grad():
                result = self.model_4.forward(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('2.7_128x128_FASNetA.pth'):
            self.fasneta_model.eval()
            with torch.no_grad():
                result = self.fasneta_model(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('2.7_128x128_FASNetCV1.pth'):
            self.fasnetcv1_model.eval()
            with torch.no_grad():
                result = self.fasnetcv1_model(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('2.7_128x128_FASNetB.pth'):
            self.fasnetb_model.eval()
            with torch.no_grad():
                result = self.fasnetb_model(img)
                result = F.softmax(result).cpu().numpy()
        elif model_name == str('2.7_128x128_fnet.pth'):
            self.fnet_model.eval()
            with torch.no_grad():
                result = self.fnet_model(img)
                result = F.softmax(result).cpu().numpy()
        else:
            return None

        return result

    def convert_to_onnx(self, model_path):
        """ convert pytorch model to onnx one """
        self._load_model(model_path)
        self.model.eval()

        onnx_model_path = model_path.replace('pth', 'onnx')
        dummy_input = torch.randn(1, 3, 128, 128).to(self.device)
        torch.onnx.export(self.model, dummy_input, onnx_model_path, verbose=True, input_names=["input"],
                          output_names=["output"], training=False, opset_version=11)


if __name__ == '__main__':
    liveness_estimator = LivenessEstimator(os.path.join(os.path.dirname(__file__), "checkpoints/anti_spoof_models_v1"), 0)
    liveness_estimator.convert_to_onnx(os.path.join(os.path.dirname(__file__),
                                                    "checkpoints/anti_spoof_models_v1/2.7_128x128_MiniFASNetV2SE.pth"))
