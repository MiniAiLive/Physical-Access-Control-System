import sys
sys.path.append('../../../..')
import logging
import argparse
import onnx
import neural_compressor as inc
from neural_compressor.experimental import Benchmark, Quantization, common
from neural_compressor import options
from src.bioauth_ml.face_feature.dataloader.casia_webface import CASIAWebFaceDataset
from src.bioauth_ml.face_liveness.quantization.main import compare_ver

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.WARN)


def parse_args():
    desc = "Resnet based on Feature Extraction"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--restore", action='store_true', default=False, help="restore onnx model to get int8 output")
    parser.add_argument('--model_path', default='../checkpoints/resnet50_Iter_486000_net.onnx',
                        type=str, help="Pre-trained vgg16 model on onnx file")
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=True, help="whether quantize the model")
    parser.add_argument('--config', default='onnx_quantization.yaml', type=str, help="config yaml path")
    parser.add_argument('--output_model', default='./checkpoints.onnx', type=str, help="output model path")
    parser.add_argument('--mode', type=str, help="benchmark mode of performance or accuracy")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    args = parse_args()

    model = onnx.load(args.model_path)

    if args.benchmark:
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator(args.mode)

    if args.tune:
        root = '/datasets/public2/upload/faces_emore_images'
        file_list = '/datasets/public2/upload/faces_emore/faces_emore.list'

        train_dataset = CASIAWebFaceDataset(root, file_list)

        options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

        quantizer = Quantization(args.config)

        quantizer.calib_dataloader = common.DataLoader(train_dataset, batch_size=1)
        quantizer.eval_dataloader = common.DataLoader(train_dataset, batch_size=1)
        quantizer.model = common.Model(model)

        if compare_ver(inc.__version__, "1.9") >= 0:
            q_model = quantizer.fit()
        else:
            q_model = quantizer()

        q_model.save(args.output_model)

    if args.restore:
        model = onnx.load("checkpoints.onnx")
        intermediate_tensor_name = "515_quantized"
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = intermediate_tensor_name
        model.graph.output.extend([intermediate_layer_value_info])
        onnx.save(model, "resnet50_Quant.onnx")
