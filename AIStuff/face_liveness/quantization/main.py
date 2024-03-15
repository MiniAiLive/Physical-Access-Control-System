import os
import sys
sys.path.append('../../../..')
import logging
import argparse
import onnx
import neural_compressor as inc
from neural_compressor.experimental import Benchmark, Quantization, common
from neural_compressor import options
from src.bioauth_ml.face_liveness.dataloader.dataset_loader import get_test_loader
from src.bioauth_ml.face_liveness.cfgs.default_config import get_default_config, update_config

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.WARN)


def parse_args():
    desc = "MiniFASNet based on Liveness Detection"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="0", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="2.7_128x128",
                        help="[org_1_128x128 / 1_128x128 / 2.7_128x128 / 4_128x128]")
    parser.add_argument('--model_path', default='../checkpoints/anti_spoof_models_v1_onnx/2.7_128x128_MiniFASNetV2SE.onnx',
                        type=str, help="Pre-trained vgg16 model on onnx file")
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False, help="whether quantize the model")
    parser.add_argument('--config', default='onnx_quantization.yaml', type=str, help="config yaml path")
    parser.add_argument('--output_model', default='./checkpoints.onnx', type=str, help="output model path")
    parser.add_argument('--mode', type=str, help="benchmark mode of performance or accuracy")

    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    return args


def ver2int(ver):
    s_vers = ver.split(".")
    res = 0
    for i, s in enumerate(s_vers):
        res += int(s)*(100**(2-i))

    return res


def compare_ver(src, dst):
    src_ver = ver2int(src)
    dst_ver = ver2int(dst)
    if src_ver>dst_ver:
        return 1
    if src_ver<dst_ver:
        return -1
    return 0


if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    args = parse_args()
    conf = get_default_config()
    conf = update_config(args, conf)

    model = onnx.load("checkpoints.onnx")
    intermediate_tensor_name = "output_quantized"
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    model.graph.output.extend([intermediate_layer_value_info])
    onnx.save(model, "2.7_128x128_MiniFASNetV2SE_Quant.onnx")

    if args.benchmark:
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator(args.mode)

    if args.tune:
        options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

        quantizer = Quantization(args.config)
        quantizer.calib_dataloader = common.DataLoader(get_test_loader(), batch_size=1)
        quantizer.eval_dataloader = common.DataLoader(get_test_loader(), batch_size=1)
        quantizer.model = common.Model(model)

        if compare_ver(inc.__version__, "1.9") >= 0:
            q_model = quantizer.fit()
        else:
            q_model = quantizer()

        q_model.save(args.output_model)
