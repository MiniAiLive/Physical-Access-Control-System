import os
import torch


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def main():
    device = torch.device("cpu:0")
    model_dir = "checkpoints"

    quantized_model_filename = "resnet50_quantized_casia.pt"
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=device)
    print(quantized_jit_model)

    quantized_jit_model.to(device)
    quantized_jit_model.eval()

    x = torch.rand(size=(1, 3, 112, 112)).to(device)

    with torch.no_grad():
        quantized_jit_model.quant(x)
        print(quantized_jit_model)


def convert_fuse_model():
    backbone_path = 'checkpoints/resnet50_Iter_486000_net.ckpt'
    margin_path = 'checkpoints/resnet50_Iter_486000_margin.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = None #FuseResNet50()
    model.backbone.load_state_dict(torch.load(backbone_path, map_location=device)['net_state_dict'])
    model.margin.load_state_dict(torch.load(margin_path, map_location=device)['net_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    torch.onnx.export(model, (dummy_input, 0), 'fuse.onnx', verbose=True, input_names=["input", "label"],
                      output_names=["output"], training=False, opset_version=11)


if __name__ == '__main__':
    # main()
    convert_fuse_model()
