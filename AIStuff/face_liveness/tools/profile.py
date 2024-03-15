"""
define the utility module for liveness detection
"""
import torch
from torch.autograd.variable import Variable
import numpy as np


USE_GPU = torch.cuda.is_available()


def calc_flops(model, input_size):

    def conv_hook(self, x, output):
        batch_size, _, _, _ = x[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, x, output):
        batch_size = x[0].size(0) if x[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, x, output):
        list_bn.append(x[0].nelement())

    def relu_hook(self, x, output):
        list_relu.append(x[0].nelement())

    def pooling_hook(self, x, output):
        batch_size, _, _, _ = x[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def test(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            test(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    test(model)
    if '0.4.' in torch.__version__:
        if USE_GPU:
            x = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
        else:
            x = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
    else:
        x = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    _ = model(x)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print(f'  + Number of FLOPs: {total_flops / 1e6 / 2}M')


def count_params(model, input_size=224):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))
    calc_flops(model, input_size)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f'The model has {params} params.')
