import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Conv2dQ(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode: str=  "zeros",
            a_bits=8,
            w_bits=8,
            first_layer=False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationQuantize(a_bits=a_bits)
        self.weight_quantizer = WeightQuantize(w_bits=w_bits)
        self.first_layer = first_layer

    def forward(self, x):
        # 量化A和W
        if not self.first_layer:
            x = self.activation_quantizer(x)
        q_input = x
        q_weight = self.weight_quantizer(self.weight)
        # 量化卷积
        output = F.conv2d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output


class WeightQuantize(nn.Module):
    def __init__(self, w_bits):
        super().__init__()
        self.w_bits = w_bits

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.w_bits != 1, '！Binary quantization is not supported ！'
        if self.w_bits == 32:
            output = x
        else:
            # 按照公式9和10计算
            output = torch.tanh(x)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = float(2 ** self.w_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
            output = 2 * output - 1
        return output


class ActivationQuantize(nn.Module):
    def __init__(self, a_bits):
        super().__init__()
        self.a_bits = a_bits

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.a_bits != 1, '！Binary quantization is not supported ！'
        if self.a_bits == 32:
            output = x
        else:
            output = torch.clamp(x * 0.1, 0, 1)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            scale = float(2 ** self.a_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
        return output


class Round(Function):

    @staticmethod
    def forward(self, x):
        output = torch.round(x)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
