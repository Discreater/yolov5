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
            padding_mode: str = "zeros",

            w_bits=8
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
        self.weight_quantizer = WeightQuantize(bits=w_bits)

    def forward(self, x):
        q_weight = self.weight_quantizer(self.weight)
        # 量化卷积
        output = F.conv2d(
            input=x,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output


class ConvBnReLU2dQ(Conv2dQ):
    """
    see https://arxiv.org/abs/1806.08342
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 a_bits=8,
                 a_scale_bits=4,
                 w_bits=8,
                 eps=1e-5,
                 momentum=0.01):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias=False, padding_mode="zeros", w_bits=w_bits)
        self.bn = nn.BatchNorm2d(out_channels, eps, momentum)
        self.act_quantizer = ActivationQuantize(bits=a_bits, scale_bits=a_scale_bits)

    def forward(self, x):
        conv_res = F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        batch_mean = torch.mean(conv_res, dim=[0, 2, 3])
        batch_var = torch.var(conv_res, dim=[0, 2, 3])
        _ = self.bn(conv_res)

        # gamma / sqrt(running_var + eps). Here use running var, and it will be scaled back later.
        # See https://arxiv.org/abs/1806.08342 Figure 9
        weight = self.weight * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).view(-1, 1, 1, 1)
        # gamma / sqrt(batch_var + eps)
        bias = (self.bn.bias - (self.bn.weight * batch_mean) / torch.sqrt(batch_var + self.bn.eps)).view(-1)

        # quantize weight
        weight = self.weight_quantizer(weight)

        # scale: sqrt(running_var + eps) / sqrt(batch_var + eps)
        scale = torch.sqrt((self.bn.running_var + self.bn.eps) / (batch_var + self.bn.eps))
        bias = bias / scale

        x = F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        x *= scale.view(1, -1, 1, 1)
        x = self.act_quantizer(x)
        return x

    def fuse(self):
        running_var = self.bn.running_var
        running_mean = self.bn.running_mean

        w_bn = self.bn.weight / (torch.sqrt(running_var + self.bn.eps))
        weight = self.weight * w_bn.view(-1, 1, 1, 1)
        weight = self.weight_quantizer(weight)
        self.weight = weight

        b_bn = self.bn.bias - self.bn.weight * running_mean / (torch.sqrt(running_var + self.bn.eps))
        self.bias = b_bn
        self.forward = self.fused_forward

    def fused_forward(self, x):
        x = F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        x = self.act_quantizer(x)
        return x


class WeightQuantize(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def extra_repr(self) -> str:
        s = "bits={}".format(self.bits)
        return s

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.bits != 1, '！Binary quantization is not supported ！'
        if self.bits == 32:
            output = x
        else:
            # 按照公式9和10计算
            output = torch.tanh(x)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = float(2 ** self.bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
            output = 2 * output - 1
        return output


class ActivationQuantize(nn.Module):
    def __init__(self, bits, scale_bits):
        super().__init__()
        self.bits = bits
        self.scale_bits = scale_bits

    def extra_repr(self) -> str:
        s = "bits={}, scale_bits={}".format(self.bits, self.scale_bits)
        return s

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.bits != 1, '！Binary quantization is not supported ！'
        if self.bits == 32:
            output = x
        else:
            output = torch.clamp(x / (1 << self.scale_bits), 0, 1)  # 特征A截断前先进行缩放 >> scale_shift，以减小截断误差
            scale = float(2 ** self.bits - 1)
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
