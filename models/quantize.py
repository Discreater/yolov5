import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter


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
            w_bits: int = 4,
            w_signed: bool = True,
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
        self.w_bits = w_bits
        self.w_signed = w_signed
        self.weight_quantizer = WeightQuantize(bits=w_bits, signed=w_signed)

    def extra_repr(self):
        s = super(Conv2dQ, self).extra_repr()
        s += ", w_bits={}, w_signed={}".format(self.w_bits, self.w_signed)
        return s

    def forward(self, x):
        # quantize weight
        q_weight = self.weight_quantizer(self.weight)

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

    def fused_forward(self, x):
        output = F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output


class WeightQuantize(nn.Module):
    bits: int

    def __init__(self, bits, signed=False):
        super().__init__()
        self.bits = bits
        self.signed = signed

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.bits != 1, '！Binary quantization is not supported ！'
        if self.bits == 32:
            output = torch.tanh(x)
            output = output / torch.max(torch.abs(output))
        else:
            # 按照公式9和10计算
            output = torch.tanh(x)

            if not self.signed:
                output = output / 2 / torch.max(torch.abs(output)) + 0.5  # -> [0,1]
                scale = float(2 ** self.bits)
            else:
                output = output / torch.max(torch.abs(output))  # -> [-1, 1]
                scale = float(2 ** self.bits - 1)

            # quantize
            output = output * scale
            output = self.round(output)
            output = output / scale

            if not self.signed:
                output = 2 * output - 1
        return output


class ActivationQuantize(nn.Module):
    def __init__(self, a_bits, scale_shift=4):
        """
        Args:
            a_bits (int): 量化位数
            scale (int): 截断前放缩系数
        """
        super().__init__()
        self.a_bits = a_bits
        self.scale_shift = scale_shift

    def extra_repr(self) -> str:
        s = "a_bits={}, scale_shift={}".format(self.a_bits, self.scale_shift)
        return s

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.a_bits != 1, '！Binary quantization is not supported ！'
        if self.a_bits == 32:
            output = torch.clamp(x, 0, 6)  # ReLU6
        else:
            output = torch.clamp(x / (1 << self.scale_shift), 0, 1)  # 特征A截断前先进行缩放（x / (2**scale_shift)），以减小截断误差
            scale = float(2 ** self.a_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
        return output


class BnActFused(nn.Module):
    def __init__(self, ch, w_bit, in_bit, out_bit, scale_shift, signed=True):
        super(BnActFused, self).__init__()
        assert signed
        self.ch = ch
        self.weight = torch.Tensor(ch)
        self.bias = torch.Tensor(ch)
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.scale_shift = scale_shift
        self.signed = signed

    def extra_repr(self) -> str:
        s = "ch={}, w_bit={}, in_bit={}, out_bit={}, scale_shift={}, signed={}".format(self.ch, self.w_bit, self.in_bit,
                                                                                       self.out_bit, self.scale_shift,
                                                                                       self.signed)
        return s

    def forward(self, x):
        assert not self.training
        x = x * self.weight + self.bias
        x = x >> (self.w_bit - 1 + self.in_bit + self.scale_shift)
        x = torch.clamp(x, min=0, max=((1 << self.out_bit) - 1))
        return x


class Round(Function):

    @staticmethod
    def forward(self, x):
        output = torch.round(x)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def fuse_bn(gamma, beta, mean, var, eps):
    w = gamma / (torch.sqrt(var + eps))
    b = beta - w * mean
    return w, b


def quantize_bn(gamma, beta, mean, var, eps, w_bit, in_bit, out_bit, scale_shift, signed=True):
    assert signed
    w, b = fuse_bn(gamma, beta, mean, var, eps)

    # n = (1 << (w_bit - 1 + in_bit + scale_shift)) / (((1 << (w_bit - 1)) - 1) * ((1 << in_bit) - 1))
    inc_q_float = (((1 << out_bit) - 1) * w)
    bias_q_float = ((1 << (w_bit - 1)) - 1) * ((1 << in_bit - 1) * ((1 << out_bit) - 1) * b)
    inc_q = inc_q_float.round().int()
    bias_q = bias_q_float.round().int()
    print('inc_q: {}'.format(inc_q))
    print('bias_q: {}'.format(bias_q))
    return inc_q, bias_q


def quantize_weight(x, bits, signed=True):
    assert signed
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x))
    x_q = x * ((1 << (bits - 1)) - 1)
    x_q = torch.round(x_q).int()
    return x_q
