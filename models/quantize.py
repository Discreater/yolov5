import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter, init


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

    def extra_repr(self):
        s = super(Conv2dQ, self).extra_repr()
        s += ', first_layer={}'.format(self.first_layer)
        return s

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


class ActQ(nn.Module):
    def __init__(self, a_bit=8):
        super(ActQ, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = ActivationQuantize(a_bits=a_bit, clamp=False)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = torch.clamp(x, 0, 6)
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q


def reshape_to_activation(x):
    return x.reshape(1, -1, 1, 1)


def reshape_to_weight(x):
    return x.reshape(-1, 1, 1, 1)


def reshape_to_bias(x):
    return x.reshape(-1)


class BatchNorm2dQ(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, w_bit=8):
        super(BatchNorm2dQ, self).__init__(num_features, eps, momentum, affine,
                                           track_running_stats)
        self.w_bit = w_bit
        self.quantize_fn = ActivationQuantize(a_bits=w_bit, clamp=False)

    def extra_repr(self):
        s = super(BatchNorm2dQ, self).extra_repr()
        s += ", w_bit={}".format(self.w_bit)
        return s

    def forward(self, x):
        # return input
        gamma = self.weight
        var = self.running_var
        mean = self.running_mean
        eps = self.eps
        bias = self.bias
        w = gamma / (torch.sqrt(var) + eps)
        b = bias - (mean / (torch.sqrt(var) + eps)) * gamma

        w = torch.clamp(w, -1, 1) / 2 + 0.5
        # w = w / 2 / torch.max(torch.abs(w)) + 0.5
        w_q = 2 * self.quantize_fn(w) - 1

        b = torch.clamp(b, -1, 1) / 2 + 0.5
        b_q = 2 * self.quantize_fn(b) - 1
        # b_q = self.quantize_fn(torch.clamp())
        # return w_q * input + b_q
        return F.batch_norm(x, running_mean=mean * 0, running_var=torch.sign(torch.abs(var) + 1), weight=w_q,
                            bias=b_q, eps=eps * 0)


class Conv2dQBN(Conv2dQ):
    """Unused"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode: str = "zeros",
                 a_bits=8,
                 w_bits=8,
                 eps=1e-5,
                 momentum=0.01,
                 first_layer=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         a_bits, w_bits, first_layer)
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def forward(self, x):
        if self.training:
            # for BN
            output = F.conv2d(
                input=x,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            # update BN param
            dims = [0, 2, 3]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)

            # fuse
            if self.bias is not None:
                bias = reshape_to_bias(
                    self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - batch_mean * (self.gama / torch.sqrt(batch_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        else:
            if self.bias is not None:
                bias = reshape_to_bias(
                    self.beta + (self.bias - self.running_mean) * (
                            self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - self.running_mean * (
                        self.gamma / torch.sqrt(self.running_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))

        q_x = self.activation_quantizer(x) if not self.first_layer else x
        q_weight = self.weight_quantizer(weight)
        if self.training:
            output = F.conv2d(
                input=q_x,
                weight=q_weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
            output += reshape_to_activation(bias)
        else:
            output = F.conv2d(
                input=q_x,
                weight=q_weight,
                bias=bias,
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

    def extra_repr(self) -> str:
        s = 'w_bits={}'.format(self.w_bits)
        return s

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
    def __init__(self, a_bits, clamp=False):
        super().__init__()
        self.a_bits = a_bits
        self.clamp = clamp

    def extra_repr(self) -> str:
        s = 'a_bits={}'.format(self.a_bits)
        return s

    def round(self, x):
        output = Round.apply(x)
        return output

    def forward(self, x):
        assert self.a_bits != 1, '！Binary quantization is not supported ！'
        if self.a_bits == 32:
            output = x
        elif self.a_bits == 1:
            output = torch.sign(x)
        else:
            if self.clamp:
                output = torch.clamp(x * 0.1, 0, 1)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            else:
                output = x
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
