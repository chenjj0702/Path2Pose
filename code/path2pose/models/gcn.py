import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # spatial kernel size
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1),
                              dilation=(t_dilation, 1), bias=bias)  # weights (in_channel, out_channel, k)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class ConvTemporal(nn.Module):
    """
    网络结构：bn + relu + t-cn
    """
    def __init__(self, d_input, d_output, kernel_size=3, stride=1, padding=1, dropout=0.1, bn=True):
        super(ConvTemporal, self).__init__()
        tcn_list = []
        if bn:
            tcn_list += [nn.BatchNorm2d(d_input)]

        tcn_list += [
            nn.LeakyReLU(),
            nn.Conv2d(
                d_input,
                d_output,
                (kernel_size, 1),
                (stride, 1),
                padding,
            ),
        ]

        if bn:
            tcn_list += [nn.BatchNorm2d(d_output)]

        tcn_list += [
            nn.LeakyReLU(),
            nn.Conv2d(
                d_output,
                d_output,
                (kernel_size, 1),
                (stride, 1),
                padding,
            ),
        ]

        self.tcn = nn.ModuleList(tcn_list)

    def forward(self, x):
        out = x
        for m in self.tcn:
            out = m(out)
        return out


class Residual(nn.Module):
    def __init__(self, d_input, d_output):
        super(Residual, self).__init__()
        res_list = []
        res_list += [nn.Conv2d(d_input, d_output, kernel_size=1, stride=1)]

        self.res = nn.ModuleList(res_list)

    def forward(self, x):
        out = x
        for m in self.res:
            out = m(out)
        return out


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # (temporal k,  spatial k)
                 stride=1,  # temporal stride
                 dropout=0,
                 residual=True,
                 bn=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvGraphical(in_channels, out_channels,
                                 kernel_size[1])

        self.tcn = ConvTemporal(out_channels, out_channels, kernel_size[0], stride, padding, dropout, bn)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = Residual(in_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.bn = bn
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        if self.bn:
            x = self.norm(x)
        x = self.lrelu(x)

        return x
        # return self.relu(x), A
