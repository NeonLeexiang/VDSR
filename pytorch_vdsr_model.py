"""
    date:       2021/3/31 2:59 下午
    written by: neonleexiang
"""
import torch
from torch import nn
from math import sqrt


class ConvReLUBlock(nn.Module):
    def __init__(self):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                              stride=(1, 1), padding=(1, 1), bias=False)
        """
            inplace=True
            对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
        """
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class pytorch_VDSR(nn.Module):
    def __init__(self, num_channels=1):
        super(pytorch_VDSR, self).__init__()
        self.residual_layer = self.make_layer(ConvReLUBlock, 18)
        self.input_layer = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3,),
                                     stride=(1, 1), padding=(1, 1), bias=False)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        out = self.relu(self.input_layer(x))
        out = self.residual_layer(out)
        out = self.output_layer(out)
        out = torch.add(out, residual)
        return out

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)



