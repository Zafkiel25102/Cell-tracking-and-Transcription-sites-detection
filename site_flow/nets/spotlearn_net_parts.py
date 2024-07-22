import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),

            # nn.Dropout2d(p=0.2),
            # nn.BatchNorm2d(mid_channel),

            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            # nn.Dropout2d(p=0.2, inplace=True),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        assert x1.shape == x2.shape, 'get wrong shape: ' \
                                     f'x1 shape: {x1.shape} x2: {x2.shape}'
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out_act = self.sigmoid(out)
        return out_act

# class ThresholdFilterLayer(nn.Module):
#     def __init__(self, initial_threshold=0.5):
#         super(ThresholdFilterLayer, self).__init__()
#         self.threshold = nn.Parameter(torch.tensor(initial_threshold), requires_grad=True)

#     def forward(self, x):
#         # 使用阈值过滤特征图
#         filtered_output = torch.where(x > self.threshold, x, torch.zeros_like(x))
#         return filtered_output