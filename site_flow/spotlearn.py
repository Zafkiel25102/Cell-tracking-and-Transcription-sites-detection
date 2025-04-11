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
    
class SpotlearnNet(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.inc = DoubleConv(input_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = OutConv(64, output_channel)
        # self.threshold_filter = ThresholdFilterLayer(initial_threshold=0.6)

    def forward(self, x):
        x1 = self.inc(x)  # (batch, 64, 128, 128)
        x2 = self.down1(x1)  # (batch, 128, 64, 64)
        x3 = self.down2(x2)  # (batch, 256, 32, 32)
        x = self.up1(x3, x2)  # (batch, 128 + 128, 64, 64)
        x = self.up2(x, x1)
        x = self.outc(x)
        # x = self.threshold_filter(x)
        return x