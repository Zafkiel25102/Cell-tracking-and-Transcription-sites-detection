from .spotlearn_net_parts import *

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