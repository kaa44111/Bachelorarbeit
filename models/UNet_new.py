""" Code imported from GitHub repository: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

""" Parts of the U-Net model """
import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_bn=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, with_bn=with_bn)
        self.down1 = Down(64, 128, with_bn=with_bn)
        self.down2 = Down(128, 256, with_bn=with_bn)
        self.down3 = Down(256, 512, with_bn=with_bn)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, with_bn=with_bn)
        self.up1 = Up(1024, 512 // factor, bilinear, with_bn=with_bn)
        self.up2 = Up(512, 256 // factor, bilinear, with_bn=with_bn)
        self.up3 = Up(256, 128 // factor, bilinear, with_bn=with_bn)
        self.up4 = Up(128, 64, bilinear, with_bn=with_bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.up4(x, x1)
        logits = self.outc(logits)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
