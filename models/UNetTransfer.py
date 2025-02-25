import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, 1, bias=False),
            nn.BatchNorm2d(F_l)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = x
        psi = F.relu(g1 + x1)
        return x * self.psi(psi)

class UNetBatchNorm1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Pre-trained ResNet34 encoder
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        # Freeze early layers
        for param in self.encoder[:6].parameters():
            param.requires_grad = False

        # Attention gates
        self.attn1 = AttentionGate(256, 128)
        self.attn2 = AttentionGate(128, 64)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Output layers
        self.seg_out = nn.Conv2d(64, n_class, 1)
        self.cls_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder[:5](x)  # layer0-layer2
        e2 = self.encoder[5](e1) # layer3
        e3 = self.encoder[6](e2) # layer4
        
        # Decoder with attention
        d1 = self.up1(e3)
        d1 = torch.cat([d1, self.attn1(e2, d1)], 1)
        d1 = self.conv1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.attn2(e1, d2)], 1)
        d2 = self.conv2(d2)
        
        d3 = self.up3(d2)
        d3 = self.conv3(d3)
        
        return self.seg_out(d3), self.cls_out(e3)
