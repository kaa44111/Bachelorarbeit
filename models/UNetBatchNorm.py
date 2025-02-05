import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torchsummary import summary
import torch
from torch.autograd import Variable
from torchviz import make_dot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Conv2d(in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        # in_chanels:
        # out_channels:
        # kernel_size:
        # padding:
        # dilation:
        # groups:
        # bias:
        # padding_mode:
        # device:
        # dtype:
def plot_graph(model):
    # Dummy-Input für das Modell
    x = Variable(torch.randn(1, 3, 128, 128)).to(device)
    y = model(x)
    params_dict = dict(list(model.named_parameters()))
    # Erstelle und speichere das Diagramm des Computation Graphs
    make_dot(y.mean(), params=params_dict).render("unet_batchnorm", format="png")


class UNetBatchNorm(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.b11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.b12 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.b21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.b22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.b31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.b32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.b41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.b42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.b51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        self.b52 = nn.BatchNorm2d(1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bd11 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bd12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bd21 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bd22 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bd31 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bd32 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bd41 = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bd42 = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.b11(self.e11(x)))
        xe12 = relu(self.b12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.b21(self.e21(xp1)))
        xe22 = relu(self.b22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.b31(self.e31(xp2)))
        xe32 = relu(self.b32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.b41(self.e41(xp3)))
        xe42 = relu(self.b42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.b51(self.e51(xp4)))
        xe52 = relu(self.b52(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.bd11(self.d11(xu11)))
        xd12 = relu(self.bd12(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.bd21(self.d21(xu22)))
        xd22 = relu(self.bd22(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.bd31(self.d31(xu33)))
        xd32 = relu(self.bd32(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.bd41(self.d41(xu44)))
        xd42 = relu(self.bd42(self.d42(xd41)))

        # Output layer
        out = self.outconv(xd42)

        return out


# # Initialisiere das Modell
# model = UNetBatchNorm(1).to(device)
# # Plot des Computation Graphs
# plot_graph(model)
# # Zusammenfassung des Modells
# summary(model, (3, 192, 192))