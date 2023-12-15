import torch.nn as nn
import torch.nn.functional as F
import torch

class StateVariableNetwork(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.state_encoder=StateEncoder()
        self.state_decoder=StateDecoder()
        self.cell=ConvGRU3DCell(input_size=2, hidden_size=2, kernel_size=(3, 3, 3))
    def forward(self,xs,vs,h):
        ex, ev = self.state_encoder(xs , vs )
        ex=ex.view(1,*ex.shape)
        ev=ev.view(1,*ev.shape)
        xh=torch.stack((ex,ev),axis=1)
        xhp=self.cell(xh,h)
        exn=xhp[:,0,:,:]
        evn=xhp[:,1,:,:]
        xsn,xvn=self.state_decoder(exn,evn)
        return xsn,xvn,xhp
class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.se=SimpleEncoder()
        self.ve=SimpleEncoder()
    def forward(self,xs,xv):
        return self.se(xs),self.ve(xv)
class StateDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sd=SimpleDecoder()
        self.vd=SimpleDecoder(activation=nn.Identity())

    def forward(self,ex,ev):
        return self.sd(ex),self.vd(ev)


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # No flatten or fully connected layers

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        return x
class SimpleDecoder(nn.Module):
    def __init__(self,activation=nn.Sigmoid()):
        super().__init__()
        # Decoder
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_activation=activation
    def forward(self, x):
        x = F.leaky_relu(self.conv_trans1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv_trans2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv_trans3(x), negative_slope=0.01)
        x = self.final_activation(self.conv_trans4(x))
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.new_memory = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, x, prev_state):
        if prev_state is None:
            # Assuming x is of shape [batch, channels, height, width]
            batch_size, _, height, width = x.size()
            prev_state = torch.zeros(batch_size, self.hidden_size, height, width, device=x.device)

        combined = torch.cat([x, prev_state], dim=1)  # concatenate along the channel dimension

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_new = torch.cat([x, reset * prev_state], dim=1)
        new_memory = torch.tanh(self.new_memory(combined_new))

        new_state = update * prev_state + (1 - update) * new_memory
        return new_state


class ConvGRU3DCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRU3DCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        padding = kernel_size[0] // 2

        # The input_size and hidden_size parameters refer to the number of channels
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.new_memory = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, x, prev_state):
        # Check if there is no previous state
        if prev_state is None:
            # Initialize prev_state with zeros
            batch_size = x.size(0)
            spatial_dims = x.size()[2:]
            prev_state = torch.zeros(batch_size, self.hidden_size, *spatial_dims, device=x.device)

        # Concatenate along the channel dimension
        combined = torch.cat([x, prev_state], dim=1)

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_new = torch.cat([x, reset * prev_state], dim=1)
        new_memory = torch.tanh(self.new_memory(combined_new))

        new_state = update * prev_state + (1 - update) * new_memory
        return new_state


import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.activ=nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.activ(self.outc(x))
        return logits

