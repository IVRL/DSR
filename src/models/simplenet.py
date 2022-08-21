import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

__all__ = ['simplenet', 'simplenet_altitude', 'simplenet_altitude_da']

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class SimpleNet(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleNet, self).__init__()
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=128, out_channels=in_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs

        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        #out = torch.add(out, inputs)

        out = self.output(self.relu(out))

        out = torch.add(out, residual)
        return out


class SimpleNetAltitude(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleNetAltitude, self).__init__()
        self.input = nn.Conv2d(in_channels=in_channels+1, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=128, out_channels=in_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, altitude):
        residual = x
        altitude = altitude.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        altitude = altitude.expand((-1, -1, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, altitude], dim=1)
        inputs = self.input(self.relu(x))
        out = inputs

        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        #out = torch.add(out, inputs)

        out = self.output(self.relu(out))

        out = torch.add(out, residual)
        return out



class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 128 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        # self.relu = nn.LeakyReLU(0.1, True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, altitude):

        b, c, h, w = x.size()

        # branch 1
        kernel = self.kernel(altitude).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x, altitude)

        return out
    
class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, altitude):
        att = self.conv_du(altitude[:, :, None, None])

        return x * att

    
class SimpleNetAltitude(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleNetAltitude, self).__init__()
        
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.altitude_encoder = nn.Sequential(nn.Linear(in_features=1, out_features=128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(in_features=128, out_features=128))
        
        self.daconv1 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.daconv2 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.daconv3 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.daconv4 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.daconv5 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.daconv6 = DA_conv(channels_in=128, channels_out=128, 
                               kernel_size=3, reduction=8)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=128, out_channels=in_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, altitude):
        
        altitude = self.altitude_encoder(torch.unsqueeze(altitude, 1))
        
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        
        out = self.daconv1(self.relu(out), altitude)
        out = self.conv1(self.relu(out))
        
        out = self.daconv2(self.relu(out), altitude)
        out = self.conv2(self.relu(out))
        
        out = self.daconv3(self.relu(out), altitude)
        out = self.conv3(self.relu(out))
        
        out = self.daconv4(self.relu(out), altitude)
        out = self.conv4(self.relu(out))
        
        out = self.daconv5(self.relu(out), altitude)
        out = self.conv5(self.relu(out))
        
        out = self.daconv6(self.relu(out), altitude)
        out = self.conv6(self.relu(out))

        out = self.output(self.relu(out))

        out = torch.add(out, residual)
        return out

def simplenet(scale, pretrained):

    return SimpleNet(in_channels=3)

def simplenet_altitude(scale, pretrained):

    return SimpleNetAltitude(in_channels=3)
