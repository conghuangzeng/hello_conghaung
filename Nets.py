import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
'''编写网络思路
卷积块
残差块
下采样块
上采样块
路由块
侦测网络卷积块converlutional  Set块

'''
transform = transforms.Compose([
    transforms.Resize((24,24)),
        transforms.ToTensor(),
     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])


class  Convolutional_layer(nn.Module):#卷积块
    def __init__(self,in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding),
            nn.BatchNorm2d(out_channels),#卷积后跟上BN层和激活
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.layers1(x)

class Res_layer(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.layers1 = nn.Sequential(
         Convolutional_layer(in_channels,in_channels//2,1,1,0),
        Convolutional_layer(in_channels//2,in_channels,3,1,1),

        )

    def forward(self, x):
        return self.layers1(x)+x


class  Downsample_layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layers1 = nn.Sequential(
            Convolutional_layer(in_channels,out_channels,3,2,padding=1)
        )
    def forward(self,x):
        return self.layers1(x)

class Upsample_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.interpolate(x,scale_factor=2,mode="nearest")

class ConverlutonalSet(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.layers1 = nn.Sequential(
            Convolutional_layer(in_channels,out_channels,1,1,0),
            Convolutional_layer(out_channels,in_channels,3,1,1),
            Convolutional_layer(in_channels,out_channels,1,1,0),
            Convolutional_layer(out_channels,in_channels,3,1,1),
            Convolutional_layer(in_channels,out_channels, 1, 1, 0),
        )
    def forward(self, x):
        return self.layers1(x)

# con =ConverlutonalSet(64,128)
# x = torch.randn(1,64,208,208)
# # x = transform(x)
# print(x.size())
# y = con(x)
#
# print(y.size())
class Main_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk_52 = nn.Sequential(
            Convolutional_layer(3,32,3,1,1),
            Downsample_layer(32,64),
            Res_layer(64),
            Downsample_layer(64,128),
            Res_layer(128),
            Res_layer(128),
            Downsample_layer(128, 256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),
            Res_layer(256),)
        self.trunk_26 = nn.Sequential(
            Downsample_layer(256,512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),
            Res_layer(512),)
        self.trunk_13 = nn.Sequential(
            Downsample_layer(512,1024),
            Res_layer(1024),
            Res_layer(1024),
            Res_layer(1024),
            Res_layer(1024)   )
        self.con_set_13 = nn.Sequential(
            ConverlutonalSet(1024,512))
        self.predict_13 = nn.Sequential(
            Convolutional_layer(512,1024,3,1,1),
            Convolutional_layer(1024,45,1,1,0))
        self.up_26 = torch.nn.Sequential(
            Convolutional_layer(512, 256, 1, 1, 0),
            Upsample_layer())
        self.con_set_26 = nn.Sequential(
            ConverlutonalSet(768, 256))
        self.predict_26 = nn.Sequential(
            Convolutional_layer(256, 512, 3, 1, 1),
            Convolutional_layer(512, 45, 1, 1, 0))
        self.up_52 = torch.nn.Sequential(
            Convolutional_layer(256, 128, 1, 1, 0),
            Upsample_layer())
        self.con_set_52 = nn.Sequential(
            ConverlutonalSet(384, 128))
        self.predict_52 = nn.Sequential(
            Convolutional_layer(128, 256, 3, 1, 1),
            Convolutional_layer(256, 45, 1, 1, 0))
    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.con_set_13(h_13)
        detetion_out_13 = self.predict_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.con_set_26(route_out_26)
        detetion_out_26 = self.predict_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.con_set_52(route_out_52)
        detetion_out_52 = self.predict_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52

if __name__ == '__main__':
    trunk = Main_net()

    x = torch.Tensor(2, 3, 416, 416)

    y_13, y_26, y_52 = trunk(x)
    print(y_13.size())
    print(y_26.size())
    print(y_52.size())