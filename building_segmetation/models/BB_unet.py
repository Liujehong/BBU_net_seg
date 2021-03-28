import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class backbonevgg(nn.Module):
    def __init__(self,vgg16= None):
        super(backbonevgg, self).__init__()
        vgg = vgg16
        # vgg.features._modules['6'] = nn.Sequential(nn.Linear(4096, numClass), nn.Softmax(dim=1))
        self.layer1 = nn.Sequential(
            vgg.features._modules['0'],
            vgg.features._modules['1'],
            vgg.features._modules['2'],
            vgg.features._modules['3'],
            vgg.features._modules['4'],
            vgg.features._modules['5'],
        )
        self.layer2 = nn.Sequential(
            vgg.features._modules['6'],
            vgg.features._modules['7'],
            vgg.features._modules['8'],
            vgg.features._modules['9'],
            vgg.features._modules['10'],
            vgg.features._modules['11'],
            vgg.features._modules['12'],
        )
        self.layer3 = nn.Sequential(
            vgg.features._modules['13'],
            vgg.features._modules['14'],
            vgg.features._modules['15'],
            vgg.features._modules['16'],
            vgg.features._modules['17'],
            vgg.features._modules['18'],
            vgg.features._modules['19'],
        )
        self.layer4 = nn.Sequential(
            vgg.features._modules['23'],
            vgg.features._modules['24'],
            vgg.features._modules['25'],
            vgg.features._modules['26'],
            vgg.features._modules['27'],
            vgg.features._modules['28'],
            vgg.features._modules['29'],

        )
        self.encoder1 = self.layer1
        self.encoder2 = self.layer2
        self.encoder3 = self.layer3
        self.encoderc4 = self.layer4
        self.cb_block = DoubleConv(512,512)
        self.encoderr4 = self.layer4
        self.rb_block = DoubleConv(512,512)

    def forward(self,x,**kwargs):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)


        if "feature_cb" in kwargs:
            e4 = self.encoderc4(e3)
            out = self.cb_block(e4)
            return out,e3,e2,e1
        elif "feature_rb" in kwargs:
            e4 = self.encoderr4(e3)
            out = self.rb_block(e4)
            return out,e3,e2,e1


# unet backbone
class backbone(nn.Module):
    def __init__(self,resnet34 = None):
        super(backbone, self).__init__()
        resnet = resnet34
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoderc4 = resnet.layer4
        self.cb_block = DoubleConv(512,512)
        self.encoderr4 = resnet.layer4
        self.rb_block = DoubleConv(512,512)

    def forward(self,x,**kwargs):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)


        if "feature_cb" in kwargs:
            e4 = self.encoderc4(e3)
            out = self.cb_block(e4)
            return out,e3,e2,e1
        elif "feature_rb" in kwargs:
            e4 = self.encoderr4(e3)
            out = self.rb_block(e4)
            return out,e3,e2,e1



#上采样后的最后卷积
class finalbone(nn.Module):
    def __init__(self):
        super(finalbone, self).__init__()


        self.finalreluc1 = nonlinearity
        self.finalconvc2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalreluc2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # # self.finaldeconvr1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelur1 = nonlinearity
        self.finalconvr2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelur2 = nonlinearity
        self.deepconv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self,x,**kwargs):
        # x = self.finaldeconv1(x)
        if "feature_cb" in kwargs:
            x = self.finalreluc1(x)
            x = self.finalconvc2(x)
            x = self.finalreluc2(x)
            x_c = self.deepconv(x)
            x_c = torch.sigmoid(x_c)
            return x,x_c
        elif "feature_rb" in  kwargs:
            x = self.finalrelur1(x)
            x = self.finalconvr2(x)
            x = self.finalrelur2(x)
            x_r = self.deepconv(x)
            x_r = torch.sigmoid(x_r)
            return x,x_r
        #




class BB_unet(nn.Module):
    def __init__(self, num_classes=1,resnet34=None, **kwargs):
        super(BB_unet, self).__init__()

        filters = [64, 128, 256, 512]
        #backbone = enconder + centerlayer
        self.backbone = backbone(resnet34=resnet34)
        #deconder
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1) # 如果需要对分类器建立两个模型 则将32扩大两倍
        self.finalrelu2 = nonlinearity
        # self.finalbone = finalbone(num_classes=num_classes)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv3 = nn.Conv2d(32, num_classes,3, padding=1) #如果需要对分类器建立两个模型 则将32扩大两倍
    def forward(self,x,**kwargs):
        #如果为classifer_flag 证明进入最后的分类器环节
        if "classifer_flag" in kwargs:
            x = self.finalrelu1(x)
            x = self.finalconv2(x)
            x = self.finalrelu2(x)
            x = self.finalconv3(x)
            return x
        #如果不为classifer_flag 则进入网络头部的主干网络
        else:
            x,e3,e2,e1=self.backbone(x,**kwargs) #进入主干网络 按照feature_a、feature_b进行两个不同分支的训练
            x=self.decoder4(x) +e3
            x=self.decoder3(x) +e2
            x=self.decoder2(x) +e1
            x=self.decoder1(x)
            x=self.finaldeconv1(x)
            # x = self.finalbone(x, **kwargs)

        return x


class BB_unet_var2(nn.Module):
    def __init__(self, num_classes=1,resnet34=None, **kwargs):
        super(BB_unet_var2, self).__init__()

        filters = [64, 128, 256, 512]
        #backbone = enconder + centerlayer
        self.backbone = backbone(resnet34=resnet34)
        #deconder
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1) # 如果需要对分类器建立两个模型 则将32扩大两倍
        self.finalrelu2 = nonlinearity
        # self.finalbone = finalbone(num_classes=num_classes)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1) #如果需要对分类器建立两个模型 则将32扩大两倍
    def forward(self,x,**kwargs):
        #同BBU_net
        if "classifer_flag" in kwargs:
            x = self.finalrelu1(x)
            x = self.finalconv2(x)
            x = self.finalrelu2(x)
            x = self.finalconv3(x)
            return x
        else:
            x,e3,e2,e1=self.backbone(x,**kwargs)
            x=self.decoder4(x) +e3
            x=self.decoder3(x) +e2
            x=self.decoder2(x) +e1
            x=self.decoder1(x)
            x=self.finaldeconv1(x)
            # x = self.finalbone(x, **kwargs)

        return x

class BB_unet_deepversion(nn.Module):
    def __init__(self, num_classes=1,resnet34=None, **kwargs):
        super(BB_unet_deepversion, self).__init__()

        filters = [64, 128, 256, 512]
        #backbone = enconder + centerlayer
        self.backbone = backbone(resnet34=resnet34)
        self.finalbone = finalbone()
        #deconder
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1) # 如果需要对分类器建立两个模型 则将32扩大两倍
        # self.finalrelu2 = nonlinearity
        # self.finalbone = finalbone(num_classes=num_classes)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1) #如果需要对分类器建立两个模型 则将32扩大两倍
        self.deepconv = nn.Conv2d(32,1,3,padding=1) # 用于计算深监督loss
    def forward(self,x,**kwargs):
        #两个分支最后进行的分类器模块
        if "classifer_flag" in kwargs:
            x = self.finalconv3(x)
            return x
        #开头使用的主干网络 x_side：深监督的的输出 x 主体网络的输出
        else:
            x,e3,e2,e1=self.backbone(x,**kwargs)
            x=self.decoder4(x) +e3
            x=self.decoder3(x) +e2
            x=self.decoder2(x) +e1
            x=self.decoder1(x)
            x=self.finaldeconv1(x)
            x,x_side = self.finalbone(x, **kwargs)

        return x,x_side


class BB_unet_vgg(nn.Module):
    def __init__(self, num_classes=1,vgg=None, **kwargs):
        super(BB_unet_vgg, self).__init__()

        filters = [64, 128, 256, 512]
        #backbone = enconder + centerlayer
        self.backbone = backbonevgg(vgg16=vgg)
        #deconder
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(64, 64, 3, padding=1) # 如果需要对分类器建立两个模型 则将32扩大两倍
        self.finalrelu2 = nonlinearity
        # self.finalbone = finalbone(num_classes=num_classes)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv3 = nn.Conv2d(64, num_classes, 3, padding=1) #如果需要对分类器建立两个模型 则将32扩大两倍
    def forward(self,x,**kwargs):
        if "classifer_flag" in kwargs:
            x = self.finalrelu1(x)
            x = self.finalconv2(x)
            x = self.finalrelu2(x)
            x = self.finalconv3(x)
            return x
        else:
            x,e3,e2,e1=self.backbone(x,**kwargs)
            x=self.decoder4(x) +e3
            x=self.decoder3(x) +e2
            x=self.decoder2(x) +e1
            x=self.decoder1(x)
            x=self.finaldeconv1(x)
            # x = self.finalbone(x, **kwargs)

        return x
