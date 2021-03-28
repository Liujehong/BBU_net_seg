#------------------------------------#
#
#双分支连接
#
#------------------------------------#

import numpy as np
import torch, math
import ttach as tta
import matplotlib.pyplot as plt
class Combiner:
    def __init__(self, epoch_number, device,args):
        self.args =args
        self.epoch_number = epoch_number
        self.device = device
        self.type = 'bnn_mix'
        self.initilize_all_parameters()
#初始化参数 针对epoch特别大的训练
    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number
#获得当前训练的epoch 用于后面跳转l参数
    def reset_epoch(self,epoch):
        self.epoch = epoch
#前向传播 用于训练
    def forward(self, model, criterion, image1, label1,image2,label2,**kwargs):
        image_a, image_b = image1, image2 #得到两个分支的输入图片
        label_a, label_b = label1, label2 #得到两个分支的输入标签
        feature_a, feature_b = (
            model(image_a, feature_cb=True), #a分支的特征图
            model(image_b, feature_rb=True), #b分支的特征图
        )

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # l系数的动态调整
        # l = 0.5  # fix
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        # l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        # l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        # l = np.random.beta(self.alpha, self.alpha) # beta distribution
        # l = 1 if self.epoch <= 120 else 0  # seperated stage
        if self.args.arch=='BB_unet':
            mixed_feature = 2 * (l * feature_a+((1-l) * feature_b)) #用于单分类器的输入（两个分支简单相加进入最后的分类器）
        else:
            mixed_feature = 2 * torch.cat((l * feature_a,(1-l)*feature_b),dim=1) #用于双分类器的输入（两个分支通道叠加最后的分类器）

        output =model(mixed_feature,classifer_flag=True) #分类器输入混合的mixed_feature
        output = torch.sigmoid(output)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b) #计算loss
        return loss #返回loss
    #用于预测的函数
    def bbn_unet(self, model, image1,  **kwargs):
        image_a = image1.to(self.device)  #得到等待预测的原图

        # label_a = label1.to(self.device)
        image_b = image_a #两个分支共同预测用一张图片
        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        l = 0.5 #两个分支公平预测
        if self.args.arch == 'BB_unet':
            mixed_feature =2* (l * feature_a )+ ((1 - l)* feature_b)
        else:
            mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)
        # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        output = model(mixed_feature,classifer_flag=True)
        output = torch.sigmoid(output)
        return output #返回特征图