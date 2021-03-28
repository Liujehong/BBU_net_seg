#------------------------------------#
#
#双分支连接
#
#------------------------------------#

import numpy as np
import torch, math
import ttach as tta
import matplotlib.pyplot as plt

class Combinerv2:
    def __init__(self, epoch_number, loss,device,args):
        self.args =args
        self.epoch_number = epoch_number
        self.device = device
        self.loss = loss
        self.type = 'bnn_mix'
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self,epoch):
        self.epoch = epoch

    def forward(self, model, criterion, image1, label1,image2,label2,**kwargs):
        image_a, image_b = image1, image2
        label_a, label_b = label1, label2
        feature_a,a_side = model(image_a, feature_cb=True)# -> 32 and a_side
        feature_b,b_side = model(image_b, feature_rb=True)# -> 32 and b_side
        loss_a = self.loss(a_side,label_a)
        loss_b = self.loss(b_side,label_b)

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        # l = 0.5  # fix
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        # l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        # l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        # l = np.random.beta(self.alpha, self.alpha) # beta distribution
        # l = 1 if self.epoch <= 120 else 0  # seperated stage
        # if self.args.arch=='BB_unet':
        #     mixed_feature = 2 * (l * feature_a+((1-l) * feature_b))
        # else:
        mixed_feature = 2 * torch.cat((l * feature_a,(1-l)*feature_b),dim=1)
        # output =  torch.sigmoid(mixed_feature)
        output =model(mixed_feature,classifer_flag=True)
        output = torch.sigmoid(output)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
        # loss_all = 0.7*loss +0.3*((l*loss_a) + ((1-l) * loss_b))
        loss_all = 0.5 * loss + 0.5 * ((l * loss_a) + ((1 - l) * loss_b))
        return loss_all
    def bbn_unet(self, model, image1,  **kwargs):
        image_a = image1.to(self.device)

        # label_a = label1.to(self.device)
        image_b = image_a
        feature_a, _ = model(image_a, feature_cb=True)  # -> 32 and a_side
        feature_b, _ = model(image_b, feature_rb=True)  # -> 32 and b_side
        # mixed_feature1 = torch.cat((feature_a,feature_a),dim=1)
        # mixed_feature2 = torch.cat((feature_b, feature_b), dim=1)
        # output1 = model(mixed_feature1, classifer_flag=True)
        # output2 = model(mixed_feature2, classifer_flag=True)
        # predict1 = torch.squeeze(output1).cpu().numpy()
        # predict2 = torch.squeeze(output2).cpu().numpy()
        # plt.imshow(predict1 * 10)
        # plt.show()
        # plt.imshow(predict2  *0.001)
        # plt.show()
        # img = torch.squeeze(image_a).cpu().numpy()
        # plt.imshow(img.reshape(512,512,3))
        # plt.show()
        l = 0.5


        mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)
        # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        output = model(mixed_feature,classifer_flag=True)
        output = torch.sigmoid(output)
        return output