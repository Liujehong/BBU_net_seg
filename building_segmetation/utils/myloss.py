import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.grad_mode import F
from torch.nn.functional import binary_cross_entropy,binary_cross_entropy_with_logits


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=False, reduce=True,):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        # self.num = num

    def forward(self, inputs, targets):
        # targets.expand(input.shape[0], self.num, targets.shape[2], targets.shape[3])
        # targets[:, 1, :, :] = 1 - targets[:, 1, :, :]
        # if inputs.shape[1]==1: #转换为2层以计算
        #     inputs.expand(input.shape[0],self.num,inputs.shape[2],inputs.shape[3])
        #     inputs[:,1,:,:] = 1-  inputs[:,1,:,:]
        if self.logits:
            BCE_loss = binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss




# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         N = target.size(0)
#         smooth = 1
#
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
#
#         intersection = input_flat * target_flat
#
#         loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         loss = 1 - loss.sum() / N
#
#         return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = BinaryDiceLoss()
        self.cross_entropy = torch.nn.BCELoss()

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return 0.5*CE_loss + 0.5*dice_loss
# class MulticlassDiceLoss(nn.Module):
#     """
#     requires one hot encoded target. Applies DiceLoss on each class iteratively.
#     requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
#       batch size and C is number of classes
#     """
#
#     def __init__(self):
#         super(MulticlassDiceLoss, self).__init__()
#
#     def forward(self, input, target, weights=None):
#
#         C = target.shape[1]
#
#         # if weights is None:
#         # 	weights = torch.ones(C) #uniform weights for all classes
#
#         dice = DiceLoss()
#         totalLoss = 0
#
#         for i in range(C):
#             diceLoss = dice(input[:, i], target[:, i])
#             if weights is not None:
#                 diceLoss *= weights[i]
#             totalLoss += diceLoss
#
#         return totalLoss

# class BinaryDiceLoss(nn.Model):
#     def __init__(self):
#         super(BinaryDiceLoss, self).__init__()
#
#     def forward(self, input, targets):
#         # 获取每个批次的大小 N
#         N = targets.size()[0]
#         # 平滑变量
#         smooth = 1
#         # 将宽高 reshape 到同一纬度
#         input_flat = input.view(N, -1)
#         targets_flat = targets.view(N, -1)
#
#         # 计算交集
#         intersection = input_flat * targets_flat
#         N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
#         # 计算一个批次中平均每张图的损失
#         loss = 1 - N_dice_eff.sum() / N
#         return loss
