'''
author:liuyang
from Chengdu Normal University
https://github.com.cnpmjs.org/Liujehong/BBU_net_seg
'''
import argparse
import subprocess
import logging
import ttach as tta
from pytorch_toolbelt import losses as L
import torch
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchsummary import summary
from torchvision import models
from tqdm import tqdm
from torch.nn.functional import pad
from models import BB_unet, conbainer
from models.conbainer import Combiner
from models.conbainerv2 import Combinerv2
from models.unet import Unet,resnet34_unet

from utils.dataset import *
from utils.metrics import *
from torchvision.transforms import transforms
from utils.plot import loss_plot
from utils.doubleimg import blending_result
from utils.plot import metrics_plot,lr_plot
# from torch.nn import Module
from utils.myloss import *
from torchvision.models import vgg16


#命令行
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action",'-ac',type=str, help="train/test/predict/val", default="train") #选择测试训练模式
    parse.add_argument("--epoch",'-e', type=int, default=20)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='BB_unet_var2',
                       help='UNet/resnet34_unet/resnet34_unet_nolock/P_unet/M_unet/N_unet/BB_unet/BB_unet_var2/BB_unet_deepversion')
    parse.add_argument("--batch_size", '-b',type=int, default=5)
    parse.add_argument('--dataset', '-d',default='reserve', help='final/reserve/testdata' )# 遥感数据
    #BBU-net使用reserve进行训练与预测
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument("--predictmode",'-p',type=str,help="samesample/expansion/",default="expansion") #samesample :采样预测
                                                                                                  #expansion ：膨胀预测
    parse.add_argument("--scheduler", '-s',type=str, help="cos/rel/", default="cos")
    parse.add_argument("--bbn", '-bb',type=str, help="yes/no/", default="yes")
    parse.add_argument("--TTA", '-tt',type=str, help="yes/no/", default="no")
    parse.add_argument("--loss",'-l',type=str,help="dice/ce/ce_dice",default="ce")
    args = parse.parse_args()
    return args

#得到打印log的地址
def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging





#得到模型
def getModel(args):
    if args.arch == 'UNet': #标准U-net
        model = Unet(3, 1).to(device)
    if args.arch == 'resnet34_unet': #预训练 res-net34
        resnet = models.resnet34(pretrained=False)
        # summary(resnet, (3, 512, 512))
        #
        # for i in resnet.named_modules():
        #     print(i)
        #     print('==============================')
        model = resnet34_unet(1,resnet34=resnet).to(device)
    if args.arch == 'resnet34_unet_nolock': # 冻结权重的res-net34
        resnet = models.resnet34(pretrained=True)
        model = resnet34_unet(1, resnet34=resnet).to(device)
        for param in resnet.parameters():
            param.requires_grad = False #冻结骨干网络，这部分网络有与训练权重
        # model.load_state_dict(torch.load(
        # r'G:\ML学习日志\building_segmetation\save_model\resnet34_unet_10_GID_51.pth', map_location="cuda"))  # 载入训练好的模型


    if args.arch == 'BB_unet':#单一分类器
        resnet = models.resnet34(pretrained=True)
        model = BB_unet.BB_unet(num_classes=1,resnet34=resnet).to(device)
    if args.arch == 'BB_unet_var2':#双分类器
        resnet = models.resnet34(pretrained=True)
        model = BB_unet.BB_unet_var2(num_classes=1,resnet34=resnet).to(device)
    if args.arch == 'BB_unet_deepversion':#利用深监督进行diceloss+ce+loss的加权 （本文的最优方法）
        resnet = models.resnet34(pretrained=True)
        model = BB_unet.BB_unet_deepversion(num_classes=1, resnet34=resnet).to(device)
    return model


#得到训练集 验证集 测试集
def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset =='final':
        train_dataset = flinalDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = flinalDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = flinalDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if args.dataset == 'testdata':
        test_dataset = testDateset(r'test','val',transform=x_transforms,target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset,batch_size=1,shuffle=True)

    if args.dataset == 'reserve':
        train_dataset = reverseDataset(r'train',str(args.bbn),transform=x_transforms,target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        val_dataset = reverseDataset(r'val',str(args.bbn),transform = x_transforms ,target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset,batch_size=5,shuffle=True)
        test_dataset = reverseDataset(r'test',str(args.bbn),transform=x_transforms,target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset,batch_size=1,shuffle=True)
    return train_dataloaders,val_dataloaders,test_dataloaders



#验证函数
def val(conbainer,model,best_iou,val_dataloaders,args):
    model= model.eval()
    maxtrx = IOUMetric(2)
    with torch.no_grad():
        i=0   #验证集中第i张图
        acc_total = 0
        miou_total = 0
        # hd_total = 0
        # dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        # for x, _,pic,mask in val_dataloaders:
        if args.bbn == 'yes':
            for sample in val_dataloaders:
                img_x_c, _,\
                img_y_c, _, \
                pic_pathc, mask_pathc, \
                _, _ = sample
                img_x_c = img_x_c.to(device)
                y = conbainer.bbn_unet(model,img_x_c)
                img_y = torch.squeeze(y).cpu().numpy()
                mask = torch.squeeze(img_y_c).cpu().numpy()
                maxtrx.add_batch(img_y, mask)
                acc, mean_iu = maxtrx.evaluate()
                acc_total += acc
                miou_total += mean_iu  # 获取当前预测图的miou，并加到总miou中
                # dice_total += get_dice(mask[0],img_y)
                print('%d,Miou=%f,acc=%f' % (i,  mean_iu, acc))
                if i < num: i += 1  # 处理验证集下一张图


        else:
            for x, mask, pic, _ in val_dataloaders:
                x = x.to(device)
                y = model(x)
                if args.deepsupervision:
                    img_y = torch.squeeze(y[-1]).cpu().numpy()
                    mask = torch.squeeze(mask).cpu().numpy()
                else:
                    img_y = torch.squeeze(y).cpu().numpy()
                    mask = torch.squeeze(mask).cpu().numpy()
                    # print(mask.dtype)
                    # print(img_y.dtype)
                    #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

                maxtrx.add_batch(img_y, mask)
                acc, mean_iu = maxtrx.evaluate()
                # hd_total += get_hd(mask[0], img_y)
                acc_total +=acc
                miou_total +=  mean_iu#获取当前预测图的miou，并加到总miou中
                # dice_total += get_dice(mask[0],img_y)
                print('%d,Miou=%f,acc=%f' % (i, mean_iu, acc))
                if i < num:i+=1   #处理验证集下一张图

        aver_miou = miou_total / num
        aver_acc =acc_total / num
        # aver_hd = hd_total / num
        # aver_dice = dice_total/num
        print('Miou=%f,aver_acc=%f' % (aver_miou,aver_acc))
        logging.info('Miou=%f,aver_acc=%f' % (aver_miou,aver_acc))
        if aver_miou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_miou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_miou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_miou
            print('===========>save best model!')
            # torch.save(model.state_dict(), r'G:/ML学习日志/building_segmetation/save_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
            torch.save(model.state_dict(), r'./save_model/' + str(args.arch) + '_' + str(
                args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch)+'_'+ str(args.loss) + '.pth')
        return best_iou,aver_miou,aver_acc

#训练函数
def train(model, criterion, optimizer, train_dataloader,val_dataloader, args,conbainer):
    #参数初始化
    best_iou,aver_iou,aver_acc = 0,0,0
    conbainer = conbainer
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    acc_list = []
    lr_list = []
    #bbn网络训练
    if args.bbn == 'yes':
        for epoch in range(num_epochs):
            model = model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(train_dataloader.dataset)
            epoch_loss = 0
            step = 0
            conbainer.reset_epoch(epoch=epoch) #获得当前epoch
            for i, sample in enumerate(train_dataloader):
                img_c,img_r,\
                label_c,label_r,\
                pic_pathc,mask_pathc,\
                pic_pathr, mask_pathr = sample
                step += 1
                input_c = img_c.to(device)
                label_c = label_c.to(device)
                input_r = img_r.to(device)
                label_r = label_r.to(device)
                # 梯度清0
                optimizer.zero_grad()

                loss = conbainer.forward(model,criterion,input_c,label_c,input_r,label_r) #计算loss
                if threshold != None:
                    if loss > threshold:
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if args.scheduler == 'cos':
                    scheduler.step()
                print("%d/%d,train_loss:%f,lr=%e" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(),optimizer.param_groups[-1]['lr']))
                logging.info(
                    "%d/%d,train_loss:%f,lr=%e" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(),optimizer.param_groups[-1]['lr']))
                lr_list.append(optimizer.param_groups[-1]['lr'])
            loss_list.append(epoch_loss / len(train_dataloader))

            # best_iou,aver_iou,aver_dice,aver_hd = val(model,best_iou,val_dataloader)
            best_iou, aver_iou, aver_acc = val(conbainer,model, best_iou, val_dataloader,args)
            # lr_scheduler.step(epoch_loss / len(train_dataloader))
            if args.scheduler == 'rel':
                scheduler.step(aver_iou)
            iou_list.append(aver_iou)
            acc_list.append(aver_acc)
            # hd_list.append(aver_hd)
            print("epoch %d loss:%f" % (epoch, epoch_loss / len(train_dataloader)))
            logging.info("epoch %d loss:%f" % (epoch, epoch_loss / len(train_dataloader)))
        loss_plot(args, loss_list)
        lr_plot(args, lr_list, len(train_dataloader))
        metrics_plot(args, 'iou&acc', iou_list, acc_list)
        # metrics_plot(args,'hd',hd_list)
        return model
    #普通网络训练
    else:
        for epoch in range(num_epochs):
            model = model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(train_dataloader.dataset)
            epoch_loss = 0
            step = 0

            # for x, y,_,mask in  train_dataloader:
            for i,sample in enumerate(train_dataloader) :
                x,y,_,mask = sample
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                if args.deepsupervision:
                    outputs = model(inputs)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, labels)
                    loss /= len(outputs)
                else:
                    output = model(inputs)
                    loss = criterion(output, labels)
                if threshold!=None:
                    if loss > threshold:
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                # scheduler.step(epoch+ i  / len(train_dataloader) )
                if args.scheduler == 'cos':
                    scheduler.step()
                print("%d/%d,train_loss:%f,lr=%e" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(),optimizer.param_groups[-1]['lr']))
                logging.info("%d/%d,train_loss:%f,lr=%e" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(),optimizer.param_groups[-1]['lr']))
                lr_list.append(optimizer.param_groups[-1]['lr'])
            loss_list.append(epoch_loss / len(train_dataloader))

            # best_iou,aver_iou,aver_dice,aver_hd = val(model,best_iou,val_dataloader)
            best_iou, aver_iou, aver_acc = val(conbainer,model, best_iou, val_dataloader,args)
            if args.scheduler == 'rel':
                scheduler.step(aver_iou)
            iou_list.append(aver_iou)
            acc_list.append(aver_acc)
            # hd_list.append(aver_hd)
            print("epoch %d loss:%f" % (epoch, epoch_loss / len(train_dataloader)))
            logging.info("epoch %d loss:%f" % (epoch, epoch_loss / len(train_dataloader)))
        loss_plot(args, loss_list)
        lr_plot(args, lr_list, len(train_dataloader))
        metrics_plot(args, 'iou&acc',iou_list, acc_list)
        # metrics_plot(args,'hd',hd_list)
        return model

#预测开放测试
def test_predic(test_dataloaders,save_predict,combainer,args):
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean',combainer=combainer)
    logging.info('final test....')
    img_size = 512

    if save_predict == True:
        dir = os.path.join(r'./save_predict', str(args.arch+'_'+args.loss), str(args.batch_size), str(args.epoch), str(args.dataset),
                           str(args.predictmode))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')

    model.load_state_dict(torch.load(
        r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(
            args.epoch)+'_'+ str(args.loss)  + '.pth', map_location="cuda"))  # 载入训练好的模型

    model.eval()
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        mpa_total = 0
        num = len(test_dataloaders)
        if args.bbn == 'yes':
            for img_x_c,_,img_y_c,_,pic_pathc,mask_pathc,_,_ in test_dataloaders:
                #膨胀预测
                if args.predictmode =="expansion":
                    stride = 256  # 滑窗采样步长
                    pad_size = 128
                    pic = img_x_c.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    #膨胀操作 边界padding
                    pic = pad(pic, pad=(pad_size,2 * pad_size, pad_size,2* pad_size),
                                mode='reflect')  # ->expandsion原图->(1,1,5384,5384)
                    width_pad = width + 2 * pad_size
                    height_pad = height + 2 * pad_size
                    predict_holder = np.zeros((width_pad, height_pad))
                    for w in range(width_pad // stride ):
                        for h in range(height_pad // stride ):
                            # print('%d,%d' %(w,h))
                            img_roi = pic[:,:,h * stride:h * stride + img_size,w * stride:w * stride + img_size] #->512,512
                            #使用TTA预测
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            else:
                            #使用单模型预测
                                predict  = combainer.bbn_unet(model,img_roi)
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict = torch.squeeze(predict).cpu().numpy()#->转换为没用batch的单张numpy形式矩阵
                            predict_holder[h * stride:h * stride + (img_size // 2),w * stride:w * stride +(img_size // 2)] = predict[128:384,128:384 ]#->将预测值中心区域即（128，384）装入对应的容器矩阵对应位置

                #普通滑窗预测
                if  args.predictmode == 'samesample':
                    stride = 512
                    pic = img_x_c.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(0, stride - width % stride, 0, stride - width % stride),
                              value=0)  # ->expandsion原图->(1,1,5122,5122)
                    # padding 后的长宽
                    width_or = width + (stride - width % stride)
                    height_or = height + (stride - height % stride)
                    predict_holder = np.zeros((width_or, height_or))
                    for w in range(width_or // stride):
                        for h in range(height_or // stride):
                            img_roi = pic[:, :, h * stride:h * stride + img_size,
                                      w * stride:w * stride + img_size]  # ->512,512
                            #使用TTA预测
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            #使用单模型预测
                            else:
                                predict  = combainer.bbn_unet(model,img_roi)
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict = torch.squeeze(predict).cpu().numpy()
                            predict_holder[h * stride:h * stride + stride, w * stride:w * stride + stride] = predict

                predict_holder = predict_holder[0:width, 0:height]
                maxtrx = IOUMetric(2)
                mask = torch.squeeze(img_y_c).cpu().numpy()  # 转换
                mask = mask.reshape((1, 1, 5000, 5000))  # 转换为可以计算的mask
                predict_holder[predict_holder >= 0.5] = 1
                predict_holder[predict_holder != 1] = 0
                predict_holder = predict_holder.reshape((1, 1, 5000, 5000))
                maxtrx.add_batch(predict_holder, mask)
                acc, mean_iu = maxtrx.evaluate()
                miou_total += mean_iu
                mpa_total += acc
                print('img %d :,Miou=%f,acc=%f' % (i, mean_iu, acc))
                if i < num:
                    i += 1
                # mpa_total += acc
                # miou_total += mean_iu  # 获取当前预测图的miou，并加到总miou中
                # 图像
                mask = mask.reshape(5000, 5000)
                predict_holder = predict_holder.reshape(5000, 5000)
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('input')
                img = Image.open(pic_pathc[0])
                plt.imshow(img)
                # print(pic_path[0])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('predict_iou: %f' % mean_iu)
                plt.imshow(predict_holder, 'Greys_r')
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('mask')
                plt.imshow(Image.open(mask_pathc[0]), cmap='Greys_r')
                saved_predict = dir + '/' + mask_pathc[0].split('\\')[-1]
                saved_predict1 = '.' + saved_predict.split('.')[1] + '.tif'
                saved_predict2 = '.' + saved_predict.split('.')[1] + 'mask' + '.tif'
                saved_predict3 = '.' + saved_predict.split('.')[1] + 'predict' + '.tif'
                Image.fromarray(np.uint8(mask* 255)).save(saved_predict2) #saved_predict,saved_predict,np.uint8(predict_holder*255)
                # Image.fromarray(np.uint8(predict_holder*255)).save(saved_predict3)
                blending_result(img,predict_holder).save(saved_predict3)

                plt.savefig(saved_predict1)




        #普通网络
        else:
            for pic,mask,pic_path,mask_path in test_dataloaders:
                #膨胀预测
                if args.predictmode =="expansion":
                    stride = 256  # 滑窗采样步长
                    pad_size = 128
                    pic = pic.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    #膨胀操作 边界padding
                    pic = pad(pic, pad=(pad_size, 2 * pad_size, pad_size, 2 * pad_size),
                              mode='reflect')  # ->expandsion原图->(1,1,5384,5384)
                    width_pad = width + 2 * pad_size
                    height_pad = height + 2 * pad_size
                    predict_holder = np.zeros((width_pad, height_pad))

                    for w in range(width_pad // stride):
                        for h in range(height_pad // stride):
                            img_roi = pic[:,:,h * stride:h * stride + img_size,w * stride:w * stride + img_size] #->512,512
                            # predict = model(img_roi) #-> 预测单张img_roi
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            else:
                                predict  = model(img_roi)
                            predict = torch.squeeze(predict).cpu().numpy()
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict_holder[h * stride:h * stride + (img_size // 2),w * stride:w * stride +(img_size // 2)] = predict[128:384,128:384 ]#->将预测值中心区域即（128，384）装入对应的容器矩阵对应位置# # predict_holder[predict_holder >= 0.5 ] = 1

                #正常预测
                elif args.predictmode =="samesample":
                    stride=512
                    pic = pic.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(0,stride - width % stride,0,stride - width % stride),value=0)  # ->expandsion原图->(1,1,5122,5122)
                    # padding 后的长宽
                    width_or  =width+ (stride - width % stride)
                    height_or =height+ (stride - height % stride)
                    predict_holder = np.zeros((width_or, height_or))
                    for w in range(width_or  // stride):
                        for h in range(height_or // stride):
                            img_roi = pic[:, :, h * stride:h * stride + img_size,
                                      w * stride:w * stride + img_size]  # ->512,512
                            if args.TTA == 'yes':
                                predict = tta_model(img_roi)
                            else:
                                predict = model(img_roi)
                            print(img_roi.shape )
                            print("%d,%d" %(h,w))
                            predict = torch.squeeze(predict).cpu().numpy()
                            predict_holder[h * stride:h * stride+stride ,w * stride:w * stride +  stride] = predict


                predict_holder = predict_holder[0:width,0:height]
                maxtrx = IOUMetric(2)
                mask = torch.squeeze(mask).cpu().numpy()#转换
                mask = mask.reshape((1,1,5000,5000)) #转换为可以计算的mask
                predict_holder[predict_holder>=0.5] =1
                predict_holder[predict_holder !=1] = 0
                predict_holder = predict_holder.reshape((1,1,5000,5000))
                maxtrx.add_batch(predict_holder, mask)
                acc, mean_iu = maxtrx.evaluate()
                miou_total +=mean_iu
                mpa_total+=acc
                print('img %d :,Miou=%f,acc=%f' % (i, mean_iu, acc))
                if i < num:
                    i += 1
                # mpa_total += acc
                # miou_total += mean_iu  # 获取当前预测图的miou，并加到总miou中
                # 图像
                mask = mask.reshape(5000,5000)
                predict_holder = predict_holder.reshape(5000,5000)
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('input')
                img =Image.open(pic_path[0])
                plt.imshow(img)
                # print(pic_path[0])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('predict_iou: %f' %mean_iu)
                plt.imshow(predict_holder, 'Greys_r')
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('mask')
                plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
                saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                saved_predict1 = '.' + saved_predict.split('.')[1] + '.tif'
                saved_predict2 = '.' + saved_predict.split('.')[1] +'mask'+ '.tif'
                saved_predict3 = '.' + saved_predict.split('.')[1]+'predict' + '.tif'
                Image.fromarray(np.uint8(mask * 255)).save(saved_predict2)  # saved_predict,saved_predict,np.uint8(predict_holder*255)
                # Image.fromarray(np.uint8(predict_holder * 255)).save(saved_predict3)
                blending_result(img, predict_holder).save(saved_predict3)

                plt.savefig(saved_predict1)

        miou_total = miou_total / i
        mpa_total  = mpa_total /i
        print("miou_total:%f,mpa_total: %f" % (miou_total,mpa_total))

#封闭测试
def predic(test_dataloaders,save_predict,combainer,args):
    if args.bbn == 'yes':
        tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean',combainer=combainer)
    else :
        tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean',
                                               combainer=combainer)
    logging.info('final test....')
    img_size = 512

    if save_predict == True:
        dir = os.path.join(r'./save_predict_closed', str(args.arch+'_'+args.loss), str(args.batch_size), str(args.epoch), str(args.dataset),
                           str(args.predictmode))
        outdir = os.path.join(r'./save_predict_closed', str(args.arch + '_' + args.loss), str(args.batch_size),
                           str(args.epoch), str(args.dataset),
                           str(args.predictmode+'_compress'))

        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            print('dir already exist!')
        #初始化模型调用
    if args.arch == 'BB_unet_var2':
        model.load_state_dict(torch.load(
            r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + 'reserve' + '_' + str(
                args.epoch)+'_'+ str(args.loss)  + '.pth', map_location="cuda"))  # 载入训练好的模型
    if args.arch == 'BB_unet_deepversion':
        model.load_state_dict(torch.load(
            r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + 'reserve' + '_' + str(
                args.epoch) + '_' + str(args.loss) + '.pth', map_location="cuda"))  # 载入训练好的模型
    elif args.arch == 'BB_unet':
        model.load_state_dict(torch.load(
            r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + 'reserve' + '_' + str(
                args.epoch)+'_'+ str(args.loss)  + '.pth', map_location="cuda"))  # 载入训练好的模型
    elif args.arch == 'UNet':
        model.load_state_dict(torch.load(
            r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + 'final' + '_' + str(
                args.epoch)+'_'+ str(args.loss)  + '.pth', map_location="cuda"))  # 载入训练好的模型

    elif args.arch == 'resnet34_unet':
        model.load_state_dict(torch.load(
            r'./save_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + 'final' + '_' + str(
                args.epoch)+'_'+ str(args.loss)  + '.pth', map_location="cuda"))  # 载入训练好的模型
    model.eval()
    # size_list = []
    # plt.ion() #开启动态模式
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        mpa_total = 0
        num = len(test_dataloaders)
        print("预测总数%d" % num)
        if args.bbn == 'yes':
            for img_x_c,pic_pathc in test_dataloaders:
                if args.predictmode =="expansion":
                    stride = 256  # 滑窗采样步长
                    pad_size = 128
                    # pad_size = (img_size - stride) // 2
                    pic = img_x_c.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(pad_size,2 * pad_size, pad_size,2* pad_size),
                                mode='reflect')  # ->expandsion原图->(1,1,5384,5384)
                    width_pad = width + 2 * pad_size
                    height_pad = height + 2 * pad_size
                    predict_holder = np.zeros((width_pad, height_pad))
                    for w in range(width_pad // stride ):
                        for h in range(height_pad // stride ):
                            # print('%d,%d' %(w,h))
                            img_roi = pic[:,:,h * stride:h * stride + img_size,w * stride:w * stride + img_size] #->512,512
                            # predict = combainer.bbn_unet(model,img_roi) #-> 预测单张img_roi
                            if args.TTA == 'yes':
                                predict = tta_model(img_roi)
                            else:
                                predict = combainer.bbn_unet(model, img_roi)
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict = torch.squeeze(predict).cpu().numpy()#->转换为没用batch的单张numpy形式矩阵
                            predict_holder[h * stride:h * stride + (img_size // 2),w * stride:w * stride +(img_size // 2)] = predict[128:384,128:384 ]#->将预测值中心区域即（128，384）装入对应的容器矩阵对应位置

                if  args.predictmode == 'samesample':
                    stride = 512
                    pic = img_x_c.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(0, stride - width % stride, 0, stride - width % stride),
                              value=0)  # ->expandsion原图->(1,1,5122,5122)
                    # padding 后的长宽
                    width_or = width + (stride - width % stride)
                    height_or = height + (stride - height % stride)
                    predict_holder = np.zeros((width_or, height_or))
                    for w in range(width_or // stride):
                        for h in range(height_or // stride):
                            img_roi = pic[:, :, h * stride:h * stride + img_size,
                                      w * stride:w * stride + img_size]  # ->512,512
                            # predict = combainer.bbn_unet(model,img_roi)  # -> 预测单张img_roi
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            else:
                                predict  = combainer.bbn_unet(model,img_roi)
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict = torch.squeeze(predict).cpu().numpy()
                            predict_holder[h * stride:h * stride + stride, w * stride:w * stride + stride] = predict

                predict_holder = predict_holder[0:width, 0:height]
                predict_holder[predict_holder >= 0.5] = 1
                predict_holder[predict_holder != 1] = 0
                predict_holder = predict_holder.reshape((5000, 5000))
                # mpa_total += acc
                # miou_total += mean_iu  # 获取当前预测图的miou，并加到总miou中
                # 图像
                saved_predict = dir + '/' + pic_pathc[0].split('\\')[-1]
                print(saved_predict)
                cv2.imwrite(saved_predict,np.uint8(predict_holder*255))


        else:
            for pic,pic_path in test_dataloaders:
                #膨胀预测
                if args.predictmode =="expansion":
                    stride = 256  # 滑窗采样步长
                    pad_size = 128
                    # pad_size = (img_size - stride) // 2
                    pic = pic.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(pad_size, 2 * pad_size, pad_size, 2 * pad_size),
                              mode='reflect')  # ->expandsion原图->(1,1,5384,5384)
                    width_pad = width + 2 * pad_size
                    height_pad = height + 2 * pad_size
                    predict_holder = np.zeros((width_pad, height_pad))
                    for w in range(width_pad // stride):
                        for h in range(height_pad // stride):
                            # print('%d,%d' %(w,h))
                            img_roi = pic[:, :, h * stride:h * stride + img_size,
                                      w * stride:w * stride + img_size]  # ->512,512
                            # predict = combainer.bbn_unet(model,img_roi) #-> 预测单张img_roi
                            # predict = tta_model(img_roi)
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            else:
                                predict  = model(img_roi)
                            print(img_roi.shape)
                            print("%d,%d" % (h, w))
                            predict = torch.squeeze(predict).cpu().numpy()  # ->转换为没用batch的单张numpy形式矩阵
                            predict_holder[h * stride:h * stride + (img_size // 2),
                            w * stride:w * stride + (img_size // 2)] = predict[128:384,
                                                                       128:384]  # ->将预测值中心区域即（128，384）装入对应的容器矩阵对应位置

                #正常预测
                elif args.predictmode =="samesample":
                    stride=512
                    pic = pic.to(device)
                    size_list = list(pic.size())
                    width = size_list[2]
                    height = size_list[3]
                    pic = pad(pic, pad=(0,stride - width % stride,0,stride - width % stride),value=0)  # ->expandsion原图->(1,1,5122,5122)
                    # padding 后的长宽
                    width_or  =width+ (stride - width % stride)
                    height_or =height+ (stride - height % stride)
                    predict_holder = np.zeros((width_or, height_or))
                    for w in range(width_or  // stride):
                        for h in range(height_or // stride):
                            img_roi = pic[:, :, h * stride:h * stride + img_size,
                                      w * stride:w * stride + img_size]  # ->512,512
                            # predict = tta_model(img_roi)  # -> 预测单张img_roi
                            if args.TTA == 'yes':
                                predict  = tta_model(img_roi)
                            else:
                                predict  = model(img_roi)
                            print(img_roi.shape )
                            print("%d,%d" %(h,w))
                            predict = torch.squeeze(predict).cpu().numpy()
                            predict_holder[h * stride:h * stride+stride ,w * stride:w * stride +  stride] = predict




                predict_holder = predict_holder[0:width, 0:height]
                predict_holder[predict_holder >= 0.5] = 1
                predict_holder[predict_holder != 1] = 0
                predict_holder = predict_holder.reshape((5000, 5000))
                # mpa_total += acc
                # miou_total += mean_iu  # 获取当前预测图的miou，并加到总miou中
                # 图像
                saved_predict = dir + '/' + pic_path[0].split('\\')[-1]
                # saved_predict3 = '.' + saved_predict.split('.')[1] + '.tif'
                print(saved_predict)
                Image.fromarray(np.uint8(predict_holder*255)).save(saved_predict)
        #按照官方要求压缩文件方便上传
        for file in os.listdir(dir):
            if file.endswith(".tif"):
                input_file = os.path.join(dir, file)
                output_file = os.path.join(outdir, file)
                command = "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 " + input_file + " " + output_file
                print(command)
                subprocess.call(command, shell=True)

        print("预测压缩完成：目录：")

        print(outdir)









if __name__ =="__main__":
    #------------------------------#
    batch_my  = 3000
    #用于在实验中自行控制的验证集的batch数
    #------------------------------#
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\npredict_mode:%s\nloss:%s,\nTTA:%s,\nscheduler:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset,args.predictmode,args.loss,args.TTA,args.scheduler))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\npredict_mode:%s\nloss:%s,\nTTA:%s,\nscheduler:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset,args.predictmode,args.loss,args.TTA,args.scheduler))
    print('**************************')
    model = getModel(args)
    #判断是否是BBN的网络
    if args.bbn == 'yes':
        Combiner = Combiner(args.epoch,device,args)
    else:
        Combiner = None
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)

    #设置dice损失函数
    if args.loss =='dice':
        criterion = L.DiceLoss(mode='binary')
    #设置交叉熵损失函数
    elif args.loss =='ce':
        criterion = torch.nn.BCELoss()
    # 设置组合损失函数
    elif args.loss =='ce_dice':
        Dice_fn =L.DiceLoss(mode='binary')
        #交叉熵,增加平滑性
        CEloss_fn = torch.nn.BCELoss()
        criterion = L.JointLoss(first=Dice_fn, second=CEloss_fn,
                              first_weight=0.5, second_weight=0.5)
    # 设置BB_unet_deepversion中组合损失函数
    if args.arch == 'BB_unet_deepversion':
        loss_side = L.DiceLoss(mode='binary')
        Combiner = Combinerv2(args.epoch,loss_side,device,args)
        criterion = torch.nn.BCELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-4,weight_decay=1e-3)
    #选择退火学习率策略
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2,eta_min=1e-5)
    #选择自适应下降学习率策略
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1,threshold=0.0001,verbose=True)

    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args,Combiner)
    if 'test' in args.action:
        test_predic(test_dataloaders, save_predict=True,combainer=Combiner,args=args,)
    if 'predict' in args.action:
        predic(test_dataloaders,save_predict=True,combainer=Combiner,args=args,)
