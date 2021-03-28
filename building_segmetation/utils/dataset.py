import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from config import *





# 均匀取样
class flinalDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = ROOT_PATH
        self.img_paths = None
        self.mask_paths = None
        self.p_p, self.p_n = self.getweight()
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
    #得到分支抽样概率
    def getweight(self):
        img_P = glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))
        img_N = glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))
        w_p = (len(img_P) + len(img_N)) / len(img_P)  # ->wp
        w_n = (len(img_P) + len(img_N)) / len(img_N)  # ->wn
        p_p = w_p / (w_p +w_n) #在P中抽取样本的概率
        p_n = w_n / (w_p +w_n) #在N中抽取的概率
        return p_p,p_n
    #得到path
    def getDataPath(self):
        img_P = glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))
        img_N = glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))
        img_path_np = np.array(img_N + img_P)
        np.random.shuffle(img_path_np)
        train_img_paths = img_path_np.tolist()
        val_img_paths = glob(os.path.join(self.root, r'dataset\val\images\*'))
        test_img_paths = glob(os.path.join(self.root, r'dataset\test\images\*'))
        self.train_img_paths = train_img_paths
        # self.train_mask_paths = glob(self.root + r'\train\gt\*')
        self.val_img_paths = val_img_paths
        # self.val_mask_paths = glob(self.root + r'\val\gt\*')
        self.test_img_paths = test_img_paths
        # self.test_mask_paths = glob(self.root + r'\test\gt\*')
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    #得到单张图片
    def __getitem__(self, index):
        if self.state == 'train':
            #在P（0）和N（1）样本中按照概率抽取
            choice = np.random.choice(a=[1, 0], size=1, p=[self.p_n, self.p_p])
            if choice == 0:
                semple_path = np.random.choice(a=
                                               np.array(glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))),
                                               size=1)
            else:
                semple_path = np.random.choice(a=
                                               np.array(glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))),
                                               size=1)
            pic_path = semple_path[0]
            mask_path = pic_path.replace('images', 'gt')
            pic = np.array(Image.open(pic_path))
            mask =np.array(Image.open(mask_path))
            pic = pic.astype('float32') / 255
            mask = mask.astype('float32') / 255
            # if self.aug:
            #     if random.uniform(0, 1) > 0.5:
            #         pic = pic[:, ::-1, :].copy()
            #         mask = mask[:, ::-1].copy()
            #     if random.uniform(0, 1) > 0.5:
            #         pic = pic[::-1, :, :].copy()
            #         mask = mask[::-1, :].copy()
            if self.transform is not None:
                img_x = self.transform(pic)
            if self.target_transform is not None:
                img_y = self.target_transform(mask)

            return img_x, img_y, pic_path, mask_path

        if self.state == 'val' or self.state == 'test':
            pic_pathc = self.pics[index]
            mask_pathc = pic_pathc.replace('images', 'gt')
            pic_c = np.array(Image.open(pic_pathc))
            mask_c = np.array(Image.open(mask_pathc))
            pic_c = pic_c.astype('float32') / 255
            mask_c = mask_c.astype('float32') / 255
            if self.transform is not None:
                img_x = self.transform(pic_c)
            if self.target_transform is not None:
                img_y = self.target_transform(mask_c)
            return img_x, img_y, pic_pathc, mask_pathc


    def __len__(self):
        return len(self.pics)
# 反采样取样
class reverseDataset(data.Dataset):
    def __init__(self, state,sample, transform=None, target_transform=None):
        self.state = state
        self.sample = sample
        self.aug = True
        self.root = ROOT_PATH
        self.p_p,self.p_n = self.getweight()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks,= self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        img_P = glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))
        img_N = glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))
        img_path_np = np.array(img_N+img_P)
        #打乱标签
        np.random.shuffle(img_path_np)
        train_img_paths = img_path_np.tolist()
        val_img_paths = glob(os.path.join(self.root, r'dataset\val\images\*'))
        test_img_paths = glob(os.path.join(self.root, r'dataset\test\images\*'))
        self.train_img_paths = train_img_paths
        self.val_img_paths = val_img_paths
        self.test_img_paths = test_img_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    #计算正负样本抽取概率
    def getweight(self):
        img_P = glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))
        img_N = glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))
        w_p = (len(img_P) + len(img_N)) / len(img_P)  # ->wp
        w_n = (len(img_P) + len(img_N)) / len(img_N)  # ->wn
        p_p = w_p / (w_p +w_n) #在P中抽取样本的概率
        p_n = w_n / (w_p +w_n) #在N中抽取的概率
        return p_p,p_n


 # 注意这里得到的是两个输入 pic_pathr => 反采样分支输入
 #                         pic_pathc =>传统学习分支输入
    def __getitem__(self, index):
        if self.state == 'train':
            semple_path = None
            pic_pathc = self.pics[index] #传统分支均匀采样训练
            #如果sample为yes 说明使用的bbn结构，需要根据得到的概率抽取正负样本
            if self.sample == 'yes' :
                #choice操作就是在按照两个概率分别从0代表的正样本和1代表的负样本抽取输入的pic_pathc
                choice = np.random.choice(a=[1,0],size=1,p=[self.p_n,self.p_p])
                if choice == 0:
                    semple_path = np.random.choice(a =
                                                np.array(glob(os.path.join(self.root, r'BNN_datasets\P_datasets\train\images\*'))),
                                                size = 1)
                else:
                    semple_path = np.random.choice(a =
                                                np.array(glob(os.path.join(self.root, r'BNN_datasets\N_datasets\train\images\*'))),
                                                size = 1)

            pic_pathr = semple_path[0]#pic_pathr作为反采样分支输入训练的输入
            mask_pathc = pic_pathc.replace('images','gt')
            mask_pathr = pic_pathr.replace('images','gt')
            # print(mask_path)
            # origin_x = Image.open(x_path)
            # origin_y = Image.open(y_path)
            pic_c = np.array(Image.open(pic_pathc))
            mask_c =np.array(Image.open(mask_pathc))
            pic_r = np.array(Image.open(pic_pathr))
            mask_r =np.array(Image.open(mask_pathr))
            pic_c = pic_c.astype('float32') / 255
            mask_c = mask_c.astype('float32') / 255
            pic_r = pic_r.astype('float32') / 255
            mask_r = mask_r.astype('float32') / 255
            # if self.aug:
            #     if random.uniform(0, 1) > 0.5:
            #         pic = pic[:, ::-1, :].copy()
            #         mask = mask[:, ::-1].copy()
            #     if random.uniform(0, 1) > 0.5:
            #         pic = pic[::-1, :, :].copy()
            #         mask = mask[::-1, :].copy()
            if self.transform is not None:
                img_x_c = self.transform(pic_c)
                img_x_r = self.transform(pic_r)
            if self.target_transform is not None:
                img_y_c = self.target_transform(mask_c)
                img_y_r = self.target_transform(mask_r)
            if self.sample == 'yes' :
                return img_x_c,img_x_r, img_y_c,img_y_r, pic_pathc,mask_pathc,pic_pathr,mask_pathr
            else:
                return img_x_c, img_y_c, pic_pathc, mask_pathc

        if self.state == 'val' or self.state == 'test':
            pic_pathc = self.pics[index]
            mask_pathc = pic_pathc.replace('images', 'gt')
            pic_c = np.array(Image.open(pic_pathc))
            mask_c = np.array(Image.open(mask_pathc))
            pic_c = pic_c.astype('float32') / 255
            mask_c = mask_c.astype('float32') / 255
            if self.transform is not None:
                img_x_c = self.transform(pic_c)
            if self.target_transform is not None:
                img_y_c = self.target_transform(mask_c)
            if self.sample == 'yes':
                return img_x_c,img_x_c,img_y_c,img_y_c,pic_pathc,mask_pathc,pic_pathc,mask_pathc
            else:
                return img_x_c, img_y_c, pic_pathc, mask_pathc




    def __len__(self):
        return len(self.pics)

#官方测试数据集
class testDateset(data.Dataset):
    def __init__(self, state,sample, transform=None, target_transform=None):
        self.state = state
        self.sample = sample
        self.aug = True
        self.root = ROOT_PATH
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics= self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        #原始目录
        test_img_paths = glob(os.path.join(ROOT_PATH,r'NEW2-AerialImageDataset\test\images\*'))
        #根目录
        # dir = r'F:\dataset_BBN\NEW2-AerialImageDataset\AerialImageDataset\test\images'
        # # 已经预测的：
        # path1 = glob(r'F:\dataset_BBN\NEW2-AerialImageDataset\savepredict\BB_unet_var2\5\10\testdata\expansion\*')
        # # 原图：
        # path2 = glob(r'F:\dataset_BBN\NEW2-AerialImageDataset\AerialImageDataset\test\images\*')
        # # 取二者集合的补集
        # for i in range(len(path1)):
        #     path1[i] = path1[i].split('\\')[-1]
        # for i in range(len(path2)):
        #     path2[i] = path2[i].split('\\')[-1]
        # path_now = set(path1) ^ set(path2)
        # path_now = list(path_now)
        # for i in range(len(path_now)):
        #     path_now[i] =dir + '\\'+ path_now[i]
        self.test_img_paths = test_img_paths
        return self.test_img_paths




    def __getitem__(self, index):
        pic_pathc = self.pics[index]
        pic_c = np.array(Image.open(pic_pathc))
        pic_c = pic_c.astype('float32') / 255
        if self.transform is not None:
            img_x_c = self.transform(pic_c)
        return img_x_c,pic_pathc



    def __len__(self):
        return len(self.pics)
