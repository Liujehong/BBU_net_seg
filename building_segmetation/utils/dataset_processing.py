import glob
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import os
import numpy as np
from tqdm import tqdm
#------------------参数------------------#
size = 512
stride = 128
# 要裁剪图像的大小

img_w = 512
img_h = 512
#------------------参数------------------#


#----------------图像增强操作--------------#
# 添加噪声
def add_noise(img):
    drawObject = ImageDraw.Draw(img)
    for i in range(250):  # 添加点噪声
        temp_x = np.random.randint(0, img.size[0])
        temp_y = np.random.randint(0, img.size[1])
        drawObject.point((temp_x, temp_y), fill="white")  # 添加白色噪声点,噪声点颜色可变
    return img


# 色调增强
def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img


def data_augment(src_roi, label_roi):
    # 图像和标签同时进行90，180，270旋转
    if np.random.random() < 0.5:
        src_roi = src_roi.rotate(90)
        label_roi = label_roi.rotate(90)
    if np.random.random() < 0.5:
        src_roi = src_roi.rotate(180)
        label_roi = label_roi.rotate(180)
    if np.random.random() < 0.5:
        src_roi = src_roi.rotate(270)
        label_roi = label_roi.rotate(270)
    # 图像和标签同时进行竖直旋转
    if np.random.random() < 0.5:
        src_roi = src_roi.transpose(Image.FLIP_LEFT_RIGHT)
        label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)
    # 图像和标签同时进行水平旋转
    if np.random.random() < 0.33:
        src_roi = src_roi.transpose(Image.FLIP_TOP_BOTTOM)
        label_roi = label_roi.transpose(Image.FLIP_TOP_BOTTOM)
    # 图像进行高斯模糊
    if np.random.random() < 0.2:
        src_roi = src_roi.filter(ImageFilter.GaussianBlur)
    # 图像进行色调增强
    if np.random.random() < 0.2:
        src_roi = random_color(src_roi)
    # 图像加入噪声
    if np.random.random() < 0.2:
        src_roi = add_noise(src_roi)
    return src_roi, label_roi


#------------------------------------------#
'''
带有筛选性质的滑窗增强采样生成的训练集
分成3种不同正负均衡度的样本进行训练
'''
#------------------划分数据集----------------#
def creat_dataset(images_num = 8650,mode='original'):
    data_path = r'G:\buiding\aerialimagelabeling\AerialImageDataset\train\images'
    print('creating dataset...')
    images_path = glob.glob(os.path.join(data_path, '*.tif'))
    images_each = images_num / len(images_path)
    gcount_n = 0
    gcount_m = 0
    gcount_p = 0
    for i in  tqdm(range(len(images_path))):
        count = 0
        src_img = Image.open(images_path[i])
        label_img = Image.open(images_path[i].replace('images', 'gt'))

        # while count < images_each:
        while 1:
            width1 = random.randint(0, src_img.size[0] - img_w)
            height1 = random.randint(0, src_img.size[1] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h

            src_roi = src_img.crop((width1, height1, width2, height2))
            label_roi = label_img.crop((width1, height1, width2, height2))
            np_label_roi = np.array(label_roi) / 255  # ->0,1
            metric = np.bincount(np_label_roi.flatten().astype(int), minlength=2)
            mask_precent = metric[1] / (metric[0] + metric[1])

            if  ((mask_precent  < 0.3) & (gcount_n<images_each)): #-> 负样本
                print("负样本 放入negative_dataset")
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)
                src_roi.save(r'F:\datasets\negative_dataset\train\images\%d.tif' % gcount_n)
                label_roi.save(r'F:\datasets\negative_dataset\train\gt\%d.tif' % gcount_n)
                gcount_n+=1
            elif ((mask_precent >0.4) & (mask_precent<0.5) &(gcount_m<images_each)): #->均衡样本
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)
                src_roi.save(r'F:\datasets\moderate_dataset\train\images\%d.tif' % gcount_m)
                label_roi.save(r'F:\datasets\moderate_dataset\train\gt\%d.tif' % gcount_m)
                print("均衡样本 放入moderate_dataset")
                gcount_m+=1
            elif ((mask_precent> 0.75)  &(gcount_p < images_each)) : #-> 正样本
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)
                src_roi.save(r'F:\datasets\position_dataset\train\images\%d.tif' % gcount_p)
                label_roi.save(r'F:\datasets\position_dataset\train\gt\%d.tif' % gcount_p)
                print("正样本 放入position_dataset")
                gcount_p+=1
                # if mode == 'augment':
                #     src_roi, label_roi = data_augment(src_roi, label_roi)
                # src_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\images\%d.tif' % g_count)
                # label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\gt\%d.tif' % g_count)
            if ((gcount_p==images_each)&(gcount_m==images_each)&(gcount_n==images_each)): # 三个数据集都够了之后
                break
'''
非随机滑窗采样生成测试集
'''
def creat_valdataset():
    data_path = r'G:\buiding\aerialimagelabeling\AerialImageDataset\val\images'
    print('creating dataset...')
    images_path = glob.glob(os.path.join(data_path, '*.tif'))
    g_count = 0
    for i in tqdm(range(len(images_path))):
        count = 0
        src_img = Image.open(images_path[i])
        label_img = Image.open(images_path[i].replace('images', 'gt'))
        # 得到宽高
        width, height = src_img.size
        # count = 0
        # 开始进行步长为stride的滑窗切割
        for w in range(width // stride):
            for h in range(height // stride):
                image_roi = src_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
                label_roi = label_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
                image_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\images\%d.tif' % g_count)
                label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\gt\%d.tif' % g_count)
                # image_sample = np.array(image_sample)
        #         plt.subplot(9, 9, count + 1)
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.grid(False)
        #         plt.imshow(image_sample)
        #         plt.xlabel('%02d' % count)
        #         count += 1
                g_count += 1
        # plt.show()



        # while count < images_each:
        #     width1 = random.randint(0, src_img.size[0] - img_w)
        #     height1 = random.randint(0, src_img.size[1] - img_h)
        #     width2 = width1 + img_w
        #     height2 = height1 + img_h
        #
        #     src_roi = src_img.crop((width1, height1, width2, height2))
        #     label_roi = label_img.crop((width1, height1, width2, height2))
        #     if (np.array(label_roi).max() == 0):
        #         print("图像为空不具备价值省略")
        #     else:
        #         if mode == 'augment':
        #             src_roi, label_roi = data_augment(src_roi, label_roi)
        #         src_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\images\%d.tif' % g_count)
        #         label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\gt\%d.tif' % g_count)
        #         count += 1
        #         g_count += 1


#------------------------------------------#
'''
带有筛选性质的滑窗增强采样生成的训练集
分成2种不同正负均衡度的样本进行训练
'''
#------------------划分数据集----------------#

def creat_traindataset(images_num = 30000,mode='original'):
    data_path = r'G:\buiding\aerialimagelabeling\AerialImageDataset\train\images'
    print('creating dataset...')
    images_path = glob.glob(os.path.join(data_path, '*.tif'))
    images_each = images_num // len(images_path)
    g_count = 0
    for i in tqdm(range(len(images_path))):
        count = 0
        src_img = Image.open(images_path[i])
        label_img = Image.open(images_path[i].replace('images', 'gt'))
        while count < images_each:
            width1 = random.randint(0, src_img.size[0] - img_w)
            height1 = random.randint(0, src_img.size[1] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h

            src_roi = src_img.crop((width1, height1, width2, height2))
            label_roi = label_img.crop((width1, height1, width2, height2))
            np_label_roi = np.array(label_roi) / 255  # ->0,1
            metric = np.bincount(np_label_roi.flatten().astype(int), minlength=2)
            mask_precent = metric[1] / (metric[0] + metric[1])
            if (mask_precent<0.5):
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)
                src_roi.save(r'F:\BNN_datasets\N_datasets\train\images\%d.tif' % g_count)
                label_roi.save(r'F:\BNN_datasets\N_datasets\train\gt\%d.tif' % g_count)
                count += 1
                g_count += 1
            else:
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)
                src_roi.save(r'F:\BNN_datasets\P_datasets\train\images\%d.tif' % g_count)
                label_roi.save(r'F:\BNN_datasets\p_datasets\train\gt\%d.tif' % g_count)
                count += 1
                g_count += 1

'''
步长切割
'''
# def creat_traindataset(mode='orginal'):
#     data_path = r'G:\buiding\aerialimagelabeling\AerialImageDataset\train\images'
#     print('creating dataset...')
#     images_path = glob.glob(os.path.join(data_path, '*.tif'))
#     g_count = 0
#     for i in tqdm(range(len(images_path))):
#         count = 0
#         src_img = Image.open(images_path[i])
#         label_img = Image.open(images_path[i].replace('images', 'gt'))
#         # 得到宽高
#         width, height = src_img.size
#         # count = 0
#         # 开始进行步长为stride的滑窗切割
#         for w in range((width -size)// stride):
#             for h in range((height-size )// stride):
#                 image_roi = src_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
#                 label_roi = label_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
#                 # image_roi.save(r'G:\buiding\aerialimagelabeling\dataset\train\images\%d.tif' % g_count)
#                 # label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\train\gt\%d.tif' % g_count)
#                 np_label_roi = np.array(label_roi) / 255  # ->0,1
#                 metric = np.bincount(np_label_roi.flatten().astype(int),minlength=2)
#                 mask_precent = metric[1] / (metric[0] + metric[1])
#                 # if mask_precent :
#                 print("不具备价值省略") #根据mask中正类占总类的样本小于0.1则不认为是有效目标
#                 # elif mask_precent: #根据mask中正类占总类的样本不能过高（0.75）则不认为是有效目标
#                 for i in range(2):
#                     if mode == 'augment':
#                         image_roi, label_roi = data_augment(image_roi, label_roi)
#                     image_roi.save(r'F:\datasets\final_dataset\train\images\%d.tif' % g_count)
#                     label_roi.save(r'F:\datasets\final_dataset\train\val\gt\%d.tif' % g_count)
#                     g_count += 1


#
#采样不同步长产生3类分别含有不同占比正负样本的数据集
#
def creat_3thdataset(mode='orginal',images_num = 8650):
    data_path = r'G:\buiding\aerialimagelabeling\AerialImageDataset\train\images'
    print('creating dataset...')
    images_path = glob.glob(os.path.join(data_path, '*.tif'))
    images_each  = images_num // len(images_path)
    gcount_n = 0
    gcount_m = 0
    gcount_p = 0
    for i in tqdm(range(len(images_path))):
        count_n = 0
        count_m = 0
        count_p = 0
        src_img = Image.open(images_path[i])
        label_img = Image.open(images_path[i].replace('images', 'gt'))
        # 得到宽高
        width, height = src_img.size
        # strides = [113,173,223 ,277 ,353 ,569 ,829 ,1019]
        strides = [1019,829,569,353,277,223]
        stride = 1019
        # count = 0
        # 开始进行步长为stride的滑窗切割
        # while 1:
        # stride  = random.randrange(128,4400,step=128)
        while 1:
            for stride in strides:
                for w in range((width-size) // stride):
                    for h in range((height-size) // stride):
                        src_roi = src_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
                        label_roi = label_img.crop((h * stride, w * stride, h * stride + size, w * stride + size))
                        # image_roi.save(r'G:\buiding\aerialimagelabeling\dataset\train\images\%d.tif' % g_count)
                        # label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\train\gt\%d.tif' % g_count)
                        np_label_roi = np.array(label_roi) / 255  # ->0,1
                        metric = np.bincount(np_label_roi.flatten().astype(int), minlength=2)
                        mask_precent = metric[1] / (metric[0] + metric[1])

                        if ((mask_precent < 0.5) & (count_n  < images_each)):  # -> 负样本
                            print("负样本 放入negative_dataset")
                            if mode == 'augment':
                                src_roi, label_roi = data_augment(src_roi, label_roi)
                            src_roi.save(r'F:\datasets\negative_dataset\train\images\%d.tif' % gcount_n)
                            label_roi.save(r'F:\datasets\negative_dataset\train\gt\%d.tif' % gcount_n)
                            gcount_n += 1
                            count_n +=1
                        elif ((mask_precent > 0.4) & (mask_precent <=0.6) & (count_m < images_each)):  # ->均衡样本
                            if mode == 'augment':
                                src_roi, label_roi = data_augment(src_roi, label_roi)
                            src_roi.save(r'F:\datasets\moderate_dataset\train\images\%d.tif' % gcount_m)
                            label_roi.save(r'F:\datasets\moderate_dataset\train\gt\%d.tif' % gcount_m)
                            print("均衡样本 放入moderate_dataset")
                            gcount_m += 1
                            count_m += 1

                        elif ((mask_precent > 0.5  ) & (count_p < images_each)):  # -> 正样本
                            if mode == 'augment':
                                src_roi, label_roi = data_augment(src_roi, label_roi)
                            src_roi.save(r'F:\datasets\position_dataset\train\images\%d.tif' % gcount_p)
                            label_roi.save(r'F:\datasets\position_dataset\train\gt\%d.tif' % gcount_p)
                            print("正样本 放入position_dataset")
                            gcount_p += 1
                            count_p += 1
                            # if mode == 'augment':
                            #     src_roi, label_roi = data_augment(src_roi, label_roi)
                            # src_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\images\%d.tif' % g_count)
                            # label_roi.save(r'G:\buiding\aerialimagelabeling\dataset\val\gt\%d.tif' % g_count)
            if (((count_p == images_each) & (count_m == images_each) & (count_n == images_each) )or  (stride ==223)):  # 三个数据集都够了之后
                break



if __name__ == '__main__':
    creat_traindataset(images_num = 30000,mode='augment')