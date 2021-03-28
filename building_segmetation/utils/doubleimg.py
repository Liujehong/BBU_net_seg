# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# image 原图
# mask 分割图
def blending_result(image,mask):

    image1 = image.convert('RGBA')
    color = np.zeros([mask.shape[0],mask.shape[1], 3],dtype=np.uint8)
    color[mask==1] = [0, 255, 0]
    image2 =Image.fromarray(color).convert('RGBA')
    # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    blending_result= Image.blend(image1, image2, 0.3)
    return  blending_result
