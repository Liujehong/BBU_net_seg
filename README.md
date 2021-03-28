# BBU-net_segmentation_SR
关于双分支U-net在遥感语义分割方向的实验性项目说明文档

实验通过对比使用重采样的方式平衡U-NET\U-NET_RESNET34来说明自适应平衡双分支U-net在非均衡遥感小目标数据集的分割工作上的优势。 


## 项目文件说明
![目录文件](/building_segmetation/readme_need/img1.png)
+ models：存放Unet、resnet-unet、BBu-net的核心模型代码
+ result
    + log :存放训练log
    + plot :存放训练过程画出的精度、loss、lr
+ save_model：保存的模型
+ save_predict_closed:封闭测试分割图像保存文件夹
+ save_predict:开放测试保存的文件夹
+ utils：
  + blending 将分割结果的掩膜覆盖在原图
  + datasets 加载的数据集
  + dataset_processing 数据集预处理与大图切割
  + metrics 得到混淆矩阵做评估
  + plot 画loss、lr、Iou、acc图像
+ compress用于压缩官方的预测文件
+ config 根目录配置文件
+ main 项目主程序（训练预测测试）
+ wrappers 更改过的TTA文件库
# 使用说明
* 首次使用`main.py`：在336行代码`    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean',combainer=combainer)` 按住`ctrl`在`SegmentationTTAWrapper`上点击左键跳转到库文件中，复制自己的`wrappers`中的代码到跳转的库文件中覆盖掉保存即可。       


## 网络
+ U-net
+ U-net（resnet34） -- baseline
+ BBU-net  -- ours
![目录文件](/building_segmetation/readme_need/BBUnet.png)


## 超参
在训练时epoch=20，batch_size=5,lr=1e-4。
优化方式：Adamw
## 学习率调整策略
+ CosineAnnealingWarmRestarts
```python
CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2，eta_min = 1e-5)
```

+ ReduceLROnPlateau
```python
lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1,threshold=0.001,verbose=True)
```
## loss
+ BE_loss (交叉熵)
+ dice_loss(相似系数)
+ be+dice （组合loss）
## 数据集
数据集官方源： [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
# 使用说明
## 训练
训练U-net：
```python
python main.py -ac train -e 20 -a UNet -b 5 -d final -bb no -s cos -l ce
```


训练resnet34_unet：
```python
python main.py -ac train -e 20 -a resnet34_unet -b 5 -d final -bb no -s cos -l ce
```

训练BBU-net：
```python
python main.py -ac train -e 20 -a BB_unet_var2 -b 5 -d reserve -bb yes -l ce -s cos
```

**tips：由于mixup的关系要使用其他损失函数进行组合需要使用下面具有深监督的结构**
mixup机制理解如下：
![目录文件](/building_segmetation/readme_need/mixup.png)

训练BBU-net_deepversion(ce+dice)：
```python
python main.py -ac train -e 20 -a BB_unet_deepversion -b 5 -d reserve -bb yes -l ce_dice -s cos
```




## 验证
+ 指标
  + 指标IoU
  + 指标PA
+ 验证集=>数据集的10%



## 测试
### 开放测试
选取数据集10%作为测试集与真实标签进行对比、
+ 验证U-net：
```python
python main.py -ac test -e 20 -a UNet -b 5 -d final -bb no -l ce
```
+ 验证resnet34_unet：
```python
python main.py -ac test -e 20 -a resnet34_unet -b 5 -d final -bb no -l ce
```
+ 验证BBU-net：
```python
python main.py -ac test -e 20 -a BB_unet_var2 -b 5 -d reserve -bb yes -l ce_dice
```
+ 验证BBU-net_deepversion：
```python
python main.py -ac test -e 20 -a BBU-net_deepversion -b 5 -d reserve -bb yes -l ce_dice
```

### 封闭测试
使用数据源提供的无公开标签的原图进行分割预测然后使用`compress.py`压缩分割图像然后上传官网测试接口进行封闭测试，以避免争对原始数据的强制拟合训练。
**步骤：**
* 执行程序
如果需要使用trick：TTA（测试时增强） 则`-tt`参数使用`yes`，不使用则使用`no`
**PS：注意使用TTA会带来约1%的提升但是预测时间变为原来的8倍！！**

*注意下面命令中`-e`,`-b`后的参数要改为和训练时一样*
+ BB_unet_var2:
```python
python main.py -ac predict -e 20 -a BB_unet_var2 -b 5 -d testdata -bb yes -tt no -l ce
```
+ U-net
```python
python main.py -ac predict -e 20 -a UNet -b 5 -d testdata -bb no -tt no -l ce
```
+ U-net(resnet34)
```python
python main.py -ac predict -e 20 -a resnet34_unet -b 5 -d testdata -bb no -tt no -l ce
```

+ BB_unet_deepversion:
```python
python main.py -ac predict -e 20 -a BB_unet_deepversion -b 5 -d testdata -bb yes -tt no -l ce_dice
```


* 上传
[官方测试上传地址](https://www.lri.fr/~gcharpia/aerial_benchmark/) 
填写邮箱上传压缩文件，等待2分钟等待官方通知结果。

## 预测对比
![目录文件](/building_segmetation/readme_need/网络预测对比图.png)

## 命令行参数：
### action操作
```
             | -train
-ac：     操作| -test
             | -predict
```

### epoch迭代次数(int)、batch_size(int:为了训练完整选择5的倍数)
```         
-e  [正整数]  -b [正整数]       
```

### dataset数据集
```         

-d：   数据集-| 
             | -final   （均匀采样）
             | -reserve   （适用于BBNU-net的反采样）
             | -testdata    （封闭测试用）    
```

### predictmode预测方式:普通滑窗预测和本文使用的膨胀预测
```   
        |-samesample （普通预测）
-p ---  |
        |-expansion  （膨胀预测）
```

### arch网络结构 
```
             | -UNet
             | -resnet34_unet
             | -resnet34_unet_nolock
-a：网络结构-| 
             | -BB_unet     
             | -BB_unet_var2
             | -BB_unet_deepversion
```

### BBN网络结构确认
```
       |-yes （使用双分支网络一定选择yes）
-bb ---|
       |-no  （非双分支使用no）
```

### TTA预测选择
```
       |-yes 
-tt ---|
       |-no  
```


# 代码说明
## `main.py`:
```python
#函数说明
def getArgs():  #命令行封装
def getLog(args): # log文件函数 输入：命令行参数args ，输出：无
def getModel(args): #网络选择函数 输入：命令行参数args ，输出：model
def getDataset(args): #数据集选择函数 输入：命令行参数args ，输出：train_dataloaders,val_dataloaders,test_dataloaders

#训练函数
def train(model, criterion, optimizer, train_dataloader,val_dataloader, args,conbainer) 
'''
 input：model 模型
       criterion 损失函数
       optimizer 优化器
       train_dataloader 训练集
       train_dataloader 验证集
       args 命令行参数
       conbainer BBN网络连接器

 return:model 每次迭代的model
'''


#验证函数
def val(conbainer,model,best_iou,val_dataloaders,args) 
'''
 input：
       conbainer BBN网络连接器
       model 模型
       best_iou 最优模型
       val_dataloader 验证集
       args 命令行参数

return best_iou 最佳IoU
       aver_miou 当前迭代训练结果在验证集上的miou
       aver_acc 当前迭代训练结果在验证集上的aver_acc
'''


#开放测试函数 （测试集为数据集的10%）
def test_predic(test_dataloaders,save_predict,combainer,args)
'''
 input：
       test_dataloaders 测试集
       save_predict 预测结果保存
       conbainer BBN网络连接器
       args 命令行参数

return best_iou 最佳IoU
       aver_miou 当前迭代训练结果在验证集上的miou
       aver_acc 当前迭代训练结果在验证集上的aver_acc
'''

#封闭测试函数
def predic(test_dataloaders,save_predict,combainer,args)
'''
 input：
       test_dataloaders 测试集
       save_predict 预测结果保存
       conbainer BBN网络连接器
       args 命令行参数


'''

```

## `config.py`:
定义全局根目录，移至项目只需要更改root根目录即可
```python
#将下面地址换成项目主地址
ROOT_PATH = r'E:\BBU_net_seg'
```
## `models/BB_unet.py`:
定义了BB_unet的网络结构
```python
class DoubleConv(nn.Module) #双重卷积类 构造Unet中的双重卷积结构
class DecoderBlock(nn.Module) #解码类 用于Unet中的上采样结构
class backbone(nn.Module) #用于构造BBU_net权重共享部分的主干网络结构
class finalbone(nn.Module) #用于构造上采样后两个不同分支的最终层网络结构
class BB_unet(nn.Module) #BB_unet 单分分类器版本
class BB_unet_var2(nn.Module) #BB_unet 双分分类器版本
class BB_unet_deepversion(nn.Module) #带有深监督分支的便于组合损失函数的BB_unet

#BBU_net网络输入图： 图a ->backbone ->分支a ->combainer.forword\
#                                                              | 连接两个分支的特征图-> 最终分类器finalbone->结果
#                   图b ->backbone ->分支b ->combainer.forword/
```

## `models/conbainer.py`:
BBunet两个分支的连接器
```python
def __init__(self, epoch_number, device,args) #用于初始化combainer
def initilize_all_parameters(self) #用于初始化训练参数 （epoch超过100时使用）
def reset_epoch(self,epoch) #得到当前迭代次数用于调整l
def forward(self, model, criterion, image1, label1,image2,label2,**kwargs) #前向传播 用于训练
def bbn_unet(self, model, image1,  **kwargs)  #用于预测的函数

tips：combainerv2跟此连接器相同 只是附加了深监督用于组合loss
```


## `models/unet.py`:
BBunet两个分支的连接器
```python
class DoubleConv(nn.Module) #双重卷积类 构造Unet中的双重卷积结构
class DecoderBlock(nn.Module) #解码类 用于Unet中的上采样结构
class Unet(nn.Module) #unet 
class resnet34_unet(nn.Module) #resnet34 unet
```


## `utils/dataset.py`:
数据集
```python
class flinalDataset(data.Dataset) #均匀采样
class reverseDataset(data.Dataset) #反采样
class testDateset(data.Dataset) #测试数据集
```


## `utils/dataset_processing.py`:
数据集
```python
图片的预处理 参考具体注释
```

## `utils/doubleimg.py`:
分割结果掩膜叠加原图显示
```python
def blending_result(image,mask)
'''
 input：
       image 原图
       mask 预测结果图
 return：blending_result 叠加结果


'''
```



## `utils/metrics.py`:
得到混淆矩阵用于计算IoU
```python
class IOUMetric:
'''
 计算混淆矩阵
    def _fast_hist(self, label_pred, label_true)
 将图片添加成一个batch
    def add_batch(self, predictions, gts)
 评估
    def evaluate(self)
'''
```
## `utils/myloss.py`:
已经作废，使用pytorch-tool库中的loss

## `utils/plot.py`:
画图工具
```python
def loss_plot(args,loss) #画loss图 input ：loss
def lr_plot(args,lr,numbatch) #画lr图 input ：len（dataset） 一代图片数量
def metrics_plot(arg,name,*args) #画精度图
```
