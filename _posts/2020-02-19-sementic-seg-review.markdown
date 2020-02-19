---
layout: post
title:  "A Review on Semantic Segmentation"
date:   2020-02-19 18:57:00 +0800
categories: ML
---
# Datasets
* Mapillary Vistas
* Cityscapes
* Berkeley Deep Drive
* KITTI

![image.png](https://i.loli.net/2020/01/27/ZR7aHmhEO15tgrk.png)

# Metrics
1. (mean) IoU - Intersection over Union
2. (mean) PA - Pixel Accuracy
3. PR curve - Precision-Recall curve
4. F1 score
5. Confusion Matrix
6. 

# Feature Extracter
## Alexnet-like
Alexnet, VGGnet, GoogLeNet...

## Resnet-like
### 1. Resnet  
**Motivation**: 在实验中发现*degradation*现象  
*degradation*: 随着网络深度增加, training error上升. 代表网络hard to optimize  

**Contribution**: 提出*skip connection*  
* 达成类似动态深度的效果  
* 解决了vanishing gradients  
* 减少了*layer*之间的依赖性, 可以通过dropout来加速训练 [reference](http://papers.nips.cc/paper/6556-residual-networks-behave-like-ensembles-of-relatively-shallow-networks.pdf)  

![image.png](https://i.loli.net/2020/01/23/uJoZ51BG86F7aCs.png) 

### 2. Resnet v2
Motivation: 优化之前的residual unit

> Residual block abstraction:
> $$y_l = x_l + \mathcal{F}(x_l, \mathcal{W}_l)$$
> $$x_{l+1} = f(y_l)$$

经过实验, clean的信号通道(identity mapping)效果比使用scaling, gating 都好  
因此, 希望安排ReLU和BatchNorm的位置, 使得residual unit易于成为identity mapping  

**Contribution**: Proposes a new residual unit (Figure b)
1. Easier to train
2. Improves generalization

![image.png](https://i.loli.net/2020/01/23/6p47FxgqCLHScdy.png)  

**使用图(b)中结构的原理**:
假设: identity mapping 效果最好, 我们期望residual block输出范围为$\R$  
若ReLU应用在skip connection之后, 则输出是非负的, 从而影响该层的表达能力.  
若ReLU应用在residual path的最后, 那么f(x)就非负, activation value单调递增, 这显然不行.  

### 3. Densenet
![image.png](https://i.loli.net/2020/01/23/mbf9BoPDyNRgu83.png)


## FPN
![image.png](https://i.loli.net/2020/01/20/6z1VYJLKXpu8WmU.png)

## Astrous/dilated convolution
Deeplab v3: ASPP

# Semantic segmentation
[A Comparative Study of Real-time Semantic Segmentation for Autonomous Driving](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w12/Siam_A_Comparative_Study_CVPR_2018_paper.pdf)

![image.png](https://i.loli.net/2020/01/29/pjAfhLZSwvyXa2l.png)  

![image.png](https://i.loli.net/2020/01/29/59MJVgZ7hWwzRb4.png)


## Patch classification: CNN + Dense
Obsolete method.

## FCN: Fully Convolution
[Code](https://github.com/wkentaro/pytorch-fcn)  

**Motivation**:   
* 传统的Patch classification时间和空间消耗非常大  
* 难以确定patch的大小位置(@RPN)  
* 感受野受限  

**Method**: 
以transposed convolution上采样, 将缩小的图像尺寸还原  
过于严重的upsampling会影响精度, 因此使用skip connection将之前的层做融合.  

> combines semantic information from a deep, coarse layer with appearance informationfrom a shallow, fine layer to produce accurate and detailed segmentations  
inference 5 FPS.  

**Contribution**:
* Set a baseline structure
* Take arbitrary input size
* Efficient: no dense layers

![image.png](https://i.loli.net/2020/01/20/6K1qtw8lRpLid9Q.png)  

## U-net: Encoder-decoder
**Motivation**: 
* 同FCN, 旧的Patch classification不够好
* 医学领域数据稀少

**Method**:  
* Intense data augmentation (not as useful in auto driving)
* Use mirroring as padding (not useful in auto driving)
* Weight map: 给紧邻细胞间的缝隙加权, 训练模型识别其为背景  

![image.png](https://i.loli.net/2020/01/27/hPKTuXdvsGwnNep.png)

**Contribution**:
* Requires little data
* New feature fusion method: concatenate by channel (previously: addition (FCN))
* Setted the *Encoder-decoder* genre

## Segnet: Encoder-decoder
TODO

## Deeplab v1, v2: CRF, Dilated convolutions

### Dilated/atrous convolution
Conv layer with holes  
Based on the hypothesis: adjacent pixels are similar
![image.png](https://i.loli.net/2020/01/20/vrHFCTLjDKYNIX5.png)  
parameter amount ↓, receptive field↑  

*To avoid the gridding effect, the rate should be carefully considered.*  


### ASPP: Atrous Spatial Pyramid Pooling  
Capture objects & contexts at multiple scales  
> 在模型最后进行像素分类之前增加一个类似 Inception 的结构，包含不同 rate（空洞间隔） 的 Atrous Conv（空洞卷积），增强模型识别不同尺寸的同一物体的能力  

![image.png](https://i.loli.net/2020/01/28/EJUBI7A9vn1z5C6.png)

### CRF
利用相邻pixel的关系, e.g. 颜色相近的相邻像素更有可能属于同一类
用二元势函数描述像素和像素之间的关系, 用以细化边缘

CRF 过于缓慢, 且难以训练 即 不适合back propagation
在Deeplab v3中被丢弃
However, CRF is still used in industrial models.

## RefineNet
## PSPNet
## Large Kernel Matters
## Deeplab v3
[paper](https://arxiv.org/pdf/1706.05587.pdf)

**Motivation**:
* 深入研究ASPP module
* 改善Deeplab v2

**Method**:  
* Include BN to speed up training
* 改进了ASPP: 采用并行结构, 以不同atrous rate在多种尺度上捕捉物体, 添加了 1x1conv 和 Image pooling (全局信息)

![image.png](https://i.loli.net/2020/01/28/PrCheXcnNU8HBaR.png)

## Deeplab v3+ (SOTA)
**Motivation**:  
之前的模型e.g.PSPNet, 虽然提取了high level feature, 却没有足够的关于object boundary的信息来达成更精细的分割. 若用astrous convolution来维持较大的resolution, 在较深网络中计算量太大. *As in the figure below, note the difference in feature map sizes*

**Method**:  
* 添加decoder模块(双线性插值upsample), 将之前的v3模型视为encoder
* 融合修改过的Xception模型
* 使用depthwise separable convolution减少参数量

### depthwise separable convolution
通常: N个kernel对每个channel卷积并加和, 参数量为$\text{in\_channel} \times k \times k \times N$  
depthwise separable convolution: 每个kernel仅对一个通道卷积, 即$N = \text{in\_channel}$, 随后使用1x1 conv来调整channel数量. 参数量$k \times k \times N + \text{(1x1 conv)}$  

**Contribution**:  
* A state-of-the-art model
* *Cited from original paper*: Can control the feature size to trade speed and accuracy
* 更加确定了encoder-decoder架构的优越性

![image.png](https://i.loli.net/2020/01/28/u1c6oxWJUaDHZ3Q.png)  
![image.png](https://i.loli.net/2020/01/28/T1GhgKPqEZuVoX6.png)

## HRNet
**almost state of the art**
[paper](https://paperswithcode.com/paper/high-resolution-representations-for-labeling)

## OCR
**state of the art**
[paper](https://paperswithcode.com/paper/object-contextual-representations-for)

## Summary
![image.png](https://i.loli.net/2020/01/27/YVQbUyEdwLhqv2e.png)  
![image.png](https://i.loli.net/2020/01/27/4a65XULgEsW27Ck.png)  

---
references:  
[2017综述](https://blog.csdn.net/qq_43222384/article/details/90729438)  
[2019综述blog](https://blog.csdn.net/qq_41997920/article/details/96479243)  
[2019综述论文: Understanding Deep Learning Techniques for Image Segmentation](https://arxiv.org/pdf/1907.06119.pdf)  
