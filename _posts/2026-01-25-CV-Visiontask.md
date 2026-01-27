---
title: "计算机视觉的主要任务和进展"
date: 2026-01-25
categories: [Computer vision]
tags: [Computer vision]
math: true

---

# 计算机视觉任务、可视化和理解

## Semantic Segmentation

![figure1]({{ "/assets/img/semantic.png" | relative_url }})

**训练集:** 给每个像素点 打上语义类别标签

**测试** : 给图片的每个像素点分类

#### 思路一: Sliding Window ,上下文提取特征

Learning Hierarchical Features for Scene Labeling(**TPAMI 2013**)

Recurrent Convolutional Neural Networks for Scene Labeling(**ICML 2014**)



**问题**:效率很低,没有利用重叠的patchs的共享features.

#### 思路二:全卷积

idea1: 提取上下文空间特征,不下采样从而保证输出和输入的shape一致$\Rightarrow$ 太贵了

Idea2:先下采样然后再上采样来keep shape

![figure1]({{ "/assets/img/fullconv.png" | relative_url }})

* Fully Convolutional Networks for Semantic Segmentation(**CVPR 2015**)

* Learning Deconvolution Network for Semantic Segmentation(**ICCV 2015**)

**下采样** : 先Pooling 再 Stride Convolution

**上采样** : Unpooling

![figure1]({{ "/assets/img/uppooling1.png" | relative_url }})

![figure1]({{ "/assets/img/uppooling2.png" | relative_url }})

**可学习的版本**: 反卷积(Transpose Convolution)

  ![figure1]({{ "/assets/img/transposeconvolution.png" | relative_url }})

#### 思路三: U-Net

![figure1]({{ "/assets/img/unet.png" | relative_url }})

$\color{red} 别关心实例,只关注像素 $​

* 先downsampling 再upsampling

## 物体检测=Classification + Localization

![figure1]({{ "/assets/img/objectdetection.png" | relative_url }})

但是对于多物体检测, Each image needs a different number of outputs!

#### 思路一: 将图片分成多个patch然后分别进行单物体检测



Problem: Need to apply CNN to huge number

of locations, scales, and aspect ratios, very

computationally expensive!

#### 思路二: Region Proposals: Selective Search

● Find “blobby” image regions that are likely to contain objects

● Relatively fast to run; e.g. Selective Search gives 2000 region

proposals in a few seconds on CPU



#### 思路三: R-CNN

Rich feature hierarchies for accurate object detection and semantic segmentation(**CVPR 2014**)



### Modern Method

#### Single-Stage Object Detectors: YOLO / SSD / RetinaNet

#### Yolo(CVPR 2016)



#### DETR:Object Detection with Transformers(ECCV 2020)

























