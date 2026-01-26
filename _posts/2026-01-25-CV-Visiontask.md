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

#### 思路一: Sliding Window单像素没法进行分类

Learning Hierarchical Features for Scene Labeling(**TPAMI 2013**)



Recurrent Convolutional Neural Networks for Scene Labeling(**ICML 2014**)



**问题**:效率很低,没有利用重叠的patchs的共享features.

#### 思路二:全卷积

idea1: 提取上下文空间特征,不下采样从而保证输出和输入的shape一致$\Rightarrow$ 太贵了

Idea2:先下采样然后再上采样来keep shape

Fully Convolutional Networks for Semantic Segmentation(**CVPR 2015**)



Learning Deconvolution Network for Semantic Segmentation(**ICCV 2015**)