---
title: "Diffusion-RL 复习（一）：PPO 与 GRPO"
date: 2026-05-22
excerpt: "复习 Diffusion-RL 相关基础知识，重点整理 PPO 与 GRPO。"
cover: "/images/2.png"
categories:
  - Reinforcement-Learning
  - Diffusion
tags:
  - PPO
  - GRPO
  - Diffusion-RL
  - Policy-Gradient
math: true
read_time: true
---

> 原始 PDF 中第 1 页为空白，第 2 页主要是 PPO 笔记，第 3 页只有“GRPO 算法”标题。本文已整理为适合 GitHub Pages / Jekyll 的 Markdown 源码。

## 目录

- [背景](#背景)
- [一、PPO 算法](#一ppo-算法)
  - [1. PPO 的目标](#1-ppo-的目标)
  - [2. 采样轨迹](#2-采样轨迹)
  - [3. 计算 Advantage](#3-计算-advantage)
  - [4. 概率比](#4-概率比)
  - [5. Clip Loss](#5-clip-loss)
  - [6. PPO 目标从哪里来](#6-ppo-目标从哪里来)
  - [7. 为什么需要 Clip](#7-为什么需要-clip)
  - [8. 关于 KL Divergence 的补充](#8-关于-kl-divergence-的补充)
- [二、GRPO 算法](#二grpo-算法)

---

鉴于新开始的项目，需要复习一下 Diffusion-RL 相关知识。本文先从 RL 相关内容开始，复习 **PPO 算法** 和 **GRPO 算法**。

---

## 一、PPO 算法

### 1. PPO 的目标

PPO 的目标是让 policy 更新更加稳定。

具体来说，PPO 通过限制每步 policy 的更新幅度，防止 policy 发生突变。

---

### 2. 采样轨迹

首先，旧策略 \(\pi_{\text{old}}\) 与环境交互，收集 \(T\) 步轨迹：

\[
\tau = (s_0,a_0,r_0),(s_1,a_1,r_1),\cdots
\]

其中：

- \(s_t\)：第 \(t\) 步的状态；
- \(a_t\)：第 \(t\) 步采取的动作；
- \(r_t\)：第 \(t\) 步获得的奖励；
- \(\pi_{\text{old}}\)：采样数据时使用的旧策略。

---

### 3. 计算 Advantage

#### 3.1 TD 残差

先计算 TD 残差：

\[
\delta_t = Q(s_t,a_t)-V_\phi(s_t)
\]

在实际估计时，可以写成：

\[
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
\]

它表示：当前 action 相比当前状态下的平均水平好多少。

---

#### 3.2 GAE Advantage

使用 GAE（Generalized Advantage Estimation）估计 advantage：

\[
\hat A_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
\]

其中：

- \(\gamma\)：折扣因子；
- \(\lambda\)：GAE 平滑系数；
- \(\delta_{t+l}\)：未来 timestep 的 TD 残差。

当 \(\lambda \to 0\) 时：

\[
\hat A_t \approx \delta_t
\]

此时更接近单步 TD，具有 **低方差、高偏差** 的特点。

当 \(\lambda \to 1\) 时：

\[
\hat A_t \approx \sum_{l=0}^{T-t}\gamma^l \delta_{t+l}
\]

此时更接近 Monte Carlo 估计，具有 **高方差、低偏差** 的特点。

需要注意的是，这里的 \(\lambda\) 建模的是从当前 \(t\) 开始，后续每个 timestep 的 TD 残差贡献。

如果 policy 更新之后，后续状态分布和动作分布都会发生变化，那么旧数据里的 \(\delta_{t+l}\) 就不一定能反映新策略下真实的 TD 残差。因此 PPO 仍然需要多轮与环境交互，持续收集新数据。

---

### 4. 概率比

对每一批数据，PPO 会做 \(k\) 次 minibatch 更新。由于数据是由旧策略 \(\pi_{\text{old}}\) 采集的，而当前要更新的是新策略 \(\pi_\theta\)，因此需要计算概率比：

\[
r_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
\]

它表示：

> 新旧 policy 对同一个 action 概率的相对变化。

如果：

\[
r_t(\theta)>1
\]

说明新策略比旧策略更倾向于选择该动作。

如果：

\[
r_t(\theta)<1
\]

说明新策略比旧策略更不倾向于选择该动作。

---

### 5. Clip Loss

PPO 的 clipped objective 可以写成：

\[
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\hat A_t
\right)
\right]
\]

训练时通常最大化该目标：

\[
\max_\theta L^{\text{CLIP}}(\theta)
\]

其中：

\[
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\]

表示把概率比限制在：

\[
[1-\epsilon,1+\epsilon]
\]

例如，当 \(\epsilon=0.2\) 时：

\[
r_t(\theta)\in[0.8,1.2]
\]

---

### 6. PPO 目标从哪里来

强化学习真正想最大化的是新策略下的期望回报：

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[R(\tau)\right]
\]

策略梯度定理给出：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}_{s_t,a_t\sim\pi_\theta}
\left[
\nabla_\theta
\log \pi_\theta(a_t|s_t)
A^{\pi_\theta}(s_t,a_t)
\right]
\]

也就是说：

- 当 \(A_t>0\) 时，说明该动作比平均水平好，希望提高它的概率；
- 当 \(A_t<0\) 时，说明该动作比平均水平差，希望降低它的概率。

但是 PPO 手里的数据来自旧策略：

\[
s_t,a_t\sim\pi_{\text{old}}
\]

而不是当前新策略：

\[
s_t,a_t\sim\pi_\theta
\]

所以需要用重要性采样思想，把新策略下的期望改写成旧策略数据上的期望。

固定状态 \(s_t\) 时，有：

\[
\mathbb{E}_{a_t\sim\pi_\theta}\left[f(a_t)\right]
=
\mathbb{E}_{a_t\sim\pi_{\text{old}}}
\left[
\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
f(a_t)
\right]
\]

令：

\[
f(a_t)=\hat A_t
\]

得到 PPO 的基础 surrogate objective：

\[
L^{\text{PG}}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right]
\]

其中：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
\]

这个目标可以理解为：

\[
\text{新策略对该动作的偏好变化}
\times
\text{该动作在旧数据中看起来有多好}
\]

---

### 7. 为什么需要 Clip

如果只使用基础目标：

\[
L^{\text{PG}}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right]
\]

那么当 \(\hat A_t>0\) 时，优化器会倾向于让：

\[
r_t(\theta)>1
\]

也就是提高该动作概率。

当 \(\hat A_t<0\) 时，优化器会倾向于让：

\[
r_t(\theta)<1
\]

也就是降低该动作概率。

问题是，某一次采样中 \(\hat A_t>0\)，并不代表这个 action 永远是好的。如果没有限制，策略可能会把某个动作的概率从很小一下子推得很大，导致 policy 更新过猛、训练不稳定。

因此 PPO 引入 clip：

\[
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\]

使得：

\[
r_t(\theta)\in[1-\epsilon,1+\epsilon]
\]

这样可以限制 policy 的更新幅度，防止：

\[
\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
\]

数值过于离谱。

外层取 \(\min\) 的作用是：当概率比变化过大时，不再给策略继续扩大变化的额外收益，从而稳定训练。

---

### 8. 关于 KL Divergence 的补充

有 clip 往往不能完全保证 \(\pi_\theta\) 和 \(\pi_{\text{old}}\) 的整体 KL divergence 一定很小。

原因是：

> clip 主要作用在当前采样到的 action 上，只能限制该 action 的概率比变化。

但是神经网络参数一旦更新，其他没有被当前样本直接约束的 action 分布也可能受到影响。

因此，即使某些 sampled action 的 ratio 被限制了，整体策略分布仍然可能发生较大变化，导致 KL divergence 变大。

也就是说：

\[
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\]

并不是一个严格的整体 KL 约束。

---

## 二、GRPO 算法

原始 PDF 第 3 页只有标题：

> GRPO 算法

后续内容待补充。
