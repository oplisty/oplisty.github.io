---

## title: "Diffusion-RL 复习（一）：PPO 与 GRPO"
date: 2026-05-22
excerpt: "复习 Diffusion-RL 相关基础知识，整理 PPO 与 GRPO 的核心思想、公式和训练流程。"
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

> 原始 PDF 中第 1 页为空白，第 2 页主要是 PPO 笔记，第 3 页只有“GRPO 算法”标题。本文将相关内容整理并合并为一篇适合 GitHub Pages / Jekyll 渲染的 Markdown 笔记。

## 目录

- [背景](#背景)
- [一、PPO 算法](#一ppo-算法)
  - [1. PPO 的目标](#1-ppo-的目标)
  - [2. 采样轨迹](#2-采样轨迹)
  - [3. 计算 Advantage](#3-计算-advantage)
    - [3.1 TD 残差](#31-td-残差)
    - [3.2 GAE Advantage](#32-gae-advantage)
  - [4. 概率比](#4-概率比)
  - [5. Clip Loss](#5-clip-loss)
  - [6. PPO 目标从哪里来](#6-ppo-目标从哪里来)
  - [7. 为什么需要 Clip](#7-为什么需要-clip)
  - [8. 关于 KL Divergence 的补充](#8-关于-kl-divergence-的补充)
- [二、GRPO 算法](#二grpo-算法)
  - [1. 从 PPO 的 Advantage 说起](#1-从-ppo-的-advantage-说起)
  - [2. GRPO 的核心思路](#2-grpo-的核心思路)
  - [3. 组内平均 Reward](#3-组内平均-reward)
  - [4. 组内归一化 Advantage](#4-组内归一化-advantage)
  - [5. 为什么还需要重要性采样](#5-为什么还需要重要性采样)
  - [6. GRPO 的 Clip Objective](#6-grpo-的-clip-objective)
  - [7. KL 惩罚项的作用](#7-kl-惩罚项的作用)
  - [8. GRPO 训练流程](#8-grpo-训练流程)
  - [9. PPO 与 GRPO 对比](#9-ppo-与-grpo-对比)
  - [10. 一句话总结](#10-一句话总结)

---

## 背景

鉴于新开始的项目，需要系统复习一下 Diffusion-RL 相关知识。本文先从强化学习中的基础 policy optimization 方法出发，整理 **PPO** 与 **GRPO** 的核心思想、关键公式和训练流程。

---

## 一、PPO 算法

### 1. PPO 的目标

PPO 的目标是让 policy 更新更加稳定。

更具体地说，PPO 通过限制每一步 policy 的更新幅度，防止 policy 在单次更新中发生过大突变，从而提升训练稳定性。

---

### 2. 采样轨迹

首先，旧策略 $\pi_{\text{old}}$ 与环境交互，收集 $T$ 步轨迹：

$$  
\tau = (s_0, a_0, r_0), (s_1, a_1, r_1), \cdots  
$$

其中：

- $s_t$：第 $t$ 步的状态；
- $a_t$：第 $t$ 步采取的动作；
- $r_t$：第 $t$ 步获得的奖励；
- $\pi_{\text{old}}$：采样数据时使用的旧策略。

---

### 3. 计算 Advantage

#### 3.1 TD 残差

先计算 TD 残差：

$$
\delta_t = Q(s_t, a_t) - V_\phi(s_t)
$$

在实际估计时，也常写成：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

它表示：当前 action 相比当前状态下平均水平到底好多少。

---

#### 3.2 GAE Advantage

使用 GAE（Generalized Advantage Estimation）估计 advantage：

$$
\hat A_t = \sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}
$$

其中：

- $\gamma$：折扣因子；
- $\lambda$：GAE 平滑系数；
- $\delta_{t+l}$：未来 timestep 的 TD 残差。

当 $\lambda \to 0$ 时：

$$
\hat A_t \approx \delta_t
$$

此时更接近单步 TD，具有 **低方差、高偏差** 的特点。

当 $\lambda \to 1$ 时：

$$
\hat A_t \approx \sum_{l=0}^{T-t}\gamma^l\delta_{t+l}
$$

此时更接近 Monte Carlo 估计，具有 **高方差、低偏差** 的特点。

需要注意的是，这里的 $\lambda$ 建模的是从当前 $t$ 开始，后续每个 timestep 的 TD 残差贡献。

如果 policy 更新之后，后续状态分布和动作分布都会发生变化，那么旧数据里的 $\delta_{t+l}$ 就不一定能反映新策略下真实的 TD 残差。因此 PPO 仍然需要多轮与环境交互，持续收集新数据。

---

### 4. 概率比

对每一批数据，PPO 会做 $k$ 次 minibatch 更新。由于数据是由旧策略 $\pi_{\text{old}}$ 采集的，而当前要更新的是新策略 $\pi_\theta$，因此需要计算概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

它表示：

> 新旧 policy 对同一个 action 概率的相对变化。

如果：

$$
r_t(\theta) > 1
$$

说明新策略比旧策略更倾向于选择该动作。

如果：

$$
r_t(\theta) < 1
$$

说明新策略比旧策略更不倾向于选择该动作。

---

### 5. Clip Loss

PPO 的 clipped objective 可以写成：

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t
\right)
\right]
$$

训练时通常最大化该目标：

$$
\max_\theta L^{\text{CLIP}}(\theta)
$$

其中：

$$
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

表示把概率比限制在：

$$
[1-\epsilon, 1+\epsilon]
$$

例如，当 $\epsilon = 0.2$ 时：

$$
r_t(\theta) \in [0.8, 1.2]
$$

---

### 6. PPO 目标从哪里来

强化学习真正想最大化的是新策略下的期望回报：

$$
J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[R(\tau)]
$$

策略梯度定理给出：

$$
\nabla_\theta J(\theta) =
\mathbb{E}*{s_t, a_t \sim \pi*\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
A^{\pi_\theta}(s_t, a_t)
\right]
$$

也就是说：

- 当 $A_t > 0$ 时，说明该动作比平均水平好，希望提高它的概率；
- 当 $A_t < 0$ 时，说明该动作比平均水平差，希望降低它的概率。

但是 PPO 手里的数据来自旧策略：

$$
s_t, a_t \sim \pi_{\text{old}}
$$

而不是当前新策略：

$$
s_t, a_t \sim \pi_\theta
$$

所以需要用重要性采样思想，把新策略下的期望改写成旧策略数据上的期望。

固定状态 $s_t$ 时，有：

$$
\mathbb{E}*{a_t \sim \pi*\theta}[f(a_t)] =
\mathbb{E}*{a_t \sim \pi*{\text{old}}}
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} f(a_t)
\right]
$$

令：

$$
f(a_t) = \hat A_t
$$

得到 PPO 的基础 surrogate objective：

$$
L^{\text{PG}}(\theta) = \mathbb{E}_t\left[r_t(\theta)\hat A_t\right]
$$

其中：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

这个目标可以理解为：

$$
\text{新策略对该动作的偏好变化}
\times
\text{该动作在旧数据中看起来有多好}
$$

---

### 7. 为什么需要 Clip

如果只使用基础目标：

$$
L^{\text{PG}}(\theta) = \mathbb{E}_t\left[r_t(\theta)\hat A_t\right]
$$

那么当 $\hat A_t > 0$ 时，优化器会倾向于让：

$$
r_t(\theta) > 1
$$

也就是提高该动作概率。

当 $\hat A_t < 0$ 时，优化器会倾向于让：

$$
r_t(\theta) < 1
$$

也就是降低该动作概率。

问题在于，某一次采样中 $\hat A_t > 0$，并不代表这个 action 永远是好的。如果没有限制，策略可能会把某个动作的概率从很小一下子推得很大，导致 policy 更新过猛、训练不稳定。

因此 PPO 引入 clip：

$$
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

使得：

$$
r_t(\theta) \in [1-\epsilon, 1+\epsilon]
$$

这样可以限制 policy 的更新幅度，防止

$$
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

数值过于离谱。

外层取 $\min$ 的作用是：当概率比变化过大时，不再给策略继续扩大变化的额外收益，从而稳定训练。

---

### 8. 关于 KL Divergence 的补充

有 clip 往往不能完全保证 $\pi_\theta$ 和 $\pi_{\text{old}}$ 的整体 KL divergence 一定很小。

原因是：

> clip 主要作用在当前采样到的 action 上，只能限制该 action 的概率比变化。

但是神经网络参数一旦更新，其他没有被当前样本直接约束的 action 分布也可能受到影响。

因此，即使某些 sampled action 的 ratio 被限制了，整体策略分布仍然可能发生较大变化，导致 KL divergence 变大。

也就是说：

$$
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

并不是一个严格的整体 KL 约束。

---

## 二、GRPO 算法

GRPO，全称 **Group Relative Policy Optimization**，可以看作是 PPO 在大语言模型场景中的一种重要变体。

它的核心思想是：

> 不训练额外的 value function，而是对同一个 prompt 生成多个回答，让这些回答在组内互相比较。

---

### 1. 从 PPO 的 Advantage 说起

在 PPO 中，计算 advantage 时通常需要 value function：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

一种近似写法是：

$$
A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中：

- $Q(s_t, a_t)$：在状态 $s_t$ 下采取动作 $a_t$ 后的期望回报；
- $V(s_t)$：状态 $s_t$ 的价值函数；
- $A_t$：该动作相对于当前状态平均水平的优势。

因此，PPO 通常需要训练一个 value function 或 critic：

$$
V_\phi(s)
$$

但对于 LLM 来说，额外训练一个 value model 的成本往往很高。

---

### 2. GRPO 的核心思路

GRPO 不再训练 $V(s)$。

它不是问：

> 这个答案比 value function 估计的平均水平好多少？

而是问：

> 这个答案比同一个 prompt 下的其他答案好多少？

也就是说，GRPO 对同一个问题生成多条回答，然后做组内比较。

假设同一个 prompt 为：

$$
q
$$

模型生成 $G$ 个回答：

$$
o_1, o_2, \dots, o_G
$$

然后对每个回答打分：

$$
r_1, r_2, \dots, r_G
$$

---

### 3. 组内平均 Reward

对于同一个 prompt 下的一组回答，先计算组内平均 reward：

$$
\bar r = \frac{1}{G}\sum_{i=1}^{G} r_i
$$

再计算 reward 的标准差：

$$
\sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \bar r)^2}
$$

---

### 4. 组内归一化 Advantage

GRPO 用组内归一化后的 reward 作为 advantage：

$$
A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}
$$

其中：

- $r_i$：第 $i$ 个回答的 reward；
- $\bar r$：同组回答的平均 reward；
- $\sigma_r$：同组 reward 的标准差；
- $\epsilon$：防止除零的小常数。

直观理解：

- 如果 $A_i > 0$，说明该回答比同组平均水平更好；
- 如果 $A_i < 0$，说明该回答比同组平均水平更差；
- 如果 $A_i \approx 0$，说明该回答接近同组平均水平。

---

### 5. 为什么还需要重要性采样

GRPO 和 PPO 类似，也会使用旧策略生成样本，再用这些样本更新当前策略。

因此需要重要性采样概率比：

$$
r_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}
$$

其中：

- $\pi_{\theta_{\text{old}}}$：生成回答时的旧策略；
- $\pi_\theta$：当前正在更新的新策略；
- $o_i$：第 $i$ 个回答；
- $q$：prompt。

基础优化目标可以理解为：

$$
\mathbb{E}[r_i(\theta) A_i]
$$

也就是：

> 如果某个回答的组内 advantage 为正，就提高它的概率；
>
> 如果某个回答的组内 advantage 为负，就降低它的概率。

---

### 6. GRPO 的 Clip Objective

和 PPO 一样，GRPO 也需要限制策略更新幅度。

因此目标函数中也会使用 clipped ratio：

$$
\min
\left(
r_i(\theta)A_i,
\operatorname{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)A_i
\right)
$$

完整目标可以写成：

## $$
J_{\text{GRPO}}(\theta) =
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\min
\left(
r_i(\theta)A_i,
\operatorname{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)A_i
\right)

\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})
\right]
$$

其中：

- $\epsilon$：clip 范围；
- $\beta$：KL 惩罚系数；
- $\pi_{\text{ref}}$：参考模型，通常是 SFT 模型或初始策略模型；
- $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$：当前策略和参考策略之间的 KL 散度。

---

### 7. KL 惩罚项的作用

目标中的 KL 项：

$$
-\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})
$$

主要有两个作用：

1. 限制当前策略不要偏离参考模型太远；
2. 防止模型在强化学习过程中出现灾难性遗忘。

如果没有 KL 约束，模型可能为了追求 reward 而破坏原本的语言能力、格式能力或安全对齐能力。

---

### 8. GRPO 训练流程

GRPO 的训练流程可以总结为：

1. **采样 prompt**
  从训练集中采一个问题：
   $$
   q
   $$
2. **生成多个回答**
  对同一个 prompt 生成 $G$ 个回答：
   $$
   o_1, o_2, \dots, o_G
   $$
3. **给每个回答打分**
  使用 reward model、规则验证器或人工偏好模型得到：
   $$
   r_1, r_2, \dots, r_G
   $$
4. **组内归一化 advantage**
  计算：
   $$
   A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}
   $$
5. **更新 policy**
  使用 GRPO objective 更新策略：
   $$
   J_{\text{GRPO}}(\theta)
   $$

---

### 9. PPO 与 GRPO 对比


| 项目                  | PPO                          | GRPO                 |
| ------------------- | ---------------------------- | -------------------- |
| Advantage 来源        | $Q(s_t, a_t) - V(s_t)$ 或 GAE | 组内 reward 相对值        |
| 是否需要 value function | 需要                           | 不需要                  |
| Baseline            | $V(s)$                       | 同组平均 reward $\bar r$ |
| 样本形式                | 环境轨迹                         | 同一 prompt 下的多个回答     |
| 稳定机制                | ratio clip + KL 监控或惩罚        | ratio clip + KL 惩罚   |
| 适合场景                | 通用 RL / RLHF                 | LLM 推理、数学、代码等可验证任务   |


---

### 10. 一句话总结

GRPO 可以理解为：

$$
\text{GRPO} = \text{PPO} - \text{Value Model} + \text{Group Relative Advantage}
$$

也就是说：

> GRPO 用同一个 prompt 下多个回答的组内相对 reward 来构造 advantage，从而省掉 value function，同时仍然保留 PPO 的 probability ratio、clip 和 KL 惩罚来保证训练稳定。

