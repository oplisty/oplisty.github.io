---
title: "Diffusion-RL 复习（二）：Diffusion RL 的困难与现有解决方法"
date: 2026-05-22
excerpt: "整理 Diffusion RL 中概率计算困难的来源，以及当前主流方法如何通过逐步转移建模来绕开这一问题。"
cover: "/images/3.jpg"
categories:
  - Reinforcement-Learning
  - Diffusion
tags:
  - Diffusion-RL
  - Importance-Sampling
  - DDPM
  - DDIM
  - ODE-Solver
  - SDE-Solver
math: true
read_time: true
---

在复习完 DPO、PPO 和 GRPO 等强化学习算法后，可以进一步分析 **Diffusion RL** 的核心难点。本文重点讨论：为什么在 LLM 中常见的强化学习做法很难直接迁移到 diffusion model 上，以及现有工作通常如何绕开这个问题。

## 目录

- [背景](#背景)
- [一、问题的出发点：重要性采样](#一问题的出发点重要性采样)
- [二、为什么 LLM 中这件事比较容易](#二为什么-llm-中这件事比较容易)
- [三、为什么 Diffusion 中这件事很难](#三为什么-diffusion-中这件事很难)
- [四、现有方法的常见思路：把去噪过程离散化](#四现有方法的常见思路把去噪过程离散化)
- [五、如何让每一步转移概率可计算](#五如何让每一步转移概率可计算)
- [六、这种方法的适用范围与局限](#六这种方法的适用范围与局限)
- [七、小结](#七小结)
- [八、后续方向](#八后续方向)

---

## 背景

在 PPO 或 GRPO 这类强化学习算法中，由于其本质上都涉及 **旧策略采样** 或 off-policy 风格的更新，因此都绕不开 **重要性采样**。

一个最核心的问题是：

> 如何计算当前策略和旧策略下，对同一个动作的概率比？

这件事在 LLM 中相对自然，但在 diffusion model 中会变得困难得多。

---

## 一、问题的出发点：重要性采样

无论是 PPO 还是 GRPO，都需要类似如下的概率比：

$$
r(\theta) = \frac{\pi_\theta(a \mid s)}{\pi_{\text{old}}(a \mid s)}
$$

也就是说，关键问题在于：**如何计算当前策略和旧策略下同一个动作的概率比**。

如果这个量可以稳定计算，那么很多策略优化方法都可以继续使用；如果这个量难以获得，整个 RL 更新就会遇到根本性障碍。

---

## 二、为什么 LLM 中这件事比较容易

在 LLM 中，可以把生成过程表示为一个自回归决策过程。

定义：

$$
s_t = (\text{prompt}, y_{<t})
$$

其中：

- $s_t$：当前状态，由 prompt 和历史已经输出的 tokens 组成；
- $a_t$：当前动作，即 next token；
- $y_t$：第 $t$ 步生成的 token。

LLM 的生成概率可以自然分解为：

$$
\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid x, y_{<t})
$$

因此，我们可以拿到每个 token 的条件概率，并进一步计算整段生成过程对应的概率，或者逐 token 地计算重要性采样比率。

LLM 中这样做成立的原因是：生成过程中的中间变量 $y_t$ 会共同构成最终结果 $y$，每个 timestep 的 token 都是可观测的，因此联合概率可以通过逐步累乘得到。

换句话说，在 LLM 中：

- 中间决策是可见的；
- 每一步的条件概率是显式可计算的；
- 最终输出与中间过程天然一致。

所以在 LLM 里，概率比这件事通常是“能算”的。

---

## 三、为什么 Diffusion 中这件事很难

与 LLM 不同，Diffusion 的中间去噪步骤 $x_t$ 通常是隐藏变量，而不是最终输出的一部分。

在 Diffusion 中：

- condition $c$ 类似于输入条件；
- 最终动作 $a$ 可以理解为输出的 image / video；
- 中间去噪路径 $x_{1:T}$ 并不会直接作为最终结果暴露出来。

如果类比 LLM，尝试对反向扩散路径进行联合概率分解，可以得到：

$$
p_\theta(x_{0:T} \mid c) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t, c)
$$

但这并不等于最终样本的边缘概率：

$$
p_\theta(x_0 \mid c) = \int p_\theta(x_{0:T} \mid c)  dx_{1:T}
$$

因此，我们实际容易拿到的是整条去噪路径的联合概率 $p_\theta(x_{0:T} \mid c)$，而不是最终生成结果的边缘概率 $p_\theta(x_0 \mid c)$。

而在 RL 里，我们真正关心的通常更接近“最终动作在当前状态下的概率”，也就是类似 $p(a \mid s)$ 的量。

问题在于，$x_{1:T}$ 是高维连续变量，对其积分通常无法精确计算。这就是 Diffusion RL 中计算 $p(a \mid s)$ 的核心困难。

---

## 四、现有方法的常见思路：把去噪过程离散化

近期工作通常将反向采样过程离散成很多步，并将每一步看作强化学习中的一次状态转移。

定义：

$$
s_t = (x_t, t, c)
$$

$$
a_t = x_{t-1}
$$

也就是说：

- 当前状态是当前噪声图像或视频 latent $x_t$、时间步 $t$ 和条件 $c$；
- 当前动作是下一步去噪结果 $x_{t-1}$。

此时，重要性采样比率可以写成每一步转移概率的比值：

$$
r_t = \frac{p_\theta(x_{t-1} \mid x_t, c)}{p_{\text{old}}(x_{t-1} \mid x_t, c)}
$$

这样做的核心思想是：

> 不直接去算最终结果 $x_0$ 的边缘概率，而是转而去算每一步局部转移的条件概率。

只要每一步局部转移概率是可计算的，就能构造逐步的重要性采样比率，并进一步进行 RL 优化。

---

## 五、如何让每一步转移概率可计算

为了让上述概率可计算，很多方法会将每一步反向转移建模为 Gaussian：

$$
p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t, c), \sigma_t^2 I\big)
$$

这样，每一步的 transition probability 就可以被解析地计算出来。

这种建模之所以成立，和 DDPM 的假设有关。DDPM 中每一步转移独立，并且带有 Gaussian noise：

$$
x_{t-1} = \mu_\theta(x_t, t, c) + \sigma_t \epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I)
$$

从这个角度看，Diffusion model 预测的本质可以理解为 Gaussian 分布的均值，或者用于确定该均值的参数。

因此，一旦采用这种 Gaussian transition 的建模方式，我们就能把 RL 所需的概率比写成显式可计算的形式。

---

## 六、这种方法的适用范围与局限

上述概率建模依赖于“每一步随机转移服从 Gaussian”的假设，因此它天然更适用于：

- DDPM；
- SDE solver。

但它不适用于，或很难直接适用于：

- ODE solver；
- DDIM；
- flow-matching ODE；
- 其他确定性或高阶求解器。

原因是这些求解器不一定具有显式的逐步 Gaussian transition probability，因此无法直接计算：

$$
p_\theta(x_{t-1} \mid x_t, c)
$$

如果训练时使用一种概率假设，而推理时换用不匹配的求解器，就可能引入 **前向-反向过程的不一致性**。

也就是说，虽然“逐步 Gaussian transition + RL ratio”这条路线是当前很自然的一种做法，但它的适用范围并不是无条件成立的。

---

## 接下来我们会讲解 DiffusionNFT 是怎么解决的

