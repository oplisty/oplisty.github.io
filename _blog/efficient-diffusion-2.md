---
title: "Efficient Diffusion（二）：Progressive Distillation 与蒸馏起点"
date: 2026-05-25
excerpt: "深入讲解 Progressive Distillation 的核心思想、v-prediction 参数化，以及为什么 2-to-1 蒸馏设计是保守但稳定的选择。"
cover: "/images/Efficient_diffusion.jpg"
categories:
  - Diffusion
tags:
  - Diffusion
  - Efficient-Diffusion
  - Few-Step
  - Progressive-Distillation
  - v-prediction
  - DDIM
math: true
read_time: true
---

书接上文，讲完了传统加速方法依赖采样器和数学优化算法，本质上是使得每一步去噪过程紧贴高维曲线而进行有限的加速。本文将着重讲解 22-23 年两年，也可以说是 Efficient Diffusion 的蒸馏起点的重要技术。

## 目录

- [Progressive Distillation](#progressive-distillation)
  - [v-prediction 参数化](#v-prediction-参数化)
  - [为什么不直接做 k-to-1 蒸馏？](#为什么不直接做-k-to-1-蒸馏)

## Progressive Distillation

承接上文的传统加速方法，本节进入「蒸馏时代」的开篇之作 —— Progressive Distillation（Salimans & Ho, 2022）。

**核心思想**：让学生模型用 1 步去近似教师模型 2 步的输出，然后让训练好的学生成为新的教师，如此递归。每次迭代将采样步数减半，经过 $K$ 轮后，原本需要 $N$ 步的采样可压缩到 $N/2^K$ 步。

**算法流程**

设教师模型为 \(\hat{x}_{\eta}(z_t)\)，学生模型为 \(\hat{x}_{\theta}(z_t)\)，初始化 \(\theta \leftarrow \eta\)。每轮蒸馏的核心步骤为：

1. 采样 $t \sim \mathcal{U}[0, 1]$，对干净样本 $x$ 加噪得 $z_t = \alpha_t x + \sigma_t \epsilon$
2. 用教师执行 2 步 DDIM 去噪：

$$
z_{t'} = \alpha_{t'} \hat{x}_\eta(z_t) + \frac{\sigma_{t'}}{\sigma_t}\bigl(z_t - \alpha_t \hat{x}_\eta(z_t)\bigr)
$$

$$
z_{t''} = \alpha_{t''} \hat{x}_\eta(z_{t'}) + \frac{\sigma_{t''}}{\sigma_{t'}}\bigl(z_{t'} - \alpha_{t'} \hat{x}_\eta(z_{t'})\bigr)
$$

3. 反推学生应当对齐的目标 $\tilde{x}$，使得学生单步从 $z_t$ 出发到 $z_{t''}$ 等价于教师两步从 $z_t$ 出发到 $z_{t''}$：

$$
\tilde{x} = \frac{z_{t''} - (\sigma_{t''}/\sigma_t)\, z_t}{\alpha_{t''} - (\sigma_{t''}/\sigma_t)\,\alpha_t}
$$

4. 优化损失

$$
\mathcal{L} = w(\lambda_t) \bigl\| \tilde{x} - \hat{x}_\theta(z_t) \bigr\|^2
$$

完成一轮训练后，令 $\eta \leftarrow \theta$，目标步数减半，进入下一轮。从 8192 步开始，经过 13 轮蒸馏即可压到 1 步。

从上面的算法流程我们也可以看出来，其实 PD 强依赖 DDIM 这类确定性采样器。PD 的核心是：教师从同一个 $z_t$ 出发执行两步，得到一个确定的 $z_{t''}$，然后反推出一个学生单步应该预测的 $\tilde{x}$。这个过程要求教师轨迹是确定的，否则同一个 $z_t$ 可能对应多个随机终点，单个 MSE 目标就不再能准确表达教师的完整分布。

### v-prediction 参数化

Progressive Distillation 的另一项重要贡献，是引入了 $v$-prediction 参数化。前向加噪过程写作：

$$
z_t = \alpha_t x + \sigma_t \epsilon,
$$

其中 $x$ 是干净样本，$\epsilon \sim \mathcal{N}(0, I)$ 是高斯噪声，$\alpha_t$ 和 $\sigma_t$ 分别控制信号与噪声的比例。传统扩散模型通常让网络预测 $\epsilon$，也就是直接预测噪声。而 PD 使用的是另一种参数化：

$$
v_t \equiv \alpha_t \epsilon - \sigma_t x.
$$

为什么这么写呢？这个 $v$ 可以理解成扩散轨迹上的"速度"。如果采用常见的三角形式：

$$
\alpha_t = \cos\phi_t, \qquad \sigma_t = \sin\phi_t,
$$

那么前向过程可以写成：

$$
z_{\phi} = \cos\phi \cdot x + \sin\phi \cdot \epsilon.
$$

也就是说，随着 $\phi$ 从 $0$ 增大到 $\pi/2$，样本 $z_\phi$ 会从干净图像 $x$ 逐渐旋转到纯噪声 $\epsilon$。对这个轨迹求导：

$$
\frac{d z_\phi}{d\phi} = -\sin\phi \cdot x + \cos\phi \cdot \epsilon = \alpha_\phi \epsilon - \sigma_\phi x.
$$

这正是 $v_\phi$。因此，$v$-prediction 并不是引入了新的信息，而是把模型输出从"预测噪声"或"预测图像"改成了"预测当前点沿扩散轨迹移动的方向"。也可以理解为 Flow-matching 的思想，不过比较原始。它和 $x, \epsilon$ 之间可以互相转换。由

$$
z_t = \alpha_t x + \sigma_t \epsilon,
$$

$$
v_t = \alpha_t \epsilon - \sigma_t x,
$$

可以反解得到：

$$
x = \alpha_t z_t - \sigma_t v_t,
$$

$$
\epsilon = \sigma_t z_t + \alpha_t v_t.
$$

这个转换非常关键，因为它没有出现除以 $\alpha_t$ 或除以 $\sigma_t$ 的操作。相比之下，如果网络预测的是 $\epsilon$，那么需要通过

$$
\hat{x}_\theta(z_t) = \frac{z_t - \sigma_t \hat{\epsilon}_\theta(z_t)}{\alpha_t}
$$

恢复干净样本。当 $t \to 1$ 时，输入接近纯噪声，$\alpha_t \to 0$，这个公式会变得非常敏感：哪怕 $\hat{\epsilon}_\theta$ 只有很小的误差，除以接近 $0$ 的 $\alpha_t$ 后，也会被放大成很大的 $\hat{x}$ 误差。原本使用很多采样步时，高噪声区域的一点误差还可以在后续许多步中逐渐修正；但当采样步数被压缩到十几步、几步甚至一步时，模型必须在极低 SNR 区域直接给出有效的生成方向。这时 $\epsilon$-prediction 的问题就会被放大：它在高噪声端主要学会"这个输入里的噪声是什么"，但这并不等价于学会"这团噪声应该生成什么图像"。

而 $v$-prediction 在两个端点上的行为更加自然。首先，当 $t \to 0$ 时，$\alpha_t \to 1,\ \sigma_t \to 0$，此时

$$
v_t = \alpha_t \epsilon - \sigma_t x \approx \epsilon.
$$

也就是说，在低噪声区域，$v$-prediction 近似退化成 $\epsilon$-prediction：输入已经接近干净图像，模型只需要预测残余噪声方向即可。

其次，当 $t \to 1$ 时，$\alpha_t \to 0,\ \sigma_t \to 1$，此时

$$
v_t = \alpha_t \epsilon - \sigma_t x \approx -x.
$$

也就是说，在高噪声区域，$v$-prediction 又近似变成了预测负的干净图像。换句话说，当输入几乎是纯噪声时，模型不会只是在回答"噪声是什么"，而是在直接学习"这团噪声应该对应到什么生成结果"。因此，$v$-prediction 的直觉可以概括为：

$$
t \to 0:\quad v_t \approx \epsilon,
$$

$$
t \to 1:\quad v_t \approx -x.
$$

低噪声时，它像是在预测噪声；高噪声时，它像是在预测图像；中间区域则平滑过渡。这种参数化正好适合少步采样，因为少步模型必须一次跨越更大的时间区间，不能再依赖许多微小步骤逐渐修正。从误差传播角度，若使用 $\epsilon$-prediction，并且噪声预测存在误差

$$
\hat{\epsilon} = \epsilon + \Delta\epsilon,
$$

则恢复出的 $x$ 误差为：

$$
\Delta x = -\frac{\sigma_t}{\alpha_t} \Delta\epsilon.
$$

当 $t \to 1$ 时，$\alpha_t \to 0$，所以误差会被严重放大。而对于 $v$-prediction，恢复公式是：

$$
\hat{x} = \alpha_t z_t - \sigma_t \hat{v}.
$$

若 $\hat{v} = v + \Delta v$，则 $\Delta x = -\sigma_t \Delta v$，由于 $0 \leq \sigma_t \leq 1$，预测误差不会因为低 SNR 而爆炸。这就是 $v$-prediction 在少步采样中更稳定的核心原因。它还带来了一个更漂亮的 DDIM 更新形式。在角度参数 $\phi$ 下，确定性 DDIM 的一步更新可以写成：

$$
z_{\phi-\delta} = \cos\delta \cdot z_\phi - \sin\delta \cdot \hat{v}_\theta(z_\phi).
$$

也就是说，DDIM 去噪可以被理解为：当前点 $z_\phi$ 沿着模型预测的速度方向 $\hat{v}_\theta$ 旋转一小段角度 $\delta$。

其实 $x$-prediction 在 PD 中也可以考虑用。严格来说，在普通去噪训练里，当 $t \to 1$ 时，输入 $z_t$ 几乎是纯噪声，与原始训练图像 $x$ 的信息联系极弱，此时直接预测 $x$ 的任务确实很难。但在 Progressive Distillation 中，学生学习的目标不是随机配对的原始图像，而是教师模型从同一个噪声点出发经过若干步 DDIM 得到的确定性结果。因此 $x$-prediction 在 PD 中仍然可以作为监督目标。只是相比之下，$v$-prediction 同时具备更稳定的端点行为、更合理的误差传播形式，以及与 DDIM 更新更自然的几何关系，所以成为后续许多少步扩散模型常用的参数化方式。

---

### 为什么不直接做 $k$-to-1 蒸馏？

理解了 $v$-prediction 后，我们再回到 Progressive Distillation 的另一个自然问题：为什么它只做「教师 2 步 $\rightarrow$ 学生 1 步」？如果让教师执行 $k$ 步，然后让学生 1 步对齐，不就可以把轮数从 $\log_2 N$ 降到 $\log_k N$ 吗？从形式上看，这当然可以写出来。设教师从 $z_t$ 出发连续执行 $k$ 步 DDIM，最终到达 $z_s$，那么同样可以反推出一个学生的单步目标：

$$
\tilde{x}^{(k)} = \frac{z_s - (\sigma_s/\sigma_t) z_t}{\alpha_s - (\sigma_s/\sigma_t)\alpha_t}.
$$

也就是说，学生只要在 $z_t$ 上预测出 $\tilde{x}^{(k)}$，它的一步 DDIM 更新就可以直接落到教师 $k$ 步之后的位置。问题不在于代数上不能做，而在于这个目标会随着 $k$ 增大迅速变得更难学习。

首先，$k$ 越大，教师轨迹跨越的时间区间越长。扩散模型的去噪过程本质上是在近似一个随时间变化的非线性流场。小步长时，局部近似比较可靠；但大步长时，学生需要用一次预测拟合一段较长的弯曲轨迹。此时 $\tilde{x}^{(k)}$ 不再是一个温和的局部目标，而可能偏离原始数据流形更远，表现出更高的方差和更强的时间依赖性。这样一来，学生学到的就不是"局部去噪方向"，而是一个更复杂的长程跳跃映射。

其次，蒸馏误差会更容易被放大。2-to-1 蒸馏只要求学生模仿教师相邻两步的结果，目标仍然比较局部；即使教师本身存在离散化误差，这个误差也相对可控。但如果直接做 $k$-to-1，教师的 $k$ 次预测误差会先在轨迹中累积，再被压缩成一个单步监督信号。训练好的学生进入下一轮后，这种偏差又会继续作为新教师被继承，导致后续蒸馏更容易退化。

而且从计算量上看，$k$ 步蒸馏也没有想象中那么划算。每个训练样本需要调用教师 $k$ 次，而总轮数是 $\log_k N$，因此教师前向计算量近似正比于

$$
k\log_k N = \frac{k}{\ln k}\ln N.
$$

这个量在 $k = 2 \sim 4$ 附近差别并不大。比如 $k = 2$ 的系数约为 $2.89$，$k = 3$ 约为 $2.73$，$k = 4$ 又回到约 $2.89$，而 $k = 8$ 反而变成约 $3.85$。也就是说，增大 $k$ 并不能显著减少教师计算量，却会明显增加监督目标的难度和训练不稳定性。

因此，Progressive Distillation 的 2-to-1 设计并不是因为多步蒸馏在公式上不可行，而是因为它把每次蒸馏控制在一个足够局部、足够稳定的区间内。每一轮只把步长放大 2 倍，学生要学习的映射不会突然变得过于困难；同时经过多轮递归，采样步数仍然可以指数级下降。换句话说，PD 选择的是一种"保守但稳定"的路径：牺牲更多轮训练，换取每一轮目标的可靠性。

这也解释了后续方法的发展方向：许多新的少步生成方法并不是简单地把 PD 中的 $2$-to-$1$ 改成 $k$-to-$1$，而是重新设计监督信号。它们的核心目标，都是绕开朴素多步蒸馏中"长程目标难学、误差累积严重、确定性轨迹依赖强"的问题。

明天我们将来看 2023 年如日中天的 Consistency Model。
