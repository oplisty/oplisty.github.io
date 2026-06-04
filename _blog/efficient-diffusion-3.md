pin jin

## title: "Efficient Diffusion（三）：Consistency Model 与 Distribution Matching Distillation"

date: 2026-05-28
excerpt: "深入讲解 Consistency Model 的自洽性原理与两种训练范式（CD/CT），以及 DMD 如何从分布匹配视角绕开轨迹蒸馏的固有困难。"
cover: "/images/Efficient_diffusion.jpg"
categories:

- Diffusion
tags:
- Diffusion
- Efficient-Diffusion
- Few-Step
- Consistency-Model
- DMD
- Distribution-Matching
math: true
read_time: true

书接上文，Progressive Distillation 通过多轮 2-to-1 蒸馏将采样步数指数级压缩，但它仍然需要反复调用教师模型，训练成本不低。2023 年，两个重要的新方向出现了：Consistency Model 提出用"自洽性"替代逐步蒸馏，而 DMD 则从分布匹配的角度彻底跳出了轨迹蒸馏的框架。本文将分别讲解这两种方法的核心思想。

## 目录

- [Consistency Model](#consistency-model)
  - [PF-ODE 与一致性函数](#pf-ode-与一致性函数)
  - [Consistency Distillation](#consistency-distillation-cd)
  - [Consistency Training](#consistency-training-ct)
  - [采样与多步精炼](#采样与多步精炼)
- [Distribution Matching Distillation](#distribution-matching-distillation-dmd)
  - [从轨迹匹配到分布匹配](#从轨迹匹配到分布匹配)
  - [KL 散度的梯度分解](#kl-散度的梯度分解)
  - [训练流程](#训练流程)
  - [与 GAN 和 Score Distillation 的联系](#与-gan-和-score-distillation-的联系)

## Consistency Model

回顾一下扩散模型的 PF-ODE。在连续时间框架下，扩散模型的逆过程可以由一个确定性微分方程描述。以 EDM 参数化为例，PF-ODE 可以写作：

$$
\frac{dx_t}{dt} = \frac{x_t - D_\phi(x_t, t)}{t},
$$

其中 $D_\phi$ 是预训练的 denoiser 网络，输入带噪样本 $x_t$ 和时间 $t$，输出对干净样本 $x_0$ 的预测。从 $x_T \sim \mathcal{N}(0, T^2 I)$ 出发，沿 PF-ODE 积分到底，就能到达数据分布 $p_{\text{data}}$。

Progressive Distillation 的做法是：用教师模型在 PF-ODE 上走两步，让学生一步复现，反复递归。这个思路自然引出一个问题：**能不能直接学一个映射，把 PF-ODE 轨迹上任意一点一次性映射到终点？**

这就是 Consistency Model（Song et al., 2023）的核心想法。

### PF-ODE 与一致性函数

定义**一致性函数** $f(x_t, t)$：

$$
f(x_t, t) \mapsto x_0,
$$

它将 PF-ODE 轨迹上任意一点 $x_t$ 直接映射为对应的干净样本 $x_0$。这个函数需要满足两个条件：

1. **边界条件**：当 $t = 0$ 时，输入已经是干净样本，函数应该退化为恒等映射

$$
f(x_0, 0) = x_0.
$$

1. **自洽性条件**：对于同一条 PF-ODE 轨迹上的任意两点 $(x_t, t)$ 和 $(x_{t'}, t')$，它们的映射结果应该相同

$$
f(x_t, t) = f(x_{t'}, t'), \quad \forall\ t, t' \in [0, T].
$$

边界条件可以通过参数化来保证。一种常见的做法是令

$$
f_\theta(x_t, t) = c_{\text{skip}}(t) \cdot x_t + c_{\text{out}}(t) \cdot F_\theta(x_t, t),
$$

其中 $c_{\text{skip}}(0) = 1$, $c_{\text{out}}(0) = 0$，从而在 $t = 0$ 时自然满足 $f_\theta(x_0, 0) = x_0$。

自洽性条件则是训练的核心约束。如果模型对同一轨迹上不同时刻的输入给出相同的输出，那就意味着它学会了整个 PF-ODE 的积分结果，从任意中间状态都能"一步到位"。这也是一致性模型得名的原因：模型输出在整条轨迹上保持**一致**。

### Consistency Distillation (CD)

CD 的目标是利用预训练扩散模型 $D_\phi$ 作为 Teacher，训练一个一致性模型 $f_\theta$。

**时间离散化**

PF-ODE 在连续时间 $[\sigma_{\min}, \sigma_{\max}]$ 上演化。训练时无法处理连续无限个时间点，需要将其离散化为 $N$ 个噪声水平：

$$
\sigma_{\min} = t_0 < t_1 < t_2 < \dots < t_N = \sigma_{\max}.
$$

这个 $N$ 是 CD 的核心超参数之一：它决定了我们将连续 PF-ODE 切成多少段。训练时，每一步随机采样一对相邻的时间点 $(t_n, t_{n+1})$，让模型学习从 $t_{n+1}$ 到 $t_n$ 的一致性。其中$N$ 不能取太大也不能取太小,太小的话相邻时间步间隔过大，前面说过Euler近似的本质还是泰勒展开拟合,单步 Euler 近似的误差会显著增加，蒸馏精度受损。$N$ 太大的话训练成本线性上升，且相邻点的差异变得极小，自洽性信号变弱(一般取$N = 18$ 或 $N = 40$ )时间步的分布通常采用 EDM 的 $\rho$-调度策略，在低噪声区域（$t$ 较小）放置更多离散点,因为数据流形附近的细节对最终生成质量影响最大，需要在低噪声端做更精细的分辨。

**算法流程**

给定预训练 Teacher $D_\phi$ 和学生一致性模型 $f_\theta$（维护其 EMA 副本 $f_{\theta^-}$，初始化 $\theta^- \leftarrow \theta$），每步训练的核心步骤为：

1. 采样干净样本 $x_0 \sim p_{\text{data}}$，噪声 $\varepsilon \sim \mathcal{N}(0, I)$，以及离散时间索引 $n \sim \text{Uniform}[0, N-1]$。
2. 在 $t_n$ 处构造带噪样本：

$$
x_{t_n} = x_0 + t_n \varepsilon.
$$

1. 用 Teacher 沿 PF-ODE 做一步 Euler 推进，得到轨迹上相邻的下一个点 $x_{t_{n+1}}$：

$$
x_{t_{n+1}} = x_{t_n} + (t_{n+1} - t_n) \cdot \frac{x_{t_n} - D_\phi(x_{t_n}, t_n)}{t_n}.
$$

确保 $x_{t_n}$ 和 $x_{t_{n+1}}$ 落在同一条 PF-ODE 轨迹上。取相邻离散点主要就是为了Euler 近似的局部误差可控。

1. 对 $(x_{t_n}, t_n)$ 和 $(x_{t_{n+1}}, t_{n+1})$ 这对轨迹上的相邻点施加自洽性约束，计算损失并更新 $\theta$：

$$
\mathcal{L}*{\text{CD}}(\theta) = \mathbb{E}*{x_0, n, \varepsilon} \Bigl[ d\bigl( f_\theta(x_{t_n}, t_n), f_{\theta^-}(x_{t_{n+1}}, t_{n+1}) \bigr) \Bigr],
$$

其中 $d(\cdot, \cdot)$ 是距离度量，采用 Pseudo-Huber 损失：

$$  
d(x, y) = \sqrt{x - y^2 + c^2} - c,  
$$

其中 $c$ 是平滑参数。训练初期 $x - y$ 较大时，Pseudo-Huber 近似 L1 损失，梯度更平滑、不易被离群点主导；随着训练进展差异缩小，它逐渐逼近 L2，收敛更精细。在图像任务上还可进一步结合 LPIPS 等感知损失来提升视觉质量。

更新参数时源端 $f_\theta$ 接收梯度、正常更新,目标端 $f_{\theta^-}$ 通过 EMA 缓慢追随 $\theta$：

$$
\theta^- \leftarrow \mu \theta^- + (1 - \mu) \theta, \qquad \mu \approx 0.999.
$$

所以用来计算"正确答案"的 $f_{\theta^-}$ 实际上是一个旧版本的模型,这种蒸馏方法其实是 off-policy 的。如果两端都用当前的 $\theta$ 并同时更新，模型可以通过两端同步漂移来"作弊"地降低损失——就像两个人约定好一起往前走，相对距离始终为零，实际上谁也没有真的到达终点。从 RL 的角度看，这与 DQN 的 target network 完全一致：用旧参数计算 TD target，避免当前网络的自举目标随自身参数震荡。CM 本质上将 PF-ODE 的积分转化为一个时序差分学习问题，EMA 目标网络则是保证这个 TD 过程收敛的关键。

然而这**损失只约束自洽性，没有任何显式的监督告诉模型"正确答案"是什么，万一 $f_{\theta^-}$ 和 $f_\theta$ 都预测错了怎么办？** 它们完全可以收敛到某个一致但错误的函数，比如对任何输入都输出一个固定的灰色图像——自洽性完美满足，但毫无用处。**边界条件**是整个训练过程中**唯一的绝对锚点**。而这也成了这种方法很大的局限性。

### Consistency Training (CT)

CD 的问题在于依赖一个预训练好的 Teacher。Consistency Model 的另一个重要贡献是 Consistency Training（CT）：**不依赖教师模型，从零开始训练一致性模型**。

CT 的核心洞察很巧妙。CD 用 Teacher 做一步 Euler 更新来获得 PF-ODE 上的相邻点对 $(x_{t_n}, x_{t_{n+1}})$。但如果我们已经有干净样本 $x_0$，那么 $x_{t_n}$ 和 $x_{t_{n+1}}$ 本就落在同一条 PF-ODE 轨迹上——因为它们是用**同一个噪声** $\varepsilon$ 和同一个 $x_0$ 构造出来的：

$$
x_{t_n} = x_0 + t_n \varepsilon, \qquad x_{t_{n+1}} = x_0 + t_{n+1} \varepsilon.
$$

也就是说，$x_{t_n}$ 和 $x_{t_{n+1}}$ 天然来自同一条轨迹的两个时刻，它们应当被映射到同一个 $x_0$。因此 CT 的损失可以定义为：

$$
\mathcal{L}*{\text{CT}}(\theta) = \mathbb{E}*{x_0, n, \varepsilon} \Bigl[ d\bigl( f_\theta(x_0 + t_{n+1} \varepsilon, t_{n+1}), f_{\theta^-}(x_0 + t_n \varepsilon, t_n) \bigr) \Bigr].
$$

与 CD 的区别仅在于相邻点的构造方式——不需要 Teacher 做 PF-ODE 步，直接用噪声注入，因此是一个**无偏估计**。但代价是方差更大：Teacher 的 Euler 步提供了 PF-ODE 的确定性信息，是低方差的；而直接噪声注入隐含了 PF-ODE 的假设（同一噪声意味着同一轨迹），其随机性更高。Consistency Model 也支持**多步精炼**：先从 $x_T$ 得到 $x_0$，再加噪声回到某个中间时间步 $t'$，再做一次一致性映射。每次精炼可以修复前一步的残留误差。经过 2-4 步精炼后，Consistency Model 通常能达到与原始多步扩散模型非常接近的质量。

但是就实践来看CT 由于高方差的特性, 训练稳定性较差，对超参数敏感而且自洽性条件的局部性（只约束相邻时间步）也意味着远端的一致性是通过链式传播间接建立的，误差仍然会在长链上累积。

---

## Distribution Matching Distillation (DMD)

### 从轨迹匹配到分布匹配

无论是 Progressive Distillation 还是 Consistency Model，核心思路都是**轨迹匹配**——让学生在 PF-ODE 的几何结构上模仿教师的行为。PD 让学生对齐教师的两步结果，CM 要求学生维护 PF-ODE 上的自洽性。

但回到问题的本质：我们为什么需要教师？因为我们想要学生生成与教师"一样好"的样本。这里的"一样好"是指两个**分布**接近，而不是两条**轨迹**接近。如果学生的分布已经和教师一致，那么中间走什么路径其实无关紧要。

DMD（Distribution Matching Distillation, Yin et al., 2024）正是从这一角度出发的方法。它直接最小化学生分布 $p_{\text{fake}}$ 与教师分布 $p_{\text{real}}$ 之间的 KL 散度：

$$
\mathcal{L}*{\text{DMD}} = D*{\text{KL}}(p_{\text{fake}}｜p_{\text{real}}).  
$$

这里 $p_{\text{fake}}$ 是学生生成器 $G_\theta$ 将噪声 $z \sim \mathcal{N}(0, I)$ 映射为数据 $x = G_\theta(z)$ 后得到的分布，$p_{\text{real}}$ 是教师扩散模型经过完整多步采样后产生的分布（在实践中通常近似为目标数据分布 $p_{\text{data}}$）。

选择 $D_{\text{KL}}(p_{\text{fake}} ｜p_{\text{real}})$（reverse KL）而非 $D_{\text{KL}}(p_{\text{real}}  ｜p_{\text{fake}})$（forward KL）是有考虑的。Reverse KL 是 mode-seeking：它倾向于让 $p_{\text{fake}}$ 集中在 $p_{\text{real}}$ 的高概率区域，即使这会错过一些 mode。对于生成任务，这意味着学生生成的样本更可能是"典型"的、高质量的，而不是为了覆盖所有 mode 而生成模糊或异常的样本。

这里先明确一下符号。DMD 的核心组件是一个**一步生成器** $G_\theta$，它从标准高斯噪声 $z \sim \mathcal{N}(0, I)$ 出发，直接输出数据空间的样本 $x = G_\theta(z)$。$z$ 是 latent code，可以理解为生成器的"随机种子"；$G_\theta$ 则是我们要训练的 student 模型。$p_{\text{fake}}$ 是 $G_\theta(z)$ 的分布，$p_{\text{real}}$ 是教师扩散模型经多步采样得到的分布（近似 $p_{\text{data}}$）。

### KL 散度的梯度分解

直接计算 $D_{\text{KL}}(p_{\text{fake}}  p_{\text{real}})$ 需要两个分布的精确概率密度，这在高维空间中不可行。DMD 的核心贡献在于给出了一种可计算的梯度估计。推导分三步。

**第一步：重参数化 + 链式法则。**

将 KL 散度用生成器重写：

$$
D_{\text{KL}}(p_{\text{fake}}  p_{\text{real}}) = \mathbb{E}*{z \sim \mathcal{N}(0,I)} \Bigl[ \log p*{\text{fake}}(G_\theta(z)) - \log p_{\text{real}}(G_\theta(z)) \Bigr].
$$

对 $\theta$ 求梯度，应用链式法则 $\nabla_\theta \log p(G_\theta(z)) = \nabla_x \log p(x) \cdot \nabla_\theta G_\theta(z)$：

$$
\nabla_\theta D_{\text{KL}} = \mathbb{E}*{z} \Bigl[ \bigl( \underbrace{\nabla_x \log p*{\text{fake}}(x)}*{s*{\text{fake}}(x)} - \underbrace{\nabla_x \log p_{\text{real}}(x)}*{s*{\text{real}}(x)} \bigr) \cdot \nabla_\theta G_\theta(z) \Bigr],
$$

其中 $x = G_\theta(z)$，$s(x) \equiv \nabla_x \log p(x)$ 是 score function——它指向概率密度增长最快的方向，本质上是"最优去噪方向"。到目前为止，这是精确的链式求导，没有近似。

**第二步：Score 的扩散化——加噪声绕开密度估计。**

$s_{\text{real}}(x)$ 和 $s_{\text{fake}}(x)$ 都是干净数据空间（$t = 0$）的 score。但数据分布在低维流形上，$t = 0$ 处的 score 是不适定的（ill-defined）：在流形之外的概率密度近乎为零，score 的值会发散。

解决方法是引入**扩散扰动**：给数据加不同程度的高斯噪声，在带噪空间中计算 score，再通过积分恢复干净空间的梯度。具体来说，定义扩散后的样本: 

$$  
x_t = \alpha_t  G_\theta(z) + \sigma_t  \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I),  
$$

其中 $\alpha_t, \sigma_t$ 遵循标准的 diffusion noise schedule（如 $\alpha_t^2 + \sigma_t^2 = 1$）。带噪分布记为 $p_t(x_t) = \int p(x_0)  \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)  dx_0$，其 score 为 $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$。

关键性质：当 $t \to 0$ 时，$p_t \to p$ 且 $s(x_t, t) \to s(x)$。通过在 $t \in [0, T]$ 上积分不同噪声水平的 score，可以得到干净空间梯度的加权表示：

$$
\nabla_\theta D_{\text{KL}} = \mathbb{E}*{z, t, \varepsilon} \Bigl[ w(t) \bigl( s*{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t) \bigr) \cdot \nabla_\theta G_\theta(z) \Bigr],
$$

其中 $w(t)$ 是与 noise schedule 相关的权重函数。这一步的本质是用 diffusion 的平滑性质，把不可直接计算的高维密度梯度转化为可计算的带噪 score 之差的积分。

**第三步：两个 Score 的来源。**

- **$s_{\text{real}}(x_t, t) = \nabla_{x_t} \log p_{\text{real}, t}(x_t)$**：真实数据分布在噪声水平 $t$ 下的 score。预训练的扩散模型（Teacher）就是干这件事的——给定带噪样本，指出 denoising 方向。直接用 Teacher $s_\phi$ 提供。
- **$s_{\text{fake}}(x_t, t) = \nabla_{x_t} \log p_{\text{fake}, t}(x_t)$**：学生生成分布在同一噪声水平下的 score。没有现成的模型——$p_{\text{fake}}$ 随 $G_\theta$ 变化而变化。需要单独训练一个"假 score 模型" $s_\psi$ 来估计它。

梯度公式的直觉非常清晰：$s_{\text{real}}$ 指向真实高密度区，$s_{\text{fake}}$ 指向当前假样本的高密度区，两者的**差**指明了"从假到真"的修正方向。当 $p_{\text{fake}} = p_{\text{real}}$ 时，两个 score 处处相等，梯度为零——这正是我们想要的收敛状态。

### 训练流程

DMD 的训练需要交替更新两个模型：

**Step 1：更新假 score 模型 $s_\psi$**

从学生生成器采样一批假数据 $x_{\text{fake}} = G_\theta(z)$，在上面训练一个小型扩散模型（或微调一个预训练模型）来估计 $s_{\text{fake}}$。这个模型只需要在学生的当前分布上准确即可，因此通常比 Teacher 轻量得多。实际操作中会维护一个生成样本的回放缓冲区，防止假 score 模型遗忘之前见过的样本分布。

**Step 2：更新生成器 $G_\theta$**

冻结 $s_\psi$ 和 $s_\phi$（Teacher），采样 $z$ 和 $t$，加噪得到 $x_t$，计算两个 score 之差，沿梯度方向更新生成器：

$$
\theta \leftarrow \theta - \eta \cdot w(t) \bigl( s_\psi(x_t, t) - s_\phi(x_t, t) \bigr) \cdot \nabla_\theta G_\theta(z).
$$

直观上，这一步让生成器的输出沿着"更像真样本"的方向移动——这正是 score difference 所指示的方向。$s_\phi$（真 score）说"往这边走才是真样本"，$s_\psi$（假 score）说"你现在这堆假样本往这边走"，两者的差就是生成器应该推动样本移动的净方向。

### 与 GAN 和 Score Distillation 的联系

DMD 的交替训练结构天然让人联想到 GAN。实际上，$s_{\text{fake}} - s_{\text{real}}$ 的差值扮演了类似判别器梯度的角色——它告诉生成器"真样本和假样本的区别在哪里"。但 DMD 与 GAN 有本质区别：

- **GAN 的判别器**区分的是原始像素空间中的真假样本。
- **DMD 的"判别器"区分的是加噪空间中**的真假样本在各噪声水平下的 score。加噪操作提供了一个平滑的密度估计任务，避免了 GAN 中常见的不稳定性和 mode collapse。

在极端情况下，如果 $s_{\text{fake}}$ 不训练（或直接设为 0），DMD 退化为 **Score Distillation Sampling（SDS）**——Poole et al.（2023）提出的 DreamFusion 中使用的技术。SDS 的更新方向只用 $s_{\text{real}}$ 的 score 来指导生成器，缺少 $s_{\text{fake}}$ 这一项。

SDS 的问题在于：它隐含地假设了 $p_{\text{fake}}$ 是无信息的均匀分布，因此更新会过度平滑，产生的样本往往缺乏细节（DreamFusion 中需要非常高的 CFG 来补偿）。DMD 通过维护 $s_{\text{fake}}$ 显式建模学生分布，绕开了这一过度平滑的问题，使得单步生成器的质量大幅提升。

此外，$t$ 的采样策略对 DMD 影响显著。当 $t \to 0$ 时，score difference 最接近原始数据分布的梯度方向，但方差大、容易被对抗攻击。当 $t \to T$ 时，任务更平滑但信息量低。实践中，$t$ 通常从中间范围采样（如 $t \in [0.02, 0.5]$ 对于归一化到 $[0,1]$ 的数据），在稳定性和有效性之间取得平衡。

DMD 的后续工作（DMD2, Yin et al., 2024）进一步通过多步生成框架和改进的假 score 估计，在 ImageNet 64×64 等 benchmark 上将一步 FID 推到了接近多步扩散模型的水平。

---

总结一下第三篇的内容：Consistency Model 用自洽性约束替代了逐步蒸馏，让模型学习整条 PF-ODE 的积分器；DMD 则从分布匹配的视角，KL 散度的梯度分解导出了一种类 GAN 的交替训练方案。前者仍然在"轨迹几何"的框架内，后者则跳出了这个框架，直接面向最终目标——匹配分布。

这两种方法各自代表了 2023 年 Efficient Diffusion 的两个重要方向，它们的思想在后续的少步生成方法中被广泛继承和组合。下一篇我们将来看 2024 年的加速方法，它们更多地融合了蒸馏、一致性和分布匹配的思想。