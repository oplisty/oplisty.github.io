---
title: "Efficient Diffusion（四）：SiD、ECT 与连续时间一致性模型"
date: 2026-06-05
excerpt: "2024 年 DMD 与 Consistency Model 的关键改进：Score identity Distillation 用恒等式消去假 score 模型，ECT 让 CT 稳定可训，sCM 将一致性模型推向连续时间极限。"
cover: "/images/Efficient_diffusion.jpg"
categories:
  - Diffusion
tags:
  - Diffusion
  - Efficient-Diffusion
  - Few-Step
  - SiD
  - ECT
  - sCM
  - Consistency-Model
  - DMD
math: true
read_time: true
---
书接上文，2023 年的 Consistency Model 和 DMD 分别从轨迹自洽性和分布匹配两个角度打开了少步生成的大门。进入 2024 年，这两个方向各自暴露出了一些结构性问题，催生了一系列改进工作。与此同时，第三个方向——将 GAN 的对抗性思路引入蒸馏——也开始崭露头角。本文先聚焦前两条改进路线：DMD 的假 score 模型能否省去？Consistency Training 的稳定性如何解决？连续时间下的一致性模型还有什么空间？

  

## 目录

  

- [Score identity Distillation (SiD)](#score-identity-distillation-sid)

- [DMD 的痛点：假 score 模型](#dmd-的痛点假-score-模型)

- [Score identity：一个被忽略的数学恒等式](#score-identity一个被忽略的数学恒等式)

- [SiD 的损失函数](#sid-的损失函数)

- [SiD vs DMD：对比与局限](#sid-vs-dmd对比与局限)

- [Easy Consistency Tuning (ECT)](#easy-consistency-tuning-ect)

- [CT 为什么难训](#ct-为什么难训)

- [ECT 的改进策略](#ect-的改进策略)

- [ECT 的实际效果与意义](#ect-的实际效果与意义)

- [sCM：连续时间一致性模型的进阶](#scm连续时间一致性模型的进阶)

- [离散化的代价](#离散化的代价)

- [sCM 的核心改进](#scm-的核心改进)

- [从 sCM 到 sCM2](#从-scm-到-scm2)
## Score identity Distillation (SiD)
### DMD 的痛点：假 score 模型

回顾 DMD 的训练流程，每轮迭代需要做两件事：先更新假 score 模型 $s_\psi$ 来估计学生当前分布 $p_{\text{fake}}$ 的 score，再用 $s_\psi$ 和 Teacher $s_\phi$ 的差值更新生成器 $G_\theta$。这个假 score 模型是一个完整的扩散模型，需要在学生当前的生成分布上持续做 denoising score matching。它引入了额外的训练成本、一组超参数（模型结构、学习率、更新频率、回放缓冲区大小），而且是训练不稳定的潜在来源——假 score 一旦估计不准，梯度方向就会偏。
一个自然的问题是：**能不能省掉 $s_\psi$，用某种数学技巧直接从 Teacher 获取 $s_{\text{fake}}$ 的估计？**

这就是 SiD（Score identity Distillation, Xu et al., 2024）要回答的问题。
### Score identity：一个被忽略的数学恒等式
SiD 的核心洞察源于扩散模型中 denoiser 和 score function 之间的一个精确关系。在 EDM 风格的参数化下，扩散模型 $D_\phi$ 预测的是干净样本 $\hat{x}_0$，而 score function 可以通过 Tweedie 公式与 denoiser 关联：
$$

s(x_t, t) = \frac{D_\phi(x_t, t) - x_t}{\sigma_t^2}.

$$
这个恒等式对**任何**分布都成立——不论输入来自 $p_{\text{real}}$ 还是 $p_{\text{fake}}$。关键点在于：同一个预训练 Teacher $D_\phi$，既可以对真实样本加噪后的 $x_t^{\text{real}}$ 去噪（给出 $s_{\text{real}}$），也可以对学生假样本加噪后的 $x_t^{\text{fake}}$ 去噪。

当 Teacher 面对假样本时会发生什么？Teacher 在真实数据上训练，从未见过假样本。但它仍然会做出预测——它"以为"输入是一个真实样本加噪的结果，并按照自己对真实分布的理解来去噪。Teacher 对假样本的 denoising 预测和假样本本身干净成分之间的差距，恰恰编码了 $p_{\text{fake}}$ 偏离 $p_{\text{real}}$ 的方向和大小。

利用上述恒等式，SiD 将 DMD 中的 score difference 重写为 Teacher denoising 预测之差：
$$

s_{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t) \propto \frac{D_\phi(x_t^{\text{fake}}, t) - D_\phi(x_t^{\text{real}}, t)}{\sigma_t^2}.

$$
注意这里不需要假 score 模型——**同一个 Teacher $D_\phi$**，分别作用在假样本和真样本的带噪版本上，两者的 denoising 预测之差就直接替代了 $s_{\text{fake}} - s_{\text{real}}$。
### SiD 的损失函数
基于上述恒等式，SiD 推导出了一个简洁的生成器损失。其梯度等价于：
$$

\nabla_\theta \mathcal{L}_{\text{SiD}} = \mathbb{E}_{z, t, \varepsilon} \Bigl[ \omega(t) \cdot \bigl( D_\phi(x_t^{\text{fake}}, t) - D_\phi(x_t^{\text{real}}, t) \bigr) \cdot \nabla_\theta G_\theta(z) \Bigr],

$$
其中 $x_t^{\text{fake}} = \alpha_t G_\theta(z) + \sigma_t \varepsilon$，$x_t^{\text{real}}$ 是真样本加噪到同一噪声水平的结果，$\omega(t)$ 是合并了 score identity 系数后的总权重。
	这个梯度的直觉与 DMD 一致——$D_\phi(x_t^{\text{fake}}, t) - D_\phi(x_t^{\text{real}}, t)$ 指明了"从假到真"的修正方向——但不需要假 score 模型。Teacher 自己就是真假判别器：它对真样本知道该往哪去噪，对假样本则会"困惑"，两者的预测差就是最直接的分布匹配信号。

受益于这个损失函数,SiD 少了一组交替更新的模型，训练动态更简单，超参数更少,**训练更稳定**且**收敛更快**(SiD 报告在 ImageNet 64×64 上只需 DMD 约 1/10 的训练迭代即可达到相近的 FID)。

但 SiD 也有局限。恒等式的使用依赖于 Teacher 对分布外的假样本也能给出"有意义的"去噪预测。如果学生生成器在训练初期的输出过于异常，Teacher 完全无法理解这些输入，恒等式推导的信号就会非常嘈杂。此外，$t$ 的采样范围和权重 $\omega(t)$ 仍然需要仔细设计。

SiD 更深层的意义在于一个洞察：**预训练扩散模型本身就已经编码了足够的信息来区分真假分布，不需要额外的判别器或 score 模型**。这个洞察直接影响了后续将 GAN 与蒸馏结合的工作——既然 Teacher 天然是一个"判别器"，为什么不进一步引入对抗训练来强化这个判别能力？这是下一篇的主题。
## Easy Consistency Tuning (ECT)

前面聊Consistency Training (CT) 的时候我们就提到了CT的几个关键设计:
1.  CT 用 $(x_{t_n}, x_{t_{n+1}})$ 这对带噪样本施加自洽性约束，其中 $x_{t_n} = x_0 + t_n \varepsilon$、$x_{t_{n+1}} = x_0 + t_{n+1} \varepsilon$，使用**同一个噪声** $\varepsilon$ 保证来自同一条 PF-ODE 轨迹。
 2.  CT 训练开始时，$f_\theta$ 的输出近乎随机，EMA 目标 $f_{\theta^-}$ 同样随机。两个随机函数互相"对齐"，自洽性约束没有提供任何朝向正确答案的驱动力——直到边界条件 $f(x_0, 0) = x_0$ 的信号从 $t = 0$ 端缓慢传播过来。这个传播过程在训练初期极其脆弱，模型极易陷入局部最优（如输出常数图像——自洽性完美满足，但毫无意义）或完全发散。
 那么ECT是怎么解决他们这两个问题的呢
### ECT 的算法流程

ECT 采取渐进式时间范围,将训练划分为 $K$ 个阶段。定义阶段 $k$ 的时间上界为 $T_k$，满足：
$$

0 < T_1 < T_2 < \dots < T_K = T,

$$
即每个阶段的时间范围逐步扩大，最终覆盖全时间域 $[0, T]$。如果一开始就在全范围 $[0, T]$ 上训练，$t$ 接近 $T$ 的样本离边界锚点太远——它们需要通过 $t = T \to t_{N-1} \to \dots \to t_1 \to 0$ 的长链才能接收到边界条件的信号。在训练初期，链条上的每个环节都不准确，信号在传播中迅速衰减为噪声。模型实际上在黑暗中摸索，极易收敛到"一致但错误"的平凡解（如输出常数灰色图像）。渐进式时间范围确保边界锚点只需传播很短的链就能覆盖整个训练域。模型先在狭窄范围内建立可靠的自洽性，然后逐步扩大范围——每个新阶段都从一个已经（近似）正确的函数出发，新的、稍远的时间步可以将其作为可靠的锚点链。

**阶段 $k$ 的训练流程**（给定 $T_k$ 和一致性模型 $f_\theta$，维护其 EMA 副本 $f_{\theta^-}$）： 
1.  $f_\theta$初始化策略采取复用 Teacher 的部分底层权重（如第一层卷积甚至整个 encoder）给 $F_\theta$ 提供了一组基础的视觉特征提取能力来大幅降低训练初期的探索难度
2. 采样干净样本 $x_0 \sim p_{\text{data}}$，噪声 $\varepsilon \sim \mathcal{N}(0, I)$，以及当前阶段时间范围内的 $t \sim \mathcal{U}[0, T_k]$。
3. 构造同一轨迹上相邻的两个带噪样本（CT 方式，无需 Teacher）：
$$

x_t = x_0 + t \,\varepsilon, \qquad x_{t+\Delta t} = x_0 + (t + \Delta t) \,\varepsilon,

$$
其中 $\Delta t$ 是固定的小时间步长（远小于 $T_k$），同一噪声 $\varepsilon$ 保证两点落在同一条 PF-ODE 轨迹上。
3. 计算加权自洽性损失并更新 $\theta$：
$$

\mathcal{L}_k(\theta) = \mathbb{E}_{x_0, t, \varepsilon} \Bigl[ w_k(t) \cdot d\bigl( f_\theta(x_{t+\Delta t}, t+\Delta t),\; f_{\theta^-}(x_t, t) \bigr) \Bigr],

$$
其中 $w_k(t)$ 是阶段 $k$ 专用的自适应时间步权重函数,由于不同时间步的自洽性难度差异极大。$t$ 接近 $T_k$ 时，输入接近纯噪声，自洽性约束要求模型从一团噪声直接预测出干净图像；$t$ 接近 0 时，输入接近干净图像，模型只需做微小修正。如果用统一权重，高噪声端的巨大损失会主导梯度，压倒低噪声端的精细学习信号。所以$w_k(t)$ 的设计原则是：在阶段 $k$ 的前半段，降低高噪声端的权重，让模型先把低噪声区域学好；在阶段后半段，逐步提高高噪声端权重，推着模型攻克困难区域。

$d(\cdot, \cdot)$ 是组合距离度量: 在 $t$ 较小（低噪声）的区域额外加入 LPIPS 感知损失，让模型捕捉纹理和细节；在 $t$ 较大（高噪声）的区域仅用 Pseudo-Huber，因为此时主要任务是全局结构重建，感知损失意义不大

$f_{\theta^-}$ 端使用 stop-gradient，参数通过 EMA 更新 $\theta^- \leftarrow \mu \theta^- + (1-\mu) \theta$。

4. 在当前阶段训练至收敛后，扩大时间上界至 $T_{k+1}$，进入下一阶段。循环直到 $T_K = T$。

ECT 与原始 CT 的关系，有点像 ADAM 之于 SGD, 它没有改变核心优化目标（自洽性约束），但通过训练策略的优化大幅提升了收敛速度和成功率。这使得一致性模型真正成为了一个独立的生成模型训练范式，而不仅仅是扩散模型的蒸馏工具。

## sCM：连续时间一致性模型的进阶
无论是 CD 还是 CT，训练时都需要将连续时间 $[0, T]$ 离散化为 $N$ 个时间点 $\{t_0, t_1, \dots, t_N\}$，在相邻点对上施加自洽性约束。这个离散化带来了两个根本性问题：
1.  自洽性约束只在相邻离散点之间局部成立。如果 $t_n$ 和 $t_{n+1}$ 的间距不够小，PF-ODE 在该区间内的弯曲就无法被线性 Euler 步充分近似，自洽性目标本身就包含了近似误差。更麻烦的是，一致性是由链式约束间接建立的——$f(x_{t_n}) \approx f(x_{t_{n+1}}) \approx f(x_{t_{n+2}}) \approx \dots$——每个环节的误差在这个链条上累积，到 $t_{\max}$ 端可能变得显著。
2. **$N$ 的困境。** 减小误差需要增大 $N$，但 $N$ 过大会增加训练成本（每个 batch 的有效学习信号与 $N$ 无关，但每个样本需要更多的点对），同时相邻点差异过小导致自洽性信号本身变弱。选择最优的 $N$ 需要大量消融实验，且最优值往往与数据集和模型规模相关，缺乏普适性。

所以sCM提出一个challenge: **能不能直接在连续时间上训练一致性模型，完全消除离散化？**

### sCM 的算法流程
sCM的核心思路是将离散化的自洽性约束替换为连续时间导数约束。
在算法开始前确定,sCM 使用与标准 CM 相同结构的参数化，但对系数做了重新推导：
$$

f_\theta(x_t, t) = c_{\text{skip}}(t) \cdot x_t + c_{\text{out}}(t) \cdot F_\theta(x_t, t),

$$
因为标准 CM 的 $c_{\text{skip}}$ 和 $c_{\text{out}}$ 假设是 $t_n$ 和 $t_{n+1}$ 之间有固定的间距。但是若$t$ 在 $[0,T]$ 上连续分布，相邻点间隔 $\Delta t$ 极小，$x_t$ 和 $x_{t+\Delta t}$ 几乎相同。此时如果 $c_{\text{out}}(t)$ 的尺度不合适，损失函数对 $F_\theta$ 的梯度会在不同 $t$ 处出现数量级差异(大 $t$ 处梯度爆炸，小 $t$ 处梯度消失)。所以这里的 $c_{\text{skip}}(t)$ 和 $c_{\text{out}}(t)$ 的形式由连续时间 PF-ODE 的解析性质确定来保证 $F_\theta$ 在不同 $t$ 下的输入输出保持单位方差量级。

**训练流程**（给定一致性模型 $f_\theta$ 和 EMA 副本 $f_{\theta^-}$，以无 Teacher 的 CT 模式为例）：
1. 采样干净样本 $x_0 \sim p_{\text{data}}$，噪声 $\varepsilon \sim \mathcal{N}(0, I)$，以及连续时间 $t \sim \mathcal{U}[t_{\min}, T]$。不再预先离散时间轴——$t$ 从连续区间中随机抽取。
2. 构造 PF-ODE 轨迹上相邻的两个点。选取一个极小的时间微元 $\Delta t \ll T$（例如 $\Delta t \approx 0.001 \cdot T$），用同一噪声 $\varepsilon$ 保证两点落在同一条轨迹上：
$$

x_t = x_0 + t \,\varepsilon, \qquad x_{t+\Delta t} = x_0 + (t + \Delta t) \,\varepsilon.

$$
	在有 Teacher 模式下，$x_{t+\Delta t}$ 由 Teacher 的 Euler 步计算，与 CD 的步骤 3 相同
使用有限差分而非解析求导可以极大减轻计算量,解析求导需要计算 PF-ODE 的向量场并做 Jacobian-vector product，代价是前向传播的两倍以上。有限差分 $\frac{f_\theta(x_{t+\Delta t}) - f_\theta(x_t)}{\Delta t}$ 只需两次函数求值，且当 $\Delta t \ll T$ 时近似精度已足够。关键是 $\Delta t$ 可以比离散 CM 的网格间距小得多——离散 CM 受限于 $N \leq 40$ 左右，而 sCM 的 $\Delta t$ 可以轻松做到网格间距的 $1/10$ 甚至 $1/100$。
3. 用有限差分近似连续时间导数，计算一致性损失并更新 $\theta$：
$$

\mathcal{L}_{\text{sCM}}(\theta) = \mathbb{E}_{x_0, t, \varepsilon} \Bigl[ \Bigl\| \frac{f_\theta(x_{t+\Delta t}, t+\Delta t) - f_{\theta^-}(x_t, t)}{\Delta t} \Bigr\|^2 \Bigr] \;\approx\; \mathbb{E}\Bigl[ \Bigl\| \frac{d}{dt} f_\theta(x_t, t) \Bigr\|^2 \Bigr],

$$
$f_{\theta^-}$ 端使用 stop-gradient，参数通过 EMA 更新 $\theta^- \leftarrow \mu \theta^- + (1-\mu) \theta$。
	相比于离散 CM 的损失是 $\|f_\theta(x_{t_n}) - f_{\theta^-}(x_{t_{n+1}})\|$，只在 $N$ 个预定义的离散点对上施加约束。
	连续导数约束 $\|\frac{d}{dt} f_\theta\|^2 \to 0$ 要求函数在**所有** $t$ 处都平滑不变，从泛函分析的角度，强制导数处处为零等价于强制函数沿整条轨迹为常数——这正是"一致性"的精确数学定义。离散 CM 只是对这个连续条件做了粗粒度的近似，而 sCM 直接优化这个精确条件。
4. 当 $t < t_{\text{cutoff}}$ 时（即样本过于接近干净图像），触发截断策略：放大 $\Delta t$（使时间差分信号更强）或改用直接监督损失（利用边界条件 $f(x_0, 0) = x_0$ 提供绝对目标），避免导数约束在该区域退化为 trivial 的零信号。
	这里解释一下,当 $t \to 0$ 时，$f_\theta(x_t, t) \to f_\theta(x_0, 0) = x_0$（由边界条件的参数化 $c_{\text{skip}}(0)=1, c_{\text{out}}(0)=0$ 保证）。在 $t=0$ 的邻域内，$f_\theta$ 几乎就是恒等映射,这不是因为模型学得好，而是因为 $c_{\text{out}}(t) \approx 0$ 把 $F_\theta$ 的输出"屏蔽"了。此时时间导数 $\frac{d}{dt} f_\theta \approx 0$，损失信号几乎为零，但其实这些区域没有提供任何有效的学习信号,模型只是在利用参数化的性质"蒙混过关",触发截止策略就是为了防止这个。

总结一下第四篇的内容：2024 年在 DMD 和 CM 两条主线上各自出现了重要的改进工作。SiD 用 score identity 恒等式消去了 DMD 中昂贵的假 score 模型，简化了训练并加快了收敛；ECT 通过渐进式时间范围训练、自适应加权和更好的初始化，让 CT 路线从理论可行变成工程可靠；sCM 则将一致性模型的离散化约束推向连续时间极限，从根本上消除了离散化误差。

这三项工作虽各有侧重，但共同指向一个趋势：**少步生成模型的训练正在从"需要精心设计的复杂流水线"向"简单、稳定、可扩展"的方向演进**。而下一篇要讲的——将 GAN 的对抗性思路与这些方法融合——则是这个趋势的下一个转折点。