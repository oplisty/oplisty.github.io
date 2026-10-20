---
title: "Pi 系列的回顾：从 π₀ 到 π₀.₅"
date: 2026-10-21
excerpt: "从数据配比、视觉语言输入到动作专家与流匹配，回顾 π₀ 系列在具身智能中的架构演进。"
cover: "/images/pi-series/01_pi0_architecture.png"
categories:
  - Embodied-AI
  - Robotics
  - VLA
tags:
  - Pi0
  - Pi0.5
  - Vision-Language-Action
  - Flow-Matching
  - Robot-Learning
  - Action-Expert
math: true
read_time: true
---

# Pi 系列的回顾：从 π₀ 到 π₀.₅

> 本文整理 π 系列在数据配比、模型架构与流匹配动作生成方面的关键变化；π₀.₅ 部分目前保留为后续补充内容。

VLA一年多的发展如洪水猛兽,但是能打过pi系列的几乎没有,那么近期笔者在探索移动操作长时程任务的模型架构,重新走一下从24年10月到26年4月,从pi 0 到pi 0.7,究竟经历了哪些变化,为什么这些变化能work呢?其中在一开始$\pi_0$ 横空出世的时候,他提到大规模训练的两个关键点,一个是模型架构,要能够利用各种数据源,同时能够表示与复杂物理场景交互所需的精细和微妙的行为,其次是正确的data recipe,所以本篇文章也从架构和recipe两个部分进行比较和分析。

## 1. 数据配比

### $\pi_0$

数据集包含大约 10w 小时的演示,并辅以开源 OXE 数据集 [10]。每个数据有任务名称和分段标注,

1. 预训练数据:尽可能多的任务,每个任务的行为尽量diverse
2. 后训练微调数据: 更有助于完成任务的行为

* **预训练数据:** 开源数据集组成,包括  OXE 、Bridge v2  和 DROID。这些数据集  中的机器人和任务通常配备一到两个摄像头,并使用 2 到 10 Hz 之间的低频控制。自采数据,单臂双臂 1:8的数据配比. 每个任务由许多复杂的子任务组成,而不纯纯是名词和动词的任意组合 例如,“清理餐桌”任务包括 将各种各样的不同餐具、杯子和器具放入清理箱中,并 将大量的垃圾物品放入垃圾桶。
  * 平衡不同长度任务的数据占比:将每个任务-机器人组合按 $n^{0.43}$ 加权,其中 n 是该组合的样本数量,使得过度表示的组合会被下调权重。
* **后训练数据:** 不同任务需要非常不同的数据集,最简单的任务仅需 5 个小时的数据,而最复杂的任 务则可能使用 100 多个小时的数据。也对每个数据放入每个任务名称和分段标注.推理的时候由上层VLM 提供子任务标签。

### $\pi_{0.5}$

提出要实现泛化必须从多种信息源种迁移经验和知识,将不同模态纳入同一序列建模框架适应机器人数据、语言数据、计算机视觉及上述组合的训练。**声称能够实现场景的泛化,可以在未见过的家庭环境中胜任**（本节待后续补充。）

## 2. 模型架构

本文的第二部份来比较一下架构,

### $\pi_0$

<figure>
  <img src="/images/pi-series/01_pi0_architecture.png" alt="pi_0 architecture overview">
  <figcaption>π₀ 的整体架构：预训练 VLM 与动作专家共同建模视觉、语言、本体状态和动作。</figcaption>
</figure>

<figure>
  <img src="/images/pi-series/02_pi0_flow_pipeline.png" alt="pi_0 flow and token pipeline">
  <figcaption>π₀ 的 token 流程：图文 prefix、当前本体状态与带噪动作 suffix 进入统一的 Transformer。</figcaption>
</figure>

#### 图像和语言

1. 头部+2个腕部相机拢共3个相机视角,默认分辨率是224*224,然后如果没有第三路相机就用黑图占位,然后编码器用SigLIP So400m, 编码成256个token,各个相机共用编码器但是独立编码,最后再concate到一起,变成768个token,然后对于黑图就mask掉对应的token.

   <u>具体而言就是224 * 224按14 *14的patch划分成256个patch,用stride=14的卷积来提取,输出通道1152,接着展平成256个token(dim=1152),最后27层SigLIP编码成dim=2048</u> .
   $$
   [B,256*3,2048]
   $$
   这里是因为SigLIP 是PaliGemma的视觉encoder,直接就搬过来用了,现在常用的还有**Qwen3-VL-4B-Instruct**,**Qwen3.5-4B,Qwen3.5-2B,Cosmos-Reason2-2B**,这里选用的一个标准就是要**使用方便，而且规模相对较小，适合实时控制**,所以理论上我们希望能够在不增加控制延迟上选用更强的VLM, 因为往往具备更强的语言理解,更好的视觉语义和grounding,甚至是更好的空间与关系表征。

2. language 先添加 BOS,来标记序列开头和区分句子边界,然后用PaliGemma的tokenizer进行编码编码,末尾追加单独编码的换行标记,然后padding 或截断到配置长度,zuihou 生成有效 token mask,维度也是2048和视觉模块对齐.
   $$
   [B,48,2048]
   $$

3. 最后把这三个concat起来得到输入$[B,816,2048]$

#### 本体动作

1. 将机器人的本体关节按照具体顺序:左臂,夹爪,右臂,右夹爪来进行排序,

2. 统计数据集中机器人状态和动作每个关节的取值分布得到均值和标准差

3. 用训练的统计量进行z-score归一化来消除不同关节输出量纲的影响,不同关节的数值取值范围差很大,不好直接给模型
   $$
   \tilde{s}_d
   =
   \frac{s_d-\mu_d}{\sigma_d+10^{-6}}
   $$

4. 将维度固定为最大的机器人配置维度,如果1. 排序后的关节填不满就补0,这里补到32维

   1. PS: 固定成 32 维，不意味着所有机器人的每个位置含义都一样,因此，使用某个机器人配置时，要保持**字段顺序、数值约定和归一化统计量与训练时一致**

5. 采样噪声步$\tau$
   $$
   u\sim\mathrm{Beta}(1.5,1)，\tau=0.999u+0.001
   $$

6. 接着经过一个MLP(带bias)把32维映射成一个1024维的token,同时初始化一个带噪声动作块,包含50个动作,每个动作也是32维的本体状态来表示,映射后变成50个token,

$$
X^\tau=(1-\tau)A+\tau\epsilon
$$

$$
[B,32]\rightarrow[B,1024]\rightarrow[B,1,1024].
$$

$$
[B,50,32]\rightarrow[B,50,1024].
$$

7. 将时间步 $\tau$ 编码成正弦余弦向量，注入动作 token，让模型知道当前的 flow 时间步：

$$
e_\tau = \operatorname{SinCos}(\tau) \in \mathbb{R}^{B\times1024}
$$

具体来说，对 batch 中的每个标量 $\tau_b$，先固定 **512 个不同的周期**。这些周期按对数间隔从 $0.004$ 到 $4.0$：

$$
p_j = 0.004\times1000^{\frac{j}{511}},
\qquad j=0,1,\ldots,511.
$$

然后计算每个周期对应的角度：

$$
\theta_{b,j}=\frac{2\pi\tau_b}{p_j}.
$$

对这些角度分别取正弦和余弦，最后拼成一个 1024 维向量：

$$
\begin{aligned}
e_{\tau_b}=\big[\;&\sin\theta_{b,0},\ldots,\sin\theta_{b,511},\\
&\cos\theta_{b,0},\ldots,\cos\theta_{b,511}\;\big].
\end{aligned}
$$

因此，一个时间标量会变成“512 个正弦值 + 512 个余弦值”的 1024 维向量；对 batch 中的 $B$ 条样本堆叠后得到：

$$
e_\tau\in\mathbb{R}^{B\times1024}.
$$

不同周期让这些分量对时间变化呈现不同快慢的响应，供后面的网络使用；这一步完全由固定公式计算，没有可学习参数。

再复制到 50 个动作位置：

$$
[B,1024]\rightarrow[B,50,1024].
$$

随后，**动作 embedding 和时间 embedding 沿特征维拼接**：

$$
[B,50,1024]+[B,50,1024] \rightarrow [B,50,2048].
$$

最后通过两层 MLP：

$$
z_{a,i} = W_3\, \operatorname{SiLU} \left( W_2 \left[ W_1x_{\tau,i};e_\tau \right] +b_2 \right) +b_3.
$$

得到：

$$
z_a\in\mathbb{R}^{B\times50\times1024}.
$$

#### 模型注意力

现在文本,语言,本体,噪声全都到了同个空间中来了,接下来分为两个分支

| 项目                       | 图文分支 `gemma_2b` | 动作分支 `gemma_300m` |
| -------------------------- | ------------------- | --------------------- |
| Transformer 层数           | 18                  | 18                    |
| Hidden width               | 2048                | 1024                  |
| MLP 中间维度               | 16384               | 4096                  |
| Query 头数                 | 8                   | 8                     |
| Key/Value 头数             | 1                   | 1                     |
| 每个 attention head 的维度 | 256                 | 256                   |

两条分支拥有各自的：

- RMSNorm；
- Q、K、V 投影；
- Attention 输出投影；
- MLP。

视觉语言分支的投影矩阵为：

$$
W_Q^P\in\mathbb{R}^{2048\times2048},
\qquad
W_K^P,W_V^P\in\mathbb{R}^{2048\times256}.
$$

动作分支的投影矩阵为：

$$
W_Q^S\in\mathbb{R}^{1024\times2048},
\qquad
W_K^S,W_V^S\in\mathbb{R}^{1024\times256}.
$$

因此，两条分支投影后的张量形状为：

$$
Q_P\in\mathbb{R}^{B\times N_P\times8\times256},
\qquad
Q_S\in\mathbb{R}^{B\times N_S\times8\times256}.
$$

$$
K_P,V_P\in\mathbb{R}^{B\times N_P\times1\times256},
\qquad
K_S,V_S\in\mathbb{R}^{B\times N_S\times1\times256}.
$$

其中 prefix 和 suffix 的 token 数为：

$$
N_P=3\times256+48=816,
\qquad N_S=51.
$$

动作分支虽然输入只有 1024 维，也会先生成总计 2048 维的 Q，再按 $8\times256$ 组织成多头；完成注意力计算后，两条分支各自用输出投影，将结果映射回 2048 维和 1024 维。

```python
q = q.permute(0, 2, 1, 3)  # [B, 8, N, 256]
k = k.permute(0, 2, 1, 3)  # [B, 1, N, 256]
v = v.permute(0, 2, 1, 3)  # [B, 1, N, 256]

scores = (q @ k.transpose(-2, -1)) / (256 ** 0.5)
# [B,8,N,256] @ [B,1,256,N]
# → [B,8,N,N]
```

接着按token 维度concat 变成512维度
$$
Q=\operatorname{Concat}_{\text{sequence}}(Q_P,Q_S),

K=\operatorname{Concat}_{\text{sequence}}(K_P,K_S),

V=\operatorname{Concat}_{\text{sequence}}(V_P,V_S).
$$
Q 和 K 还会加入 RoPE 位置编码，然后计算：
$$
Y=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{256}}+M
\right)V,
$$
其中 $M$ 是注意力 mask，分成三个块：图像与语言、当前状态、整个带噪动作块。

| Query \ Key/Value | 图像与语言 | 当前状态 | 全部动作 token |
| ----------------- | ---------- | -------- | -------------- |
| 图像与语言        | ✓          | —        | —              |
| 当前状态          | ✓          | ✓        | —              |
| 任意动作 token    | ✓          | ✓        | ✓              |

具体构造就是对于 (N) 个 token 构造一个块起始标记。比如对于 2 个 prefix token、1 个状态 token、3 个动作 token，序列为

$[p_0,\ p_1,\ s,\ a_0,\ a_1,\ a_2]$。

构造一个 0、1 序列，在切换处标为 1：

```
ar_mask = [0, 0, 1, 1, 0, 0]
```

然后累加得到块编号

```
block_id = [0, 0, 1, 2, 2, 2]
```

然后构造

```python
cumsum = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
```

也就是,只能看到比自己小的部分得到

| Query ↓ / Key → | $p_0$ | $p_1$ | $s$ | $a_0$ | $a_1$ | $a_2$ |
| --------------- | ------- | ------- | ----- | ------- | ------- | ------- |
| $p_0$           | 1       | 1       | 0     | 0       | 0       | 0       |
| $p_1$           | 1       | 1       | 0     | 0       | 0       | 0       |
| $s$             | 1       | 1       | 1     | 0       | 0       | 0       |
| $a_0$           | 1       | 1       | 1     | 1       | 1       | 1       |
| $a_1$           | 1       | 1       | 1     | 1       | 1       | 1       |
| $a_2$           | 1       | 1       | 1     | 1       | 1       | 1       |

mask 为 `1` 时保留原分数；mask 为 `0` 时，将原分数替换成极大的负数，使 softmax 后对应位置的权重变为 0。

然后输出仍然保留所有 token 的位置。模型按照 prefix、suffix 的序列范围切开结果，再应用各自的输出投影：

$$
Y_P \rightarrow [B,N_P,2048],
\qquad
Y_S \rightarrow [B,N_S,1024].
$$

一个 block 结束后，两条分支分别进行残差连接和 MLP 更新：

$$
\begin{aligned}
h' &= h+\operatorname{AttentionOutput}(h),\\
h_{\text{next}} &= h'+\operatorname{MLP}\left(\operatorname{RMSNorm}(h')\right).
\end{aligned}
$$

MLP 使用门控形式，可以概括为：

$$
\operatorname{MLP}(x)
= \left[\operatorname{GELU}(xW_g)\odot(xW_u)\right]W_d.
$$

图文分支的 MLP 中间维度为 16384，动作分支为 4096。

#### 损失计算

取 suffix 的最后 $H$ 个位置，即动作 token，经过输出投影：

$$
[B,H,1024] \rightarrow [B,H,32],
$$

得到动作速度预测：

$$
\hat{v}_\tau=v_\theta(X_\tau,o,\tau).
$$

流匹配损失为：

$$
\begin{aligned}
\mathcal{L}
&=\mathbb{E}_{o,A,\epsilon,\tau}\Bigg[
\frac{1}{HD}\sum_{i=1}^{H}\sum_{j=1}^{D}\\
&\qquad\left(
v_\theta(X_\tau,o,\tau)_{i,j}-(\epsilon-A)_{i,j}
\right)^2
\Bigg].
\end{aligned}
$$

推理时默认使用 10 次欧拉积分，步长为：

$$
\Delta\tau=-\frac{1}{N}=-0.1.
$$

带噪动作块从标准高斯分布采样，每一步按照下面的公式更新：

$$
X_{\tau+\Delta\tau}=X_\tau+\Delta\tau\,v_\theta(X_\tau,o_k,\tau).
$$

同一个动作块的 10 次 flow 迭代中，图像、语言和当前状态保持不变，只有带噪动作与 flow 时间持续变化。又因为 prefix 不读取 suffix，所以各层的 prefix K/V 与带噪动作无关，可以预先计算并缓存。实际执行顺序是：编码图像与语言；执行一次 prefix 前向；缓存每一层的 prefix K/V；然后在每次 flow 迭代中运行动作分支并读取这些缓存。

#### 训练细节

1. 本体状态向量 $q_t$  和动作向量 $a_t$  总是具有数据集中最大机器人( 18,两个 6 自由度手臂、2 个夹爪、一个移动基座(x,y,yaw)和一个垂直可动躯干)的维度。
   1. 本体状态空间和动作空间维度较低的机器人,我们对配置和动作向量进行零填充
   2. 对于图像数量少于三张的机器人,屏蔽掉缺失的图像槽位。

#### 实验 insight

1. 从较大的预训练 VLM 初始化有利于模型在高级指导下的复杂任务上更好的自主性能。
2. 微调评测的时候,预训练见过的类似任务的任务和预训练完全没见过的任务的难度是截然不同的,如果只在预训练见过的任务或者类似任务评测是不合理的。

### $\pi_{0.5}$

本节待后续补充。
