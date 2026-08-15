---
title: "浅谈VLA中的Action Tokenizer"
date: 2026-08-15
excerpt: "从标量离散化、轨迹压缩到学习式动作表征，梳理 VLA 中 Action Tokenizer 的三大技术路线，以及 RT-1、FAST、ActionCodec、OAT 各自的设计权衡。"
cover: "/images/action-tokenizer/three_action_tokenizer_routes.png"
categories:
  - Embodied-AI
  - Robotics
tags:
  - VLA
  - Action-Tokenizer
  - Tokenization
  - RT-1
  - FAST
  - ActionCodec
  - OAT
  - Robot-Learning
math: true
read_time: true
---

# 浅谈VLA中的Action Tokenizer

## 从标量离散化、轨迹压缩到学习式动作表征

Transformer如日中天的今天, VLA 中使用 Transformer 已经是没有什么好说的选择了,但是这里存在一个十分明显的矛盾: **离散 token与连续动作的不对齐**

一个机械臂在时刻 $t$ 输出的动作通常是一组连续值：

$$
a_t =
[\Delta x,\Delta y,\Delta z,
\Delta r,\Delta p,\Delta y,
g]
$$

它们描述末端执行器的位置变化、旋转以及夹爪状态。如果进一步使用目前 VLA 中常见的 **Action Chunking**，一次预测未来 $H$ 个时间步，那么模型真正需要输出的是：

$$
A_t=[a_t,a_{t+1},...,a_{t+H-1}]
\in \mathbb{R}^{H\times D}
$$

**怎样把一个连续、高维、具有强时间相关性的动作轨迹，变成 Transformer 能够高效学习和生成的离散符号就是 Action Tokenization**。

过去几年，这个问题已经逐渐形成三条不同的技术路线：

1. **Scalar Discretization：逐维数值离散化**
2. **Trajectory Compression：轨迹压缩式 Tokenization**
3. **Learned Action Representation：学习式动作表征**

它们看起来都在做 “continuous action → discrete token”，但实际上回答的是三个不同的问题：

- **连续动作怎么离散化？**
- **一整段轨迹怎么更高效地表示？**
- **什么样的离散动作表示最适合模型学习和生成？**

![Three technical routes of Action Tokenization](/images/action-tokenizer/three_action_tokenizer_routes.png)

# 1. 为什么 Action Tokenizer 会成为 VLA 的核心问题？

如果 VLA 采用连续动作头，问题相对直接。VLM 得到视觉和语言表示后，可以接一个 MLP、Diffusion Policy 或 Flow Matching Action Expert：

$$
(V,L)\rightarrow h\rightarrow A
$$

但另一条非常诱人的路线，是完全复用语言模型最成熟的训练范式：

$$
P(x_1,x_2,\cdots,x_n)
=
\prod_i P(x_i|x_{<i})
$$

如果动作也能表示成离散 token：

$$
A\rightarrow(c_1,c_2,\cdots,c_K)
$$

机器人策略就可以被改写成：

$$
P(c_1,\cdots,c_K|V,L)
=
\prod_{k=1}^{K}
P(c_k|V,L,c_{<k})
$$

也就是说，**机器人控制第一次能够真正进入 next-token prediction 框架。**但语言本来就是符号，而机器人动作不是。Tokenizer 如何构造，会直接改变 VLA 所面对的学习问题。

两个 tokenizer 即使都能够以很小的 reconstruction error 恢复动作：

$$
\|A-\hat A\|\approx 0
$$

它们对 VLA 来说却未必一样容易学习。一个 tokenizer 可能让相邻动作得到完全不同的 token，另一个则能保持高度稳定的离散表示。前者会人为制造分类边界，后者则保留了物理动作空间原本的连续性。

因此 Action Tokenizer 真正决定的是机器人连续控制空间与离散 Transformer 之间的 **representation interface**。

---

# 2. 路线一：Scalar Discretization——把连续控制变成分类

最早、也最直接的思路，就是：

> **每一个连续 action scalar 都直接量化成一个离散类别。**

RT-1 是这类思路非常具有代表性的工作。

![RT-1 architecture](/images/action-tokenizer/rt1_full_model.png)

他们对于连续动作维度，将每个维度分别划分成固定数量的均匀 bins。

假设：

$$
\Delta x\in[-1,1]
$$

可以构造：

$$
Q(\Delta x)
=
\left\lfloor
\frac{\Delta x+1}{2}\times255
\right\rfloor
$$

于是：

```text
Δx = -0.42 → token 74
Δx =  0.03 → token 131
Δx =  0.81 → token 231
```

Transformer 不再预测一个连续数，而是预测一个离散类别$P(c_i|V,L)$ 

这把机器人动作预测从 regression problem 转成了 classification problem。

这个方法显而易见的优点就是:

1. 它非常简单。Tokenizer 本身无需学习，也不存在额外训练不稳定性。
2. 离散 categorical distribution 可以天然表达多模态动作。假设机器人既可以从物体左边绕过去，也可以从右边绕过去，连续均值回归可能给出一个不可执行的“平均动作”，而离散分布则可以给两个 mode 分配不同概率。

需要注意的是，RT-1 本身并不是后来典型的逐 action-token 自回归 VLA。这里更重要的是它确立了“把连续动作转成离散分类目标”的表示思路。

## 2.2 Scalar Tokenization 的问题：Action Chunk 一长，Token 就爆炸

假设机器人控制频率 50 Hz；action dimension 14；一次预测未来 1 秒。

那么：

$$
H=50,\quad D=14
$$

如果每个 scalar 一个 token：

$$
N_{\text{token}}=H\times D=700
$$

也就是说，为了描述未来一秒动作，自回归模型可能需要处理数百个动作 token。

更重要的是，高频动作具有极强时间冗余：

$$
0.120,\,
0.121,\,
0.122,\,
0.123,\,
0.124,\,
...
$$

量化后很可能得到：

```text
131 131 131 132 132 132 132 ...
```

于是：

$$
P(c_t|c_{t-1})
$$

可能远强于：

$$
P(c_t|V,L)
$$

最容易降低 loss 的策略变成了“复制前一个动作 token”，而不是根据当前视觉和语言推断行为。

所以问题开始改变如果一条轨迹本身高度平滑，为什么要逐采样点表示它？

---

# 3. FAST 把动作当作信号

FAST 的核心判断非常简单：

> **Robot actions are signals. Compress them before tokenizing them.**

机器人轨迹通常相对平滑。一秒钟采样 50 次，并不意味着真的存在 50 份独立信息。大量采样点只是同一条连续运动曲线的不同取样。

因此 FAST 把它看成一个需要压缩的多通道连续信号。

## 3.1 DCT → Quantization → BPE

FAST 的主流程可以写成：

$$
A
\xrightarrow{\text{DCT}}
C
\xrightarrow{\text{Quantize}}
\bar C
\xrightarrow{\text{Flatten}}
S
\xrightarrow{\text{BPE}}
T
$$

![FAST DCT tokenization pipeline](/images/action-tokenizer/fast_dct_method.png)

### DCT：把时间冗余变成频域稀疏性

FAST 对每一个 action dimension 做 DCT-II：

$$
C_j^i
=
\sum_{t=0}^{H-1}
a_t^i
\cos
\left[
\frac{\pi}{H}
\left(t+\frac12\right)j
\right]
$$

对于平滑轨迹，绝大多数信息集中在低频 coefficient：

$$
\text{trajectory}
\approx
\text{low-frequency structure}
+
\text{small high-frequency residual}
$$

因此 DCT 相当于把原本时域中的大量冗余重新组织成频域中的稀疏结构。

### Quantization：高频自然归零

对 coefficient 进行 scale-and-round：

$$
\bar C_j^i = \text{round}(\gamma C_j^i)
$$

大量高频 coefficient 会变成 0，因此 representation 变得非常稀疏。

### Frequency-first Flatten

FAST 按 frequency 维度优先排列：

```text
all dimensions at frequency 0
↓
all dimensions at frequency 1
↓
all dimensions at frequency 2
↓
...
```

这样序列前部天然更偏向整体轨迹结构，后部更多对应局部细节。

### BPE：继续压缩重复 coefficient pattern

量化后的 coefficient 序列仍然存在大量重复 pattern，因此 FAST 借用 NLP 中的 BPE，把高频出现的 coefficient 组合合并成单一 token。

最终，一条高频 action chunk 被压缩成远短于 $H\times D$ 的 token sequence。

## 3.2 FAST 真正解决的：Sampling Rate ≠ Information Rate

这条路线最关键的思想在于 高采样率只意味着我们更密集地观察同一条连续轨迹，并不意味着动作信息量同比增加。

而token的长度理论上来说应该和轨迹复杂度成正相关,从信息论的角度来看

![FAST main result](/images/action-tokenizer/fast_main_result.png)

## 3.3 FAST 的局限

FAST 仍然有明显的 hand-designed 成分。DCT basis 是人为规定的，而不是从机器人数据中学习出来的。

此外，BPE 是 variable-length coding。对于一个自回归 policy 来说，模型可能生成训练 tokenizer 从未产生过的 token 组合，因此“任意输出是否都能稳定 decode”也成为后续方法会进一步关注的问题。

---

# 4. 路线三：Learned Action Representation——从编码动作到设计监督

与 FAST 使用固定 DCT basis 不同，学习式 tokenizer 希望直接从机器人数据中学习：

$$
A
\xrightarrow{\text{Encoder}}
Z
\xrightarrow{\text{Quantizer}}
C
\xrightarrow{\text{Decoder}}
\hat A
$$

其中 $C$ 是离散 latent token。其实说白了就是VAE那一套让模型自己发现：跨时间的运动 pattern；不同 action dimension 的相关性；抓取、移动、旋转等动作结构；可能跨 embodiment 共享的行为模式。

但是一个传统 autoencoder tokenizer 通常优化：

$$
\mathcal L_{\text{rec}}
=
\|A-\hat A\|
$$

因此最自然的评价方式是 reconstruction error。但 VLA 真正要优化的是：

$$
P(C|V,L)
$$

而不是：

$$
P(A|C)
$$

所以“decoder 能不能从 token 恢复 action”和“VLA 能不能从图像语言预测这些 token”其实是两个不同问题。而decoder 能不能从 token 恢复 action其实相比LA 能不能从图像语言预测这些 token 更成熟,所以大部份工作都在前者的area做一些contribution, FAST 主要把 compressibility 拉进了 Action Tokenizer 的设计空间；ActionCodec 进一步把 learnability 拉进来；OAT 则显式把 autoregressive generatability 拉进来。

# 6. ActionCodec：什么样的 Token 更容易被 VLA 学会？

ActionCodec 的出发点非常关键：

> 现有 tokenizer 往往主要看 reconstruction fidelity，但 VLA 最终优化的是 token prediction，而不是动作 autoencoding。

于是 Action Tokenizer 被重新理解成：

## Supervision Generator

动作：

$$
A
$$

经过 tokenizer 后生成训练 label：

$$
C=\mathcal F(A)
$$

随后 VLA 优化：

$$
-\log P_\theta(C|V,L)
$$

因此 $\mathcal F$ 的性质会直接改变整个 VLA 的训练动力学。

ActionCodec 给出一个非常有解释力的信息论分解：

$$
H(C|V,L)
=
H(C|A)
+
I(C;A)
-
I(C;V,L)
$$

可以粗略理解成三部分：
* $H(C\|A)$：tokenizer 自身带来的不稳定性
* $I(C;A)$：token 中装了多少动作信息
* $I(C;V,L)$：这些 token 信息有多少能从视觉和语言中推断出来。

![ActionCodec design principles](/images/action-tokenizer/actioncodec_teaser.png)

## 6.1 相似动作应该产生稳定 Token

机器人动作空间本身是连续的。如果$A\approx A'$ 却得到$C\not\approx C'$

VLA 就必须拟合 tokenizer 人为制造出来的 label discontinuity。因此，一个好的 tokenizer 应尽量保留物理动作空间中的局部连续性。ActionCodec 用 Overlap Rate 一类指标衡量相邻 action chunk 的 token 稳定程度。

## 6.2 Token Capacity 不是越大越好

一个常见直觉是 codebook 越大、token 越多，信息容量越强，所以效果应该越好。但其实从 VLA 的角度看，Action Tokenizer 需要一个适度的信息瓶颈, 因为容量太大也会编码控制噪声；数据采集者特有操作习惯和视觉上不可知的高频 residual。

## 6.3 Vision-Language Alignment

传统 action autoencoder 只要求：

$$
A\rightarrow C\rightarrow\hat A
$$

但 VLA 真正做的是：

$$
V,L\rightarrow C
$$

如果 token 中大量信息根本无法从图像和语言中判断，那么这些信息对 reconstruction 有帮助，却会成为 VLA 的噪声监督。因此理想 tokenizer 应提高：

$$
I(C;V,L)
$$

也就是说，token 应尽可能与当前 perception 和 task semantics 对齐。

## 6.4 Token Independence

如果后续 token 强依赖前面的 action token：

```text
c1 → c2 → c3 → c4
```

autoregressive policy 就可能形成一个捷径：

```text
previous action tokens
        ↓
      next token
```

而不是：

```text
current image + language
        ↓
      next token
```

其实在RT 1我说的bin tokenizer也考虑了这个问题也就是所谓的shortcut,对闭环机器人控制来说，这种“动作续写”可能削弱策略对最新视觉反馈的响应。

![ActionCodec token dependency analysis](/images/action-tokenizer/actioncodec_token_independence.jpg)

因此 ActionCodec 强调降低不必要的 token dependency，让每个 token 更直接地受到 observation 和 language 的约束。

---

# 7. OAT：Learnability 之外，还有 Generatability

## 7.1 OAT 的三个要求

OAT 这篇paper希望一个 action tokenizer 同时满足：

1. **High Compression**
2. **Total Decodability**
3. **Left-to-right Causal Ordering**

其中第三点最关键,因为autoregressive Transformer 严格按照：

$$
P(T_1,\ldots,T_K)
=
P(T_1)
P(T_2|T_1)
\cdots
P(T_K|T_{<K})
$$

进行生成, 在Fast中也是为什么要把高频和低频重新排列的重要原因,token sequence 应从全局、粗粒度信息逐渐走向局部、细粒度信息。

```text
Token 1: global motion
Token 2: coarse correction
Token 3: more detail
...
Token K: fine residual
```

## 7.2 Nested Dropout 如何产生 Coarse-to-Fine Token？

OAT 的关键训练机制是 Nested Dropout。

假设完整 token sequence 为：

```text
T1 T2 T3 T4 T5 T6 T7 T8
```

训练时随机只保留 prefix：

```text
T1
T1 T2
T1 T2 T3
T1 T2 T3 T4
...
```

剩余位置用 mask 补齐，然后仍然要求 decoder 恢复完整动作：

$$
\mathcal L
=
\|D(T_{1:K},MASK)-A\|_2^2
$$

因为只有 $T_1$ 时也要尽量恢复动作，$T_1$ 被迫承载最重要的全局信息；加入 $T_2$ 后，最有效的策略是让它补充 $T_1$ 未编码的内容；后续 token 再逐步补 residual。

于是 representation 自然形成Progressive Action Representation

![OAT architecture](/images/action-tokenizer/oat_method.png)

## 7.3 Prefix Decoding：Token Budget 变成 Inference Budget

这种 coarse-to-fine representation 带来一个特别漂亮的性质：

任意 prefix 都可以解码。

![OAT prefix coarse-to-fine reconstruction](/images/action-tokenizer/oat_prefix_coarse_to_fine.png)

因此：

```text
1 token  → coarse but valid action
2 tokens → better action
4 tokens → higher fidelity
8 tokens → full fidelity
```

Action Tokenizer 第一次从静态 representation 进一步变成了Inference Budget Controller. 模型可以在 latency 和 action fidelity 之间进行动态折中。

进一步还可以考察 horizon 和 prefix 长度如何共同影响 representation：

![OAT horizon/prefix analysis](/images/action-tokenizer/oat_horizon_heatmap.png)

---

# 8. ActionCodec 和 OAT 矛盾吗？

把 ActionCodec 和 OAT 放在一起，会看到一个表面矛盾：

- ActionCodec：token 应尽量独立；
- OAT：token 应具有显式因果顺序。

但它们实际上讨论的是两个不同维度。

## 8.1 Independence 讨论 Dependency

ActionCodec 担心的是：

$$
I(c_k;c_{<k}|V,L)
$$

过高。也就是模型主要根据 previous action tokens 预测 next token，而不是根据当前 observation 和 instruction。它希望减少short cut

## 8.2 Ordering 讨论 Information Priority

OAT 关注的是：

$$
I(T_1;A)
>
I(T_2;A|T_1)
>
I(T_3;A|T_{1:2})
>\cdots
$$

也就是说信息按照重要程度排列：

$$
\text{global}
\rightarrow
\text{local}
$$

或：

$$
\text{coarse}
\rightarrow
\text{fine}
$$

一个理想 tokenizer 完全可能同时做到：

- 每个 token 都直接与视觉语言输入有关；
- token 间不形成过强的“动作语言捷径”；
- 整体信息又按照 coarse-to-fine 顺序排列。

# 9. 用统一坐标评价 Action Tokenizer

从这几条路线共同得到的经验来看，单独汇报 reconstruction error 已经远远不够。

一个比较完整的 Action Tokenizer evaluation card 至少应该包括：

| 维度 | 说明 / 核心量 |
|---|---|
| **Reconstruction Fidelity** | $E_{\text{rec}}=\\\|A-D(E(A))\\\|$ |
| **Token Efficiency** | $R=\frac{H\times D}{K}$ |
| **Temporal Stability** | 相似动作是否得到相似 / 高重叠 token？ |
| **Vocabulary Efficiency** | 需要看 codebook utilization、entropy、rare tokens 和 redundancy。 |
| **Vision-Language Predictability** | 核心量可以抽象为 $H(C\|V,L)$，即给定图像和指令后，动作 token 到底有多难预测。 |
| **Token Dependency** | 模型是在看当前视觉做动作，还是主要在续写前面的 action token？ |
| **Autoregressive Ordering** | 是否存在稳定的 coarse-to-fine 信息顺序？ |
| **Total Decodability** | 任意合法 token sequence 是否都能得到合法 action？ |
| **Inference Flexibility** | 部署时能否根据场景动态改变 token budget？ |

---

# 参考工作

- RT-1: Robotics Transformer for Real-World Control at Scale
- FAST: Efficient Action Tokenization for Vision-Language-Action Models
- ActionCodec: What Makes for Good Action Tokenizers
- OAT: Ordered Action Tokenization
