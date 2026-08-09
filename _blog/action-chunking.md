---
title: "浅谈具身智能中的 Action Chunking"
date: 2026-08-10
excerpt: "从为什么要一次预测多个动作，到应该预测多长，再到到底执行多长，梳理 Action Chunking 在具身智能中的机制与设计取舍。"
cover: "/images/action-chunking/01_why_action_chunking.png"
categories:
  - Embodied-AI
  - Robotics
tags:
  - Action-Chunking
  - VLA
  - Robot-Learning
  - Behavioral-Cloning
  - Long-Horizon-Control
math: true
read_time: true
---

# 浅谈具身智能中的 Action Chunking

> 从“为什么要一次预测多个动作”，到“应该预测多长”，再到“到底执行多长”。

如今几乎所有的vla都采取action chunk的推理和控制,但是其实这一直是个玄学，如果只从经典闭环控制的直觉出发，Action Chunking 其实是一件有些反常识的事。

机器人在时刻 $t$ 得到观测 $o_t$，最自然的做法似乎应该是预测一个动作 $a_t$，执行之后立刻得到新的观测 $o_{t+1}$，再预测 $a_{t+1}$。也就是说，策略始终工作在最高反馈频率下：

$$
o_t \rightarrow a_t \rightarrow o_{t+1} \rightarrow a_{t+1}\rightarrow\cdots
$$

既然每一个时间步都可以重新观察、重新计算，为什么现代机器人学习系统反而越来越喜欢一次预测几十步动作？

从 ACT、Diffusion Policy 到今天大量连续控制 VLA，常见的接口已经变成：

$$
\pi(o_t,\ell)\rightarrow
\mathbf A_t=(a_t,a_{t+1},\ldots,a_{t+H-1}),
$$

即策略不是给出“下一步该怎么动”，而是直接给出一小段未来轨迹，也就是 **Action Chunk**。

Action Chunking 的优势长期以来主要来自经验：它往往让行为克隆更稳定，让扩散或 flow-based policy 更容易得到连贯动作，也常常能显著提高长程操作任务的成功率。但“为什么一口气预测一段动作会比一步一步预测更好”，直到最近仍缺乏令人满意的解释。

本文围绕三篇工作展开：

- *Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?*
- *Mixture of Horizons in Action Chunking*
- *PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking*

我更愿意把它们看成对同一个问题的三层追问：

$$
\boxed{\text{Why chunk?}}
\rightarrow
\boxed{\text{How long to predict?}}
\rightarrow
\boxed{\text{How long to execute?}}
$$

第一篇试图解释 Action Chunking 为什么有效；第二篇进一步追问 chunk 的时间跨度应该是多少；第三篇则指出，一个经常被混淆的问题是：**预测多长和执行多长，其实根本不是一回事。**

---

## 1. 从 Single-step Policy 到 Action Chunk

一个最普通的单步策略可以写成：

$$
\pi_\theta(a_t\mid o_t,\ell),
$$

其中 $o_t$ 是当前视觉、本体感知等观测，$\ell$ 是语言指令，策略只负责预测当前动作 $a_t$。Action Chunking 则把输出改成一段动作：

$$
\pi_\theta(\mathbf A_t\mid o_t,\ell),\qquad
\mathbf A_t=(a_t,a_{t+1},\ldots,a_{t+H_p-1}).
$$

我们规定记号

$$
H_p=\text{Prediction Horizon},
$$

表示模型一次预测多远

$$
H_e=\text{Execution Horizon},
$$

表示机器人在重新获取观测、重新调用策略之前，实际执行这个 chunk 的多少步。大部份工作和业界基本上在$\pi$ 上都采取

$$
H_p=50,\qquad H_e=10.
$$

执行前 10 步之后，剩下的 40 步全部丢弃，机器人重新观察环境，再生成一个新的 50-step chunk：

$$
o_t
\overset{\pi}{\longrightarrow}
a_{t:t+49}
\overset{\text{execute 10}}{\longrightarrow}
o_{t+10}
\overset{\pi}{\longrightarrow}
a'_{t+10:t+59}.
$$

这其实已经非常接近经典的 **Receding Horizon Control（滚动时域控制）**:每次规划一段未来，但只执行其中一个前缀，然后利用新的反馈不断重规划。

---

## 2. 第一个问题：既然每一步都能重规划，为什么还需要 Action Chunk？

Action Chunking 最早给人的解释通常非常直观。

第一种解释是 **temporal consistency**：机器人动作具有很强的时间相关性，把连续动作一起预测，自然比独立预测每一步更平滑。

第二种解释是 **horizon reduction**：如果一次生成并执行 $k$ 步动作，那么一个长度为 $T$ 的任务只需要大约 $T/k$ 次策略决策，行为克隆的 compounding error 因而减轻。

第三种解释是 **representation learning**：预测未来多个动作本身就是更丰富的监督信号，迫使视觉或状态表示编码更多与运动相关的信息。

2026 年的 *Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?* 做了一件非常值得注意的事情：作者没有再提出一种新的 chunk 架构，而是反过来把 Action Chunking 拆开，问——**到底是哪一部分真正产生了收益？**

![Action Chunking 的机制分解](/images/action-chunking/01_why_action_chunking.png)

*图 1：论文对 Action Chunking 的机制分解。传统的 temporal consistency、horizon reduction 和 representation learning 并不足以完整解释性能提升；作者最终将主要收益归因于 non-Markovian expressivity、reduced compounding error 和 implicit ensembling。

具体的action chunk的归因如下

**1. Action Chunk 首先获得了更强的非马尔可夫表达能力** 

单步策略学习的是：

$$
a_t\mid o_t.
$$

它隐含了一个很强的假设：当前观测 $o_t$ 足以确定当前动作。但真实机器人示范并不总是满足这个假设。例如机械臂到达抽屉把手附近之后，人类操作者可能会有一个很短的停顿，再进入拉动阶段。视觉上，停顿前后的状态几乎没有变化，但“此刻继续等待”还是“开始拉动”却依赖于前面的运动历史。类似的现象在 grasp、contact、release、phase transition 处非常常见。也就是说，专家行为可能具有明显的 non-Markovian structure。

Action Chunking 在训练时并不仅仅学习

$$
a_t\mid o_t,
$$

同一个动作 $a_t$ 还会作为过去不同观测所对应 chunk 的未来元素出现：

$$
a_t\mid o_{t-1},\quad
a_t\mid o_{t-2},\quad
\ldots
$$

所以从另一个角度看，一个长度为 $k$ 的 chunk policy 同时学习了多种“观测延迟—动作”的时间关系。这也是论文为什么引入 delayed policy：

$$
a_t\sim\pi_{\text{delay}}^d(a_t\mid o_{t-d}).
$$

一个很反直觉的结果是，在 LIBERO-90 上，单纯使用过去观测预测当前动作的 Delay policy 就已经能够获得很强的性能：

| 方法 | LIBERO-90 Success Rate |
|---|---:|
| Markovian / Single-step | 68.9% |
| Action Chunking | 89.2% |
| Delayed Policy | **94.0%** |
| Randomized Delay Ensemble | 93.6% |

这意味着，Action Chunking 的一部分优势并不来自“连续动作一定要一起执行”，而来自它可以利用**过去更稳定、更接近专家分布的观测来预测未来动作**。

**2.它降低了 Compounding Error，但原因不只是“少决策几次”** 

行为克隆最经典的问题之一就是 distribution shift。 

训练时模型看到的状态来自专家：

$$
o_t\sim d_{\text{expert}},
$$

部署时却来自自己的历史动作：

$$
o_t\sim d_{\pi}.
$$

早期一个很小的动作误差会改变之后的状态，之后的状态又不在训练分布里，于是误差继续放大。过去常见的说法是：Action Chunking 一次执行多步，所以决策次数减少，等效 horizon 变短，从而减少 compounding error。

但是其实没有这么简单。Delayed policy 即使仍然每一步都重新产生一个动作，也可以获得和 Action Chunking 类似的 compounding-error 改善。原因是它在时刻 $t$ 使用的是更早的 $o_{t-d}$。更早的观测受到 policy 自身累积误差的污染更少，因此往往更接近训练数据分布。换句话说，真正有意义的不只是**少做几次决策**而是**当前动作可以由更早,更 in-distribution 的信息预测**. 这也是为什么“只执行 chunk 的前几步”依然可能比纯 single-step policy 更好：模型训练时已经学习了更加丰富的跨时间预测关系。

**3.这篇论文 最有意思的解释：Action Chunking 是一种隐式集成**

如果只靠 delayed policy 就能解释一切，那么 Action Chunking 本身似乎也没那么神秘。但作者在 Robomimic 上发现，单一 delay 并不能始终追上 chunk policy。例如 Tool Hang 上，Delay 只有 51.6%，而 Action Chunking 达到 75.2%。

差距来自哪里？作者给出的解释是 **implicit ensembling**。

一个长度为 $k$ 的 Action Chunking policy，会同时学习：

$$
a_t\mid o_t,\;
a_t\mid o_{t-1},\;
a_t\mid o_{t-2},\ldots,
a_t\mid o_{t-k+1}.
$$

可以把它理解成一组共享参数、拥有不同 temporal delay 的预测器：

$$
\{\pi_0,\pi_1,\ldots,\pi_{k-1}\}.
$$

它们使用的时间上下文并不完全相同，因此错误也不完全相关。把这些预测关系放在同一个网络中学习，相当于天然形成了一种 temporal ensemble。论文进一步构造 Randomized Delay Ensemble：每一步随机选择一个 delay 关系来产生单步动作。结果显示，即使不真正执行 Action Chunk，也能在很多环境里复现其性能优势。显式 ensemble 甚至还能进一步超过普通 Action Chunking。例如 Robomimic Transport 上：

$$
12.6\%\rightarrow41.5\%.
$$

这说明 Action Chunking 的价值并不能简单概括成“让轨迹更平滑”。它其实改变了学习问题本身：同一个动作被放进了多个不同的时序预测关系中。

### 那么到 VLA 上，这个现象还成立吗？

这点很关键，因为上述机制如果只在小型 diffusion policy 上成立，对今天的 VLA 意义有限。论文专门在 $\pi_{0.5}$ 的 LIBERO 微调模型上做了验证。以 LIBERO-10 为例：

$$
\text{Markovian}=84.0\%,\qquad
\text{AC(10)}=93.8\%,
$$

而 randomized-delay deployment 同样达到：

$$
93.8\%.
$$

在 Spatial、Goal、Object 等套件上也观察到了类似兼容性。

真机结果同样支持 delayed relation 和 randomized ensemble 的解释。

![真机实验中不同策略的对比](/images/action-chunking/02_realworld_chunking.png)

*图 2：真机任务中的 Action Chunking、Delay 与随机延迟集成结果。核心现象是：Action Chunking 的不少优势可以在“不执行完整动作块”的情况下通过不同 temporal relation 复现。*

我觉得还可以换一个更直观的角度理解。

Single-step supervision：

$$
o_t\rightarrow a_t
$$

只告诉模型一件事：**现在应该怎么动。**

而 chunk supervision：

$$
o_t\rightarrow
(a_t,a_{t+1},\ldots,a_{t+H-1})
$$

实际上同时告诉模型：**当前动作属于怎样的一段运动。**

一个 action 本身只是轨迹上的一个点；一段 action sequence 却携带了方向、速度变化、接近、接触、抓取、运输、释放等结构。

因此，我更愿意把 Action Chunking 理解成一种 **trajectory-level supervision**。

它不是简单地把一个 label 变成 $H$ 个 label，而是在向网络提供“当前时刻之后的局部未来”。模型为了把未来几十步一起预测出来，必须在内部判断当前处于什么运动阶段、往哪个方向演化、距离下一个 phase transition 还有多远。

---

## 3. 第二个疑问：既然 Chunk 有效，那么是不是越长越好？

如果 Action Chunking 的优势来自更丰富的 temporal structure，一个很自然的推论似乎是：

$$
H_p\text{ 越大越好}.
$$

但实验并不支持这个结论。

*Mixture of Horizons in Action Chunking* 的出发点就是：现代 VLA 往往直接固定一个 chunk horizon，但不同 horizon 实际上对应不同的 temporal bias。

![不同 horizon 在 π0 上表现出明显权衡](/images/action-chunking/03_horizon_tradeoff.png)

*图 3：不同 prediction horizon 在 $\pi_0$ 上的表现。较长 horizon 在 Long 套件上更有优势，但固定单一 horizon 无法同时取得最好的局部精度和长期预见；Mixture of Horizons 在多个套件上进一步提升。*

### 3.1 长 Horizon 和短 Horizon 学到的其实不是同一种东西

短 horizon 的预测问题更容易：

$$
o_t\rightarrow a_{t:t+h_s}.
$$

未来比较近，环境不确定性小，动作监督和当前观测强相关，因此它通常更擅长 fine-grained local control。这对于 grasp alignment、contact、插入、release 等阶段尤其重要。

长 horizon 则要求：

$$
o_t\rightarrow a_{t:t+h_l},\qquad h_l\gg h_s.
$$

它必须看到更长的轨迹演化，因此拥有更强的 global foresight，尤其容易改善 long-horizon manipulation。但代价是，远期动作本身更难预测。未来越远，环境扰动、视觉变化、接触误差都会逐渐增加，模型对每一个具体动作的精度会下降。

于是出现了一个非常自然的 trade-off：

$$
\boxed{\text{Long horizon: foresight }\uparrow,\; \text{local precision }\downarrow}
$$

$$
\boxed{\text{Short horizon: local precision }\uparrow,\; \text{foresight }\downarrow}
$$

如果机器人任务本身只存在一种时间尺度，选一个固定 $H_p$ 当然没有问题。

但真实 manipulation 明显不是这样。

一个 pick-and-place 可能依次经历：

$$
\text{approach}
\rightarrow
\text{pre-grasp alignment}
\rightarrow
\text{contact}
\rightarrow
\text{grasp}
\rightarrow
\text{transport}
\rightarrow
\text{placement}.
$$

其中 transport 可以是几十步的平滑运动，而 contact 可能在两三步之内就决定成功或失败。因此，“整个任务都应该使用同一个最佳 horizon”本身就是一个很强、也很可疑的假设。

### 3.2 Mixture of Horizons：为什么一定要选一个 Horizon？

MoH 的思路很直接：既然不同 horizon 各自擅长不同时间尺度，就让模型同时学习多个 horizon。假设最大长度为 $H$，构造一组：

$$
\mathcal H=\{h_1,h_2,\ldots,h_N\}.
$$

例如：

$$
\mathcal H=\{3,6,9,\ldots,30\}.
$$

同一段 ground-truth trajectory 会形成不同长度的训练目标：

$$
A_t^{(3)},A_t^{(6)},\ldots,A_t^{(30)}.
$$

这些不同 horizon 共享同一个 action transformer，并行计算，再由一个很轻量的 gating head 逐时间步融合：

$$
\hat a_{t,k}
=
\sum_{h\in\mathcal H,\;k\le h}
\alpha_{t,k,h}\hat a_{t,k}^{(h)}.
$$

直觉上，相当于让同一个 policy 内部同时保留“近视”和“远视”两种能力。

![Mixture of Horizons 架构](/images/action-chunking/04_moh_architecture.png)

*图 4：MoH 的整体架构。多个不同 horizon 的 action segment 共享同一个 action transformer，并通过轻量 gating 按时间步融合。*

实验结果并不是巨大到改变整个 benchmark 格局，但非常稳定。

在 $\pi_{0.5}$ 上，固定 $H=30$ 时 LIBERO 平均成功率为：

$$
97.7\%.
$$

加入 $\{3,6,\ldots,30\}$ 的 MoH 后：

$$
99.0\%.
$$

其中 LIBERO-Long：

$$
95.4\%\rightarrow98.4\%.
$$

在 $\pi_0$ 上平均从 93.8% 提升到 95.1%；在 regression policy $\pi_{\text{reg}}$ 上从 95.2% 提升到 96.4%。在 RoboCasa 的 GR00T 设置下，平均成功率也从 28.0% 提升到 31.4%。

这一点很重要：**MoH 的收益并不是 diffusion/flow matching 特有的现象。**

也就是说，horizon 本身已经可以被看成一种 temporal representation，而不是某种生成模型专属的采样技巧。

### 3.3 多 Horizon 的真正价值：Temporal Diversity

实验的消融显示如果只是复制十份相同的 $H=30$ predictor，性能只从 97.7% 到 97.9%；而真正使用异质 horizon，才能到 99.0%。

这说明收益并不是“多几个头做 ensemble”这么简单，而来自：

$$
\boxed{\text{不同时间尺度之间的异质性}}
$$

这和第一篇论文的 implicit ensembling 其实有一个很有趣的呼应。

第一篇告诉我们：普通 Action Chunking 已经通过

$$
a_t|o_t,\;
a_t|o_{t-1},\;
a_t|o_{t-2},\ldots
$$

引入了不同 temporal delay 的预测关系。MoH 又进一步把“时间尺度的多样性”显式化，让同一个模型同时学习短期和长期的 trajectory structure。

### 3.4 Horizon Consensus 又把问题推向了执行端

MoH 还有一个容易被忽略的设计：动态推理。

如果多个 horizon 对前几步动作预测高度一致，说明这部分未来比较稳定，可以多执行一些；如果不同 horizon 很快产生分歧，则意味着附近可能出现决策点或精细操作，应该更早重新观察。

![MoH 的动态推理](/images/action-chunking/05_moh_dynamic_inference.png)

*图 5：MoH 的动态推理示例。不同 horizon 对动作预测的共识高时执行更长前缀；出现明显分歧时提前停止并重新规划。*

到这里，一个非常重要的概念开始浮现出来：**预测 horizon 和执行 horizon 应该被分开设计。**

---

## 4. 第三个疑问：预测了 50 步，为什么一定要执行 50 步？

现在的action chunk基本都是预测50步然后走10步然后重新预测,大家都把这个当作行业共识,但是why?没人研究，

PACE 最重要的贡献之一，就是把这个长期被当作实现细节的 **execution horizon** 单独拿出来研究。

![PACE 总览](/images/action-chunking/06_pace_overview.png)

*图 6：PACE 总览。左上展示固定 execution horizon 的成功率高度非单调且任务相关；中间把预测 chunk 截断为动态选择的执行前缀；右侧和下方展示 RoboTwin2.0、真机以及单条 rollout 中的自适应执行。*

### 4.1 固定 Execution Horizon 是一个相当脆弱的假设

PACE 对同一个 $\pi_{0.5}$ checkpoint 扫描不同 execution horizon，发现成功率对 $H_e$ 的关系不是简单的“越短越闭环，所以越好”，也不是“越长越连贯，所以越好”。

它往往是明显非单调的，而且任务差异很大。某些任务喜欢很短的执行段；某些任务喜欢较长 chunk；甚至同一个任务的曲线也可能出现多个峰值。

这其实非常符合 manipulation 的物理直觉。如果当前是 free-space transport，动作基本沿着一个平滑方向前进，那么每 1 步都重新调用一次大模型没有太大意义，反而可能因为模型噪声、推理延迟或 mode switching 破坏连贯性。但如果机械臂正接近 contact，或者处于插入、抓取、堆叠对齐等敏感阶段，执行几十步 open-loop 又显然危险。

所以真正的问题不应该是：“最佳固定 $H_e$ 是多少？”而应该是：**“当前这个 chunk，在哪里最适合重新规划？”**

### 4.2 PACE：把重规划点放到运动相位的边界

PACE 将执行步长写成：

$$
h_i=g(\mathbf A_i).
$$

也就是说，每一次 policy query 得到一个 action chunk 后，不直接使用固定长度，而是根据这个 chunk 自身决定执行多少步。其核心先验非常朴素：机器人操作轨迹往往由多个局部连贯的 motion phase 组成，而 phase transition 附近通常会出现减速。例如：

$$
\text{approach}
\rightarrow
\underbrace{\text{slow down}}_{\text{phase boundary}}
\rightarrow
\text{grasp}.
$$

PACE 从预测动作计算速度 profile，做平滑，再寻找显著的 low-speed valley，把它视作候选重规划边界。如果检测到了合适的 valley，就执行到那里；否则才执行到最大允许长度。

于是部署过程变成：

$$
o_{\tau_i}
\rightarrow
\mathbf A_i
\rightarrow
h_i
\rightarrow
a_{i,1:h_i}
\rightarrow
o_{\tau_i+h_i}
\rightarrow\cdots
$$

而不是固定：

$$
h_i=H,\quad\forall i.
$$

在 RoboTwin2.0 的 50 个任务上，同一批 $\pi_{0.5}$ checkpoint：

$$
H_e=5:48.8\%,
$$

$$
H_e=25:57.8\%,
$$

$$
H_e=50:53.4\%,
$$

而 PACE 达到：

$$
64.2\%.
$$

真机上，平均成功率从全长执行基线的 50.7% 提升到 70.4%。

更重要的是，即使把固定 baseline 的平均执行长度调到和 PACE 相似，PACE 依然更好。这说明收益并不只是“平均多查询了几次”，而是：

$$
\boxed{\text{在正确的位置重新查询}}
$$

比单纯提高查询频率更重要。

![PACE 在单条 rollout 中动态改变执行长度](/images/action-chunking/08_pace_adaptive_rollout.png)

*图 7：PACE 在同一条 rollout 内会选择明显不同的执行前缀。平滑运动阶段执行得长，接近相位过渡时执行得短。*

---

## 5.预测得更远，即使不执行，仍然有价值

PACE 有一个消融，作者固定 deployment 时真正执行的长度 $H_{\text{eval}}$，只改变训练时 prediction horizon $H_{\text{train}}$。

当：

$$
H_{\text{eval}}=10
$$

时，如果训练也只预测 10 步，成功率是：

$$
57.2\%.
$$

如果训练时让同一个策略预测 50 步：

$$
H_{\text{train}}=50,
$$

但部署时仍然只执行最前面的 10 步，成功率变成：

$$
77.9\%.
$$

也就是：

$$
\boxed{57.2\%\rightarrow77.9\%}
$$

提升 **20.7 个百分点**。

注意这里最反直觉的地方：

> 后面 40 个动作在部署时根本没有执行。

它们只是训练 target。

![更长的训练 prediction horizon 能改善被执行的短前缀](/images/action-chunking/07_pace_training_horizon.png)

*图 8：PACE 的 training-horizon 消融。固定 execution horizon 后，增大训练时 prediction horizon 仍能持续改善成功率；当 $H_{\text{eval}}=10$ 时，从训练 10 步提升到训练 50 步带来 +20.7 个百分点。*

这个结果对“为什么不直接预测下一步”提供了一个非常直接的回答。

因为 Action Chunk 的作用并不仅发生在 open-loop execution 阶段。

即使你最后只执行前几步，要求网络联合预测更长的未来，本身就会改变网络学到的表示和动作分布。

这和前面“trajectory-level supervision”的理解几乎完美对应:被丢弃的 future actions 并不是无用预测，它们构成了当前动作的 temporal context。 所以 prediction horizon 和 execution horizon 的最优设计很可能天然不同.

---

## 6. Action Chunk 也许不应该被理解成“多个动作”，而应该被理解成 Local Motor Plan

这是我读完这三篇之后最强烈的感受。从张量形式上看，Action Chunk 只是：

$$
\mathbf A_t\in\mathbb R^{H\times d_a}.
$$

但从功能上看，它已经越来越接近一个 **Local Motor Plan**。

传统 hierarchical robotics 往往会显式区分：

$$
\text{Task Planning}
\rightarrow
\text{Motion Planning}
\rightarrow
\text{Control}.
$$

今天的 VLA 把很多层级压进了一个神经网络里，但层级本身并没有真正消失。

语言和视觉输入负责描述任务和场景；Action Chunk 则给出接下来一小段可执行的局部运动意图；更低层的 execution mechanism 再决定如何利用最新反馈安全地消费这个 plan。

在这个视角下，Action Chunking 的意义就不仅是“让 diffusion 更容易收敛”或者“提高推理吞吐”。

如果要把上面的讨论落实到今天的连续控制 VLA，我觉得**固定单一 prediction horizon 可能不是最终形态**。MoH 的结果说明，机器人任务天然是 multi-timescale 的。与其寻找一个全局最优 $H_p$，不如让模型同时拥有不同时间尺度的预测能力，或者最终直接学习状态依赖的 horizon。**Execution Horizon 最好是内容感知的**。PACE 用速度 valley 作为 phase boundary；MoH 用跨 horizon disagreement 作为置信信号。它们看似不同，背后的思想其实一致：应该根据当前预测的结构决定什么时候需要反馈，而不是机械地每隔 $N$ 步重规划。**评估 Action Chunking 时应该进行二维而不是一维消融**。很多工作只报告 chunk length，但真正需要扫的是：

$$
(H_p,H_e).
$$

更进一步，还应该分别研究 training horizon 和 inference prediction horizon。否则“chunk=10 比 chunk=50 好”这样的结论很容易把学习和控制两个因素混在一起。

---

## 9. 还有哪些问题没有解决？

尽管这三篇工作已经把 Action Chunking 拆得比以前清楚很多，但仍然有一些关键问题没有真正回答。PACE 使用运动速度中的低速谷寻找 phase boundary，这是一种非常漂亮的 hand-crafted prior，但未来的 VLA 也许应该自己预测：

$$
p(\text{replan}\mid o_t,\mathbf A_t),
$$

甚至直接预测 duration、termination、contact likelihood 或 uncertainty。

这时 Action Chunk 就不再是固定长度矩阵，而可能变成：

$$
(\mathbf A_t,\tau_t,c_t),
$$

其中 $\tau_t$ 表示这段 local plan 的有效持续时间，$c_t$ 表示模型对其可靠性的估计。

最后，今天绝大多数讨论仍然建立在 imitation learning 上。强化学习中，action repeat、temporal abstraction、options 与 chunked policy 之间又有一套不同的 credit assignment 和 exploration 问题。Action Chunking 在 RL 中往往不是“没有就不工作”，但它为什么经常仍能提高稳定性，可能需要另一套解释。

---

## 主要参考论文

1. Filippo Lazzati, Kyle Stachowicz, William Chen, Alberto Maria Metelli, Andrew Wagenmaker, Sergey Levine. **Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?** arXiv:2608.02547, 2026.
2. Dong Jing, Gang Wang, Jiaqi Liu, et al. **Mixture of Horizons in Action Chunking.** arXiv:2511.19433, ICML 2026.
3. Junnan Nie, Jiayi Li, Jiachen Zhang, et al. **PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking.** arXiv:2606.00537, 2026.
