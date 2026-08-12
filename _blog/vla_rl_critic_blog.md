---
title: "从 Reward 到 World Critic：VLA 强化学习为什么重新需要 Value Function？"
date: 2026-08-12
excerpt: "从 AWR、IQL、RECAP 到 ViVa、ROVE、WCM，梳理 VLA-RL 中 Critic 的演进：value model 如何一步步变成 world critic，以及 Progress Model 与 Critic-Free RL 的路线之争。"
cover: "/images/vla-rl-critic/recap_architecture.png"
categories:
  - Embodied-AI
  - Robotics
tags:
  - VLA
  - RL
  - Critic
  - Value-Function
  - Robot-Learning
  - World-Model
math: true
read_time: true
---

# 从 Reward 到 World Critic：VLA 强化学习为什么重新需要 Value Function？

> **副标题：从 AWR、IQL、RECAP 到 ViVa、ROVE、WCM，以及 Progress Model 与 Critic-Free RL 的路线之争**

近年来, VLA-RL 通常烦恼与没有dense reward, 基本上奖励呈现高度二分化,成功和失败,所以大家开始研究critic,目的是**以真实 outcome 为 anchor、把这个 anchor 向前传播到中间步骤**,过去一年,经历了从 scalar return regressor 演化成 temporal model、predictive model，最后到 World Critic的变化,本文的目的是疏离今年的critic model的具体演进,并且思考未来的方向。现在的路线主要聚焦于两

* Progress Model（提供更 dense 的语义监督）
*  Critic-free outcome RL（干脆把 Critic 删掉）

**术语约定：** 本文中，**outcome** 指 episode 结束时来自环境的 success / failure；**progress** 指对“任务完成到什么程度”的估计；**credit** 指把 outcome 或 progress 分配到具体 state–action 上的过程。**Critic** 泛指任何输出 $V$、$Q$ 或 advantage 的模型，不限于经典 actor-critic 里的那一个网络。

---

## 引言：谁来告诉 VLA，它刚才做得好不好？

VLA（Vision-Language-Action）已经把机器人学习带到了一个很有意思的位置：模型终于可以“看懂”视觉、理解自然语言，再直接输出连续动作。但当我们希望它在部署中继续变强时，一个旧问题重新浮出水面,**谁来告诉 VLA，它刚才做得好不好？**

如果任务只是“按一下按钮”，这个问题并不难：执行结束后给一个 success / failure 就够了。但当任务变成折衣服、装箱、穿鞋带、擦白板、操作咖啡机，整个 episode 可能持续几十秒甚至几分钟，包含几十到数百个动作。最后那一个 success 当然重要，却几乎没有告诉我们：

- 第 17 步的 grasp 是不是关键的正向动作？
- 第 31 步的重新抓取，是 recovery，还是无谓的反复？
- 第 52 步短暂远离目标，是错误，还是为后续操作做准备？
- 一段人类 intervention 里，到底哪些动作值得模仿，哪些只是接管瞬间的犹豫？

这就是 VLA-RL 里最核心、也最容易被“用了 PPO / 用了 offline RL”这类算法名掩盖的问题.

---

## 主要矛盾：越可信的 reward 越稀疏，越 dense 的 reward 越容易有 bias

机器人学习里的监督信号存在一个非常根本的张力。

**最可信的是 environment outcome：**

$$
r_T\in\{0,1\}.
$$

例如：鞋带最终有没有穿进正确鞋孔？衣服有没有折好？咖啡有没有做完？面包有没有成功放进烤面包机？ 完成了就是完成了，没有就是没有。但它的问题是**极端稀疏**,只有成功和失败

**另一端是 VLM / Progress Model：**

$$
P(o_t,g)=0.72.
$$

它可以每一步都给一个 dense score，却引入另一类风险：这个 0.72 到底有没有可校准的物理意义？视觉上相似的两个状态会不会被打成相似分数，而实际上一个在进步、一个在滑落？policy 会不会学会 exploit reward model 的视觉 shortcut？同一任务中“打开抽屉”和“抓到杯子”，究竟哪个应该更高 progress？这个序关系由谁定义？于是一个很自然的折中是Outcome 是 anchor，Critic 负责传播

Critic 不要求任何人去人为定义“现在完成了 63%”，而是从真实回报中学习：

$$
V^\pi(s_t)
=
\mathbb E_\pi[G_t\mid s_t],
$$

然后得到：

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

也就是说，我们重点关注 从当前状态继续执行，未来结果有多好？这个动作相对于当前状态下的平均行为，到底好多少？

这就是 Critic-based VLA-RL 的出发点。本文剩下的部分，基本都是在回答这个出发点带来的三个后续问题：**Critic 怎么接进 VLA（Part I–II）、Critic 凭什么相信自己看到了 state（Part III–V）、以及我们该不该完全依赖它（Part VI 及之后）。**

---

# Part I：Critic 从哪里来？

## 1. 从经典 RL 到 VLA：$V$、$Q$、$A$ 分别解决什么问题？

传统 RL 里三个最重要的量，在 VLA 场景下可以用非常直观的语言重新解释。需要先声明一句：下面这些定义都建立在 MDP 假设之上，而真实机器人是 POMDP, 这个裂缝会在 Part III 变成整条路线的主要矛盾。

### 1.1 State Value：从这里继续，前景如何？

$$
V^\pi(s)
=
\mathbb E_\pi[G_t\mid s_t=s].
$$

机器人在当前状态下，如果继续按照策略 $\pi$ 执行，预期未来 return 是多少？

对于一个 long-horizon manipulation task，它可以近似理解为：**当前离“成功”还有多远？**这也是为什么 VLA 文献里 value 和 progress 会频繁互相靠近。RECAP 的 value function 就被训练成能够反映 expected time-to-success，并展示出明显的 task-progress 曲线。[4]

但要注意一个容易被忽略的细节：$V^\pi$ 是**对某个特定策略 $\pi$ 的**估计。同一个状态，对一个熟练策略而言是“还差三步”，对一个笨拙策略而言可能是“大概率失败”。Critic 从来不是对世界的客观描述，它永远绑定着一个隐含的行为分布. 这一点在后面讨论 ROVE 的 optimistic value 和 bootstrap error 时都会回来。

### 1.2 Action Value：从这里执行这个动作，前景如何？

$$
Q^\pi(s,a)
=
\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

和 $V(s)$ 相比，它多回答了一件关键的事：**不是“这个状态好不好”，而是“这个动作好不好”。** 这对 VLA 很关键，因为 VLA 最终要优化的对象不是 state，而是 action，通常甚至是 action chunk：

$$
a_{t:t+H}.
$$

### 1.3 Advantage：这个动作比平均行为好多少？

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
$$

Advantage 的含义可以直接翻译成 policy update：

$$
A>0
\quad\Rightarrow\quad
\text{更应该模仿 / 提高这个动作的概率};
$$

$$
A<0
\quad\Rightarrow\quad
\text{少模仿 / 降低这个动作的概率}.
$$

对于 VLA，这个定义有一个非常重要的工程意义：**Critic 不必直接控制机器人；它只需要告诉一个已经很强的生成式 policy，“哪些动作更值得学”。**

---

## 2. AWR → IQL：为什么 advantage-weighted imitation 天然适合大型 VLA？

如果从算法史往前看，今天 VLA-RL 的很多设计并不是凭空出现的。

### 2.1 AWR：把 RL 改写成 weighted supervised learning

AWR（Advantage-Weighted Regression）最重要的思想，他把 policy improvement 改写成非常熟悉的 maximum-likelihood / regression 问题。[2]

先得到 advantage $A(s,a)$，然后定义权重：

$$
w(s,a)
=
\exp\left(\frac{A(s,a)}{\beta}\right),
$$

最后优化：

$$
\mathcal L_\pi
=
-\,
w(s,a)\log \pi(a\mid s).
$$

这意味着：

- 好动作：梯度更大；
- 一般动作：正常学习；
- 差动作：影响被自动衰减。

**RL 变成了 weighted imitation。**

这个形式本质上是“在 KL 约束下最大化期望回报”这一优化问题的闭式解，$\beta$ 就是那个约束的温度：$\beta$ 越小越激进（更接近只学最好的动作，方差更大），越大越保守（更接近纯 BC）。在实际 VLA 训练里，$\beta$ 往往比 RL 算法的选择更影响结果。大型 VLA 原本最擅长的就是稳定的大规模 supervised training，而不是把一个几十亿参数的生成式 action model 硬塞进传统 actor-critic policy gradient 里。

### 2.2 IQL：不要为了“改进策略”而去评价大量 OOD action

Offline RL 的另一个核心难点是 OOD action。如果我们拿一个 Q-function 去评价 dataset 从未出现过的动作：

$$
Q(s,a_{\text{OOD}}),
$$

很容易出现 extrapolation error，而且这种误差会被 $\max_a$ 操作系统性地放大——因为被选中的恰恰是被高估得最厉害的那个动作。

IQL 尽量避免显式评价 dataset 之外的动作：它用 expectile regression 学一个偏向高回报的 value，

$$
L_2^\tau(u)=\lvert\tau-\mathbf 1(u<0)\rvert\, u^2,
\qquad \tau\in(0.5,1),
$$

再结合 dataset action 的 $Q$ 来构造 advantage，最后用 advantage-weighted behavior cloning 做 policy extraction。[3]

核心思想就是**不要让 Critic 替 VLA 做 action search；让 Critic 在数据分布内部判断“哪些行为更值得学”。**

这也是 后面RECAP、ROVE、ProgVLA 等方法都越来越偏好的接口。

如果把今天的很多 VLA-RL 工作抽象掉模型名字，会看到一个共同趋势：

$$
\text{RL signal}
\rightarrow
\text{advantage / optimality}
\rightarrow
\text{BC / flow matching / conditioning}
$$

对 foundation policy 来说，**RL 的核心挑战已经从“怎么更新 actor”转向“怎么可靠地产生 credit”。**

---

# Part II：RECAP——Critic 正式进入 Foundation VLA

## 3. $\pi^*_{0.6}$ / RECAP：现代 Critic-based VLA-RL 的转折点

Physical Intelligence 的 $\pi^*_{0.6}$ / RECAP 是这条路线里非常关键的一篇工作。[4] 它做的事情可以用一句话概括：**把 heterogeneous robot experience 先交给 Critic 解释，再把 Critic 产生的 advantage 变成 VLA 的条件变量。**

RECAP 同时利用：1. demonstration；2.autonomous rollout；3.success / failure outcome；4. teleoperation intervention；

### 3.1 架构：Critic 和 VLA 是两个模型

![RECAP architecture](/images/vla-rl-critic/recap_architecture.png)

*图：RECAP 中 value function 与 $\pi^{*}_{0.6}$ 的交互。*

**第一，Value Function 是独立模型。** 它接受视觉 observation、task prompt / metadata，输出一个 value distribution，而不是嵌在 policy 内部的一个 head。

**第二，value 不直接控制 action，而是先用于计算 advantage：**
$$
A(o_t,a_t)
=
r_{t:t+N}
+
V(o_{t+N})
-
V(o_t).
$$

**第三，continuous advantage 被二值化：**

$$
I_t
=
\mathbf 1[A(o_t,a_t)>\epsilon],
$$

然后 $I_t$ 作为 metadata / conditioning 被送回 VLA。

也就是说，最终的 policy 是 

$$
\pi(a\mid o,l,I).
$$

这里有一个值得单独指出的 trade-off：二值化丢掉了 advantage 的**幅度**信息。一个“稍微好一点”的动作和一个“关键性正确”的动作，在 $I_t=1$ 之下被同等对待。换来的是对 critic 数值误差的鲁棒性——只要 critic 的**序**大致正确，条件就正确。这是典型的“用信息量换稳健性”，也是后面 ROVE 改用 percentile 分档、ProgVLA 改用连续 clipped weight 的动机来源。

### 3.2 为什么 RECAP 不直接用 continuous advantage 做 policy gradient？

因为现代 VLA 的 action head 往往是 flow matching / diffusion。这类模型最自然的训练目标不是显式最大化 $\log \pi_\theta(a\mid s)$，而是预测 velocity / denoising direction因为它们根本没有一个便宜可得的 likelihood 可供加权。

RECAP 的 cleverness 在于把 RL 改造成一个条件生成问题。**

如果 $A>0$，就在 prompt / metadata 里标成：

> Advantage: positive

然后训练：

$$
\pi_\theta(a\mid o,l,I=1).
$$

推理时固定给出 positive 条件（或用 classifier-free guidance 放大它），VLA 就可以继续使用原本稳定的 supervised / flow-matching training stack，同时消费 RL 的 value information。

### 3.3 Value Model：distributional + Monte Carlo return

RECAP 训练 distributional value：

$$
p_\phi(V\mid o_t,l).
$$

论文把 return 离散成 201 个 bins，用 cross-entropy 训练，再取期望恢复 continuous value。[4]

这是一个很工程化的选择：

- bounded output 更稳定，不会被少数极端 return 拖走；
- classification 的梯度尺度比裸 regression 更友好；
- value distribution 也更容易表达不确定性和多模态回报。

### 3.4 Value 曲线

![RECAP value curve](/images/vla-rl-critic/recap_value_curve.png)

*图：RECAP 的 value 在成功、犯错、恢复过程中的变化。*

这张图真正说明的是：一个好的 Critic **不应该只是“随时间单调上升”**。

在左侧折衣服的任务里，机器人先取得进展，之后因为动作破坏了已经折好的衣物，value 明显下降；发生 recovery 后，value 又重新升高。

所以好的 execution-time signal 必须能识别 progress、stagnation 和 regression，而不能把“时间变晚”简单当成“进度变高”。[1]

值得注意的是，这类定性曲线目前几乎是整个领域展示 critic 质量的默认方式。它很有说服力，但也很容易挑选

---

# Part III：Critic 最大的问题——它真的理解机器人状态吗？

## 4. Static Value Model 的根本限制：真实机器人是 POMDP

RECAP 解决了“Critic 怎么接 VLA”的问题，但没有解决另一个更基础的问题:**Critic 到底看到了真正的 state 吗？**

考虑两个视觉上非常相似的时刻。

**状态 A：** gripper 刚刚稳定抓起杯子。
**状态 B：** gripper 仍然夹着杯子，但杯子正在滑落。

单张 RGB snapshot 可能极其相似，但：

$$
V_A \gg V_B,
$$

因为 A 的未来大概率继续成功，而 B 的未来可能马上失败。

更一般地说，机器人的真实 state 往往依赖：过去的 motion；接触历史；velocity / acceleration；gripper stability；slip；retry 次数；已完成的 subgoal；当前到底是在 recovery 还是 regression。

于是：

$$
o_t \not\Rightarrow s_t.
$$

这也是 Progress Reward Modeling 综述为什么要把 state interface 分成 single observation、temporal context、relational comparison、privileged state / API 几类——不同接口允许模型访问完全不同程度的 evidence。[1]

![Progress model interface](/images/vla-rl-critic/progress_survey_interface.png)

*图：Progress Reward Modeling 综述对 progress interface 的统一分类。*

如果一个 Critic 只看单帧视觉，它面临的困难和 single-frame progress model 完全一样**视觉语义相似，不代表动力学语义相同。**

从这里开始，Critic 的演化重心就从“RL objective”转向了“state representation”。接下来的三篇工作，可以看成对同一个问题的三种回答：ViVa 说**去预测未来**，ROVE 说**去看更多样的经验**，WCM 说**光有历史输入还不够，得有 world modeling 的 loss**。

---

# Part IV：从 Value Model 到 World Critic

## 5. ViVa：预测未来，才能评价现在

ViVa 的出发点很直接：**Value estimation 本质上是 future anticipation。**

如果 $V(s_t)$ 表示从当前状态出发的未来回报，那么只做静态视觉识别其实是一个很奇怪的建模方式。ViVa 因此没有继续堆 VLM，而是把 pretrained video generator（Wan2.2）改造成 value model。[5]

### 5.1 核心结构

![ViVa architecture](/images/vla-rl-critic/viva_architecture.png)

*图：ViVa architecture。*

ViVa 的输入是：

$$
(\text{multi-view RGB},\ q_t),
$$

其中 $q_t$ 是 proprioception。模型同时预测两个东西：

$$
\begin{cases}
V_t\\
q_{t+K}
\end{cases}
$$

也就是：

$$
\boxed{
\text{当前 Value}
+
\text{未来 proprioception}
}
$$

它借用 pretrained video diffusion Transformer 的时空先验，把当前 multi-view images、当前 proprio、future proprio、scalar value 编码进统一的 latent sequence，让 diffusion model 同时 denoise future embodiment state 和 value target。[5]

### 5.2 为什么预测 future proprio，而不是只做 video prediction？

这是这篇 paper 最值得思考的地方之一。

机器人是否“真的在进步”，很多时候体现在 embodiment dynamics，而不仅仅是图像。例如：arm 是否真的抬起来；grasp 是否保持；joint trajectory 是否进入合理区域；执行动作后 proprio state 是否符合成功过程的分布。

future proprio 提供了一种更直接的 physical grounding，而且它比 pixel-space 预测便宜得多、也干净得多, 它不需要模型去关心背景里飘过的窗帘。

![prop](/images/vla-rl-critic/prop.png)

*图：future proprioception 预测为 value 提供更直接的 physical grounding。*

论文的 ablation 也显示，future-state supervision 的权重过低会削弱对 execution error 的识别；在他们的任务上，适中的 prediction horizon 更有效。[5] 这个结论方向上很好理解：horizon 太短则未来与现在几乎无差别，学不到动力学；太长则预测本身变得高度不确定，反而污染了 value 表示。

一个必须提的代价：把 video diffusion backbone 拿来当 critic，成本远高于一个 MLP value head。它作为**离线的数据打分器 / advantage 生成器**非常合适，但要放进需要高频求值的 on-policy 训练循环里，推理开销就是一个真实的工程约束。

值得一提的是，ViVa 本身并不替换 Part II 的接口, 论文正是把它接进 RECAP 的 pipeline 来验证收益的。[5] 这再次印证了前文的洞察：接口稳定之后，Critic 可以被独立地迭代。如果只给一个 scalar return target，网络很可能学到 shortcut：

$$
\text{某种视觉 pattern}
\rightarrow
\text{高 value}.
$$

但 future prediction 强迫它学习：

$$
\text{当前动态}
\rightarrow
\text{未来状态}.
$$

这已经是 World Model 思想进入 Critic 的第一步。

---

## 6. ROVE：Critic 不只评价机器人，还开始评价“数据值不值得学”

ROVE 把问题换了一个角度。[6] 它研究的是 humanoid manipulation + human intervention，这类数据很特殊,VLA 自主执行；快失败时，人类接管；操作者需要先对齐 humanoid 的身体和手部姿态；再完成 recovery；最终把任务做完。

这意味着 **intervention trajectory 并不天然是 expert trajectory**。

### 6.1 整体 pipeline

![ROVE framework](/images/vla-rl-critic/rove_framework.png)

*图：ROVE framework。*

ROVE 把 intervention episode 显式分成三段：

- Autonomous rollout；
- Intervention adaptation；
- Recovery & completion。

关键观察是：

> adaptation 阶段里，人类常常在“接上机器人当前姿态”，动作可能犹豫、不经济，甚至短暂降低 task value。

如果直接做 HG-DAgger 式的 imitation，把 $a^{\text{human}}$ 一律当成 expert action，就会把这些 suboptimal behavior 一起学进去。而且这类噪声很隐蔽——它出现在最“珍贵”的人类数据里，恰恰是最容易被无条件信任的那一部分。

### 6.2 OVE：不要学“平均未来”，而要学“可恢复的高价值未来”

ROVE 的 Critic 使用 Optimistic Value Estimation（OVE）。先构造 $H$-step TD target：

$$
\hat V_t
=
\sum_{i=t}^{t+H-1}\gamma^{i-t}r_i
+
\gamma^H V_{\bar\phi}(s_{t+H}),
$$

然后使用 expectile regression，使 value 偏向 mixed-quality data 中较高回报的可恢复行为。[6]

直觉上是：

> 在同一个状态附近，如果既有人类犹豫、又有失败 rollout、也有成功 recovery，那么“平均 return”可能过于悲观；我们真正想知道的是“这个状态**有没有**高质量恢复路径”。

这和 IQL 的 expectile 思想同源，但被用在了 human intervention 场景。当然，乐观是有代价的：expectile 越激进，critic 就越容易把噪声中的幸运样本当成可达的高价值路径。这本质上和 offline RL 里 optimism 与 overestimation 的老矛盾是同一个。

### 6.3 Cross-embodiment human video 进入 Critic

ROVE 还有一个非常值得关注的设计：

> **human video 不是拿来直接教 action，而是用来教 Critic 什么叫 progress / failure / recovery。**

这很有潜力。因为 human video 和 robot action space 并不对齐：

$$
a^{\text{human}}\neq a^{\text{robot}},
$$

但 **progress semantics 是对齐的**：白板越擦越干净；面包逐步进入 toaster；错误状态被恢复；任务最终完成。

因此 human video 对 Critic 的价值，可能比对 policy 的直接 imitation 更自然。这也给了“人类视频到底该怎么用”一个新答案: 价值监督。**

### 6.4 Critic 输出 Advantage，再做 advantage conditioning

ROVE 最终仍然回到 RECAP 风格的 policy extraction：

$$
A^\pi(s_t,a_{t:t+H-1})
=
\sum_{i=t}^{t+H-1}\gamma^{i-t}r_i
+
\gamma^H V(s_{t+H})
-
V(s_t),
$$

然后按 advantage distribution 的 percentile 划分 positive / negative condition，再用 CFG-style decoding 引导 actor。[6] 用 percentile 而非绝对阈值，是对 3.1 节那个二值化 trade-off 的一个温和修正：它让条件划分对 critic 的整体尺度漂移不敏感。

### 6.5 Critic 开始承担“数据引擎”的质量控制角色

![ROVE value analysis](/images/vla-rl-critic/rove_value_analysis.png)

*图：加入 human experience 后，ROVE 的 value curve 更能区分 incomplete / recovered states。*

---

## 7. WCM：Critic 最终变成 World Model？

WCM 把这条趋势推得更彻底。[7] 它的核心论点是：

> Critic-based VLA-RL 的根本瓶颈，是 **state approximation problem**。

单帧不够，所以自然会想到把历史堆进去：

$$
o_{t-K+1:t}.
$$

但 WCM 认为：**仅仅把 history stack 喂给 Critic 仍然不够。** 原因是训练监督仍然只有一个 scalar return——它太弱了，弱到不足以逼迫一个大网络去学习跨时间的动力学结构。

### 7.1 整体结构

![WCM architecture](/images/vla-rl-critic/wcm_architecture.png)

*图：WCM architecture，以及 on-policy / off-policy 的接入方式。*

WCM 建立在轻量 LeJEPA 架构之上，首先编码过去 $K$ 帧：

$$
o_{t-K+1:t}
\rightarrow
z_{t-K+1:t},
$$

然后由 World Predictor 同时输出 $\hat z_{t+1}$ 和 $\hat V_t$。训练目标：

$$
\mathcal L
=
\mathcal L_{\text{value}}
+
\lambda\mathcal L_{\text{pred}}
+
\eta\mathcal L_{\text{SIGReg}},
$$

其中：

$$
\mathcal L_{\text{pred}}
=
\|\hat z_{t+1}-z_{t+1}\|_2^2 .
$$

也就是说，Critic 的 latent state 必须同时满足两个条件：

1. 能预测 return；
2. 能预测世界下一时刻的 latent state。

第三项 SIGReg 是 LeJEPA 里的分布正则，作用是防止 latent 空间坍缩——这在任何“预测自己的表示”的自监督目标里都是必需品，否则模型可以让 $z$ 恒等于常数，把预测 loss 降到零而什么都没学到。

### 7.2 为什么 history stacking 本身不够？

WCM 的 ablation 非常有意思。

![WCM history ablation](/images/vla-rl-critic/wcm_history_ablation.png)

*图：MLP / history ViT / WCM 在不同 history length 下的比较。*

论文比较了三种设置：single-frame / multi-frame MLP value head；有 history ViT 但没有 world prediction；以及完整的 WCM。结果说明：**有 temporal encoder 不等于真的学到了 temporal dynamics。**[7]

这是一个非常重要的 distinction：

$$
\text{输入有 history}
\neq
\text{representation 使用了 history}.
$$

如果 loss 只要求预测 return，一个足够大的网络完全可能忽略真正的 dynamics，继续依赖最容易拟合的 visual shortcut。world prediction objective 的作用，就是给 temporal representation 一个强得多的学习约束,它把“必须看历史才能答对”的题目硬塞给了模型。

### 7.3 WCM 不绑定某一种 RL 算法

WCM 另一个值得注意的设计是：它不是“新的 policy optimization algorithm”，而是一个可替换的 value 组件。它可以接：

**On-policy：**

$$
V_{\text{WCM}}
\rightarrow
\text{GAE}
\rightarrow
\text{PPO / Flow-SDE}.
$$

**Off-policy AWR：**

$$
A_t
\approx
G_t-V(o_{t-K+1:t}),
$$

$$
\mathcal L_{\text{actor}}
=
-
\log\pi(a_t\mid o_t)
\exp\left(
\frac{G_t-V(o_{t-K+1:t})}{\beta}
\right).
$$

**Off-policy RECAP：**
$$
V_{\text{WCM}}
\rightarrow
N\text{-step advantage}
\rightarrow
\text{advantage conditioning}.
$$

这让 WCM 的主张变得很清晰：**如果 Critic 的 state representation 足够好，很多现有 RL pipeline 都能一起受益。**

---

# Part V：另一条路线——$Q$ 本身就是 Progress

## 8. GR-RL：sparse reward 学出的 $Q$，为什么可以自然变成 task progress？

GR-RL 是一个很适合连接“Critic”和“Progress Model”的工作。[8] 它的目标是 long-horizon、high-precision、dexterous manipulation，典型任务是穿鞋带。

### 8.1 Policy + Critic

![GR-RL critic architecture](/images/vla-rl-critic/grrl_critic_architecture.png)

*图：GR-RL 的 VLA policy 与 distributional Q critic。*

GR-RL 同时包含 policy 和 critic：

$$
\pi_\theta(o_t,l,s_t),
\qquad
Q_\phi(o_t,l,s_t,a_t).
$$

Critic 不是 scalar regression，而是 distributional $Q$，并且论文把分布的支撑限制在 $[0,1]$。于是从 sparse success reward 学出来的 expected Q-value，天然就落在 0 到 1 之间，表现得很像一个 progress scalar。[8]

这一步值得停下来欣赏一下：**“progress 的取值范围”不是被人为规定的，而是被 reward 的定义域和 Bellman 方程自动继承的。** 没有人去标注“这里是 60%”，但一个有界的、随成功概率单调的量自己出现了。

### 8.2 Q-as-progress：这条思路为什么有吸引力？

GR-RL 用

$$
\rho_t
=
\mathrm{mean}
\left(
Q_\phi(o_t,l,s_t,a_t)
\right)
$$

作为 progress，然后检查 action chunk 内有没有明显的 value drop：

$$
\rho_{t:t+k}.
$$

如果出现超过阈值的下降，就把对应 transition 视为 suboptimal，从 demonstration dataset 中过滤掉。也就是说：

$$
\boxed{
\text{Sparse Success}
\rightarrow
Q
\rightarrow
\text{Progress}
\rightarrow
\text{Data Filtering}
}
$$

注意它和 ROVE 落到了同一个地方：critic 的产物最终被用来**编辑数据集**，而不只是被用来算梯度。这可能是目前 critic 在真实系统里最容易兑现价值的用法——因为数据过滤对 critic 误差的容忍度，远高于把 critic 直接放进 bootstrapping 循环。

### 8.3  value curve 

![GR-RL progress curve](/images/vla-rl-critic/grrl_progress_curve.png)

*图：GR-RL distributional critic 学到的 task-progress 曲线。*

最上面的例子里，鞋带滑出鞋孔后 value 下跌，重新正确穿过之后又上升。第三个例子更有意思：机器人短期重新调整 grasp，看起来像“倒退”，但 Critic 给了这个动作更合理的长期价值。

这正说明了 Value Critic 和简单 temporal progress 的区别：

> **真正的 value 不要求每一步在视觉上都更接近目标，它关心的是 long-term return。**

这是 Critic 相对于 purely monotonic progress estimator 一个很大的理论优势——**允许“以退为进”，是 long-horizon 操作里的刚需。**

VLA 通常输出 $a_{t:t+H}$，所以真正有用的 critic 可能应该是：

$$
Q(
o_{t-K:t},
a_{t:t+H},
g
)
$$

而不是只输出 $V(o_t)$。这会把 temporal history、language goal、action chunk、long-term outcome 全部放进同一个 credit model 里。GR-RL 已经出现了这一方向的雏形 这个是不是和 WCM 的 history encoder 很像，

---

## 9. ProgVLA：把 $Q/V/\text{Success}$ 直接长进 policy 内部

ProgVLA 更进一步把 progress heads 做成 VLA 内部的 auxiliary heads。[9]

![ProgVLA architecture](/images/vla-rl-critic/progvla_architecture.png)

*图：ProgVLA 的 context encoder、progress heads 与 action expert。*

共享 context token $c_t$ 之后，模型同时预测：

$$
\hat V(c_t),
\qquad
\hat Q(c_t,a_t),
\qquad
\hat S(c_t),
$$

然后：

$$
A_t
=
\hat Q(c_t,a_t)
-
\hat V(c_t),
$$

再变成一个被截断的权重：

$$
w_{A,t}
=
\min
\left\{
\exp(A_t/\beta),
C
\right\},
$$

最后用这个 weight 去乘 flow-matching imitation loss。[9] 形式上就是 2.1 节的 AWR，只不过 advantage 来自模型自己的 head，并且用 $C$ 做了截断以防少数样本主导梯度。

这里有一个需要非常谨慎的地方：ProgVLA 自己明确指出，它的数据主要来自 successful demonstrations，$Q$ 和 $V$ 也回归同一个 Monte-Carlo progress target，因此这个 $Q-V$ **不能被过度解释成严格的 offline-RL action advantage**。

---

# Part VI：Progress Model 和 Critic——竞争者，还是上下游？

## 10. $P_t$、$\Delta P_t$、$V_t$、$A_t$ 之间存在很深的结构对应

Progress Reward Modeling 综述把 progress model 的输出大体归纳为四类：state-wise score、progress delta、ranking / preference、executable reward function。[1]

只看最核心的两类：$P_t$ 表示 absolute progress，

$$
\Delta P_t
=
P_{t+1}-P_t
$$

表示 transition progress。而 RL 里，$V_t$ 表示 expected future return，$A_t$ 表示局部 action 相对 baseline 的改进程度。

从数学结构上，两组量高度平行：

$$
P_t
\overset{\Delta}{\longrightarrow}
\Delta P_t
\qquad\text{对应}\qquad
V_t
\overset{TD\ /\ Q-V}{\longrightarrow}
A_t.
$$

于是可以粗略写成：

$$
P_t\leftrightarrow V_t
,
\qquad

\Delta P_t\leftrightarrow A_t
$$

但两者的**语义来源完全不同**，一个来自“看起来像完成了多少”，一个来自“实际上还能拿到多少回报”。

| 维度 | Progress Model | Value Critic |
|---|---|---|
| 核心语义 | 任务是否在向目标推进 | 当前状态 / 动作的 expected return |
| 监督来源 | temporal / VLM / demo / human labels | environment reward / return |
| Dense 程度 | 天然可以 dense | 通过 MC / TD / bootstrapping 传播 |
| Semantic generalization | 通常更强 | 目前往往更 task / data specific |
| Outcome grounding | 可能较弱 | 强 |
| 被 hack 的风险 | reward hacking | critic exploitation（同源问题） |
| Long-horizon reasoning | 取决于 temporal interface | Bellman / return 天然面向未来 |
| Cross-task potential | 高 | 仍在发展 |
| 允许“以退为进” | 单调 progress 假设下困难 | 定义上自然支持 |

？

Progress Model 的最大价值是 **dense semantic supervision**；Critic 的最大价值是 **outcome-grounded credit propagation**。这两件事并不互斥，反而正好互补。

一个很自然的未来系统是把 progress 当成 shaping term 接进 reward：

$$
r_t
=
\alpha\, r_t^{\text{progress}}
+
\beta\, r_t^{\text{success}},
\qquad
r_t
\rightarrow
V/Q
\rightarrow
A.
$$

分工是清楚的：

- **success** 提供不会漂移的 anchor；
- **progress** 提供中间语义；
- **critic** 负责 long-horizon temporal credit。

这里有一个经典理论工具值得提醒：如果 progress 项以 potential-based shaping 的形式给出，即 $r_t^{\text{shape}}=\gamma\Phi(s_{t+1})-\Phi(s_t)$，那么它在理论上**不改变最优策略**。这为“加 dense signal 但不引入新 bias”提供了一个原则性的设计方向，尽管在学习出来的、有噪声的 $\Phi$ 上这个保证会打折扣。

# Part VIII：五条 VLA post-training 路线怎么比较？

## 12. 横向对比

| 路线 | 核心 signal | 代表方法 | 最大优势 | 最大风险 |
|---|---|---|---|---|
| SFT / imitation | expert action | $\pi_0$、OpenVLA 等 | 简单、稳定、可规模化 | 无法有效利用失败和 suboptimal experience |
| Progress / Reward Model | $P,\Delta P,r$ | ARM、VLAC、Robometer | dense、语义强、可跨任务 | calibration / reward hacking |
| Critic-based | $V,Q,A$ | RECAP、ViVa、ROVE、WCM | **temporal credit assignment** | critic error / bootstrap / OOD overfit |
| Q-as-progress | $Q(s,a)$ | GR-RL | action-aware、outcome grounded | offline / bootstrapping bias |
|                         |                  |                        |                                |                                          |

 Critic-based 路线相对其他路线

1.**它把 heterogeneous experience 变成“有方向的数据”**

如果有一条失败 trajectory $\tau_{\text{fail}}$，SFT 往往只能：**不用它。**

但 Critic 可以把这条失败 episode 拆成：

$$
\{A_t>0\}
\cup
\{A_t\approx0\}
\cup
\{A_t<0\}.
$$

即使最终失败，前半段的很多动作仍然可能是好动作。同样，一条成功的 human intervention 也不代表每一步都值得学（ROVE 的核心观察）。

2.它天然适合 long horizon

长任务最困难的是 delayed consequence。某个动作 $a_t$ 可能短期降低 visual progress，却提高最终成功率。

Progress-delta model 容易误判这种动作；而 $V$ / $Q$ 从定义上就是对 long-term return 的估计。这也是 GR-RL 那条 value 曲线最有意义的地方：它能把“短期重新调整 grasp”判断成长期正向动作，而不是机械地要求 progress 单调。

# Part IX：怎样判断一个 Critic 是不是好 Critic？

## 13. 定性曲线之外，我们还应该看什么

整个领域目前展示 critic 质量的默认方式，是画一条 value 曲线，指着上面的下跌说“这里它发现了错误”，指着回升说“这里它识别了 recovery”。RECAP、ROVE、GR-RL 都用了这套叙事，而且确实有说服力。

但只看这个是不够的，原因很简单：**曲线是被挑出来的，而训练用的是全部数据。** 我认为至少还应该关注下面四类证据，它们对应 critic 的四种不同用途。

**（一）rank的正确性（用于 conditioning / reweighting）。** 如果 critic 只被用来划分 positive / negative condition，那真正重要的不是数值准不准，而是**排序准不准**：在 held-out 数据上，$V(o_t)$ 与真实 return $G_t$ 的 Spearman 相关、以及 advantage 符号与“该 transition 是否属于成功轨迹”的一致率，比 value MSE 更有信息量。

**（二）数值校准（用于 bootstrapping）。** 一旦 critic 进入 TD 循环，序正确就不够了。这时应该看 calibration：把预测 value 分桶，检查每桶的实际成功率是否与预测吻合。distributional critic（RECAP、GR-RL）在这里有天然优势，因为它本来就输出一个分布。

**（三）反事实敏感性。** 一个好的 critic 应该对**真正改变未来的事件**敏感，而对无关变化不敏感。可以构造成对样本来测：同一时刻，一个杯子稳、一个杯子在滑；同一时刻，一个背景灯光变了但物理状态不变。前者 value 应该显著分开，后者应该几乎不动。这直接检验 $\pi$-StepNFT 所担心的 nuisance feature 问题。

**（四）下游收益，而且要和 critic-free baseline 比。** 最终判据仍然是策略性能。但这里有一个很容易犯的错误：把“加了 critic 的完整 pipeline”和“什么都不做”对比，然后把全部收益归给 critic。更有信息量的对照组是**同样的数据、同样的 policy extraction、但用随机或常数 advantage**如果性能不掉多少，说明收益其实来自数据筛选或训练配方，而不是 critic 学到的价值。

value 曲线最好在**失败 episode** 上多看几条。成功轨迹上的曲线普遍好看，因为它们本来就单调；critic 的真本事体现在它能不能在事故发生的那一刻及时低头。

---

## 一些需要注意的点

**Critic exploitation：Actor 可能学会优化一个假的世界**

一个错误的 Critic 给出

$$
V(s)=0.9,
$$

并不意味着环境真的会给高回报。

如果 critic 错把“gripper 靠近桌面”当成高 value 的视觉 pattern，actor 就可能通过不断制造这种视觉状态来 exploit critic。这和 reward hacking 本质相同，只是被 hack 的对象从 reward model 换成了 value model。

在这一点上，advantage conditioning 这类接口在这里**比 policy gradient 更安全一些**,  因为 policy 并不是在对 critic 做梯度上升，它只是在模仿被 critic 标为 positive 的数据。critic 的错误会导致学错东西，但不会导致 policy 主动去搜索 critic 的漏洞。这可能是 RECAP 这套接口被广泛沿用的一个隐性原因。

**2. Bootstrap error：Critic 的错误会自我传播**

典型的 TD 更新：

$$
V(s_t)
\leftarrow
r_t
+
\gamma V(s_{t+1}).
$$

如果 $V(s_{t+1})$ 错了，$V(s_t)$ 也会跟着错，进而 $A_t$ 错，actor 学到错误动作，policy 访问新的 OOD state，再让 critic 更错。形成一个闭环：

$$
\boxed{
V\text{ error}
\rightarrow
A\text{ error}
\rightarrow
\pi\text{ shift}
\rightarrow
\text{OOD state}
\rightarrow
V\text{ error}
}
$$

这是 actor-critic 体系里最经典的失败模式，在 multimodal VLA 中被进一步放大，因为视觉输入维度极高，OOD 的定义本身就模糊，你甚至很难察觉自己已经走出了数据分布。

顺带一提，这也是为什么很多 VLA 工作偏好 **Monte-Carlo return 而非深度 bootstrapping**（RECAP 的 value 就是从实际 return 回归的）：MC 方差大但无偏，TD 方差小但会累积偏差。在数据异质、分布不稳的机器人场景里，很多团队宁可要方差。

**3.Partial observability：看不见的物理量不会因为“模型更大”而自动解决** 

很多失败模式根本不是 RGB 语义问题，而是 sensing 问题：contact force；slip；object mass；joint torque；hidden latch state；material deformation；tactile cue。

一个夹持力不足的抓取，和一个夹持力刚好的抓取，在任何分辨率的 RGB 上都可能长得一模一样。这类信息不在观测里，堆参数是堆不出来的。

这也是为什么未来的 VLA Critic 很可能不会只吃 RGB，而会逐渐加入：

$$
\text{vision}
+
\text{proprio}
+
\text{history}
+
\text{tactile}
+
\text{action}.
$$

ViVa（proprio）和 WCM（history）已经明显在朝这个方向走，tactile 大概是下一个。

### 我觉得未来World Model 和 Critic 会进一步融合

ViVa 是 future prediction $+$ $V$；WCM 是 history $+$ latent world prediction $+$ $V$。趋势已经很清晰：未来的 Critic 可能不再是挂在 policy 外面的一个“小 value head”，而是一个真正理解“现在是什么状态、刚才发生了什么、这个动作会导致什么未来”的 dynamics model。

有意思的是，这条路走到尽头会遇到一个身份问题：如果 critic 已经是一个 world model，那它和 policy 内部的表示还应不应该分开？分开意味着两份计算和两套表示，合并意味着 critic 的误差会直接污染 policy 的感知。目前主流选择是分开（RECAP、ViVa、WCM），而 ProgVLA 选择了合并。这个分歧还没有定论。

 Progress Model 很可能成为 Critic 的 auxiliary supervision，而不是竞争者. Part VI 已经论证过融合的形式，这里只补一个我觉得被低估的点：**如果 Progress Model 的输出带不确定性**，
$$
(P_t,\sigma_t),
$$

那么它就不只是一个 reward 项，而可以变成一个**带门控的** reward 项——在 progress model 自己也没把握的区间，降低 pseudo reward 的权重，让 critic 退回去依赖真实 outcome。

纯 environment outcome 的问题是 sparse，纯 semantic progress 的问题是 bias。一个会说“我这里不确定”的 progress model，恰好能让系统在两者之间自动调节，而不是靠一个固定的 $\alpha$。 

## 未来可以scale 的方向: Generalist / Foundation Critic

今天的 VLA policy 已经可以跨很多 task，但 Critic 往往还是 task-specific、dataset-specific、robot-specific、environment-specific 的。这是一个明显的不对称：**我们有了通用的执行者，却还没有通用的裁判。**

真正有价值的下一步可能是：

$$
C(
o_{t-K:t},
a_{t:t+H},
l,
q_t
)
\rightarrow
\{
V,Q,A,P_{\text{succ}},\sigma
\},
$$

并且能跨 task、object、scene、embodiment 工作。

这里有一个未解的难题值得说清楚：value 的定义天然依赖策略（1.1 节），而“通用”意味着要跨越很多不同能力的策略。一个可能的出路是让 critic 显式地条件于“执行者是谁”，或者干脆去估计**近似最优策略下的**价值而非当前策略的价值. 这也正是 IQL / OVE 那类 optimistic 目标在做的事。Foundation Critic 要成立，大概绕不开这个问题。

另一方面, 未来的 Critic 不应该只输出 $V=0.83$，还应该输出,给到一个confidence,来减少幻觉的影响
$$
(V,\sigma_V).
$$

注意这条和前文讨论的三个缺陷是一一对应的：uncertainty 让 exploitation 可被检测（critic 在被 exploit 的区域通常也是它没见过的区域）、让 bootstrap error 可被截断（高不确定时不要 bootstrap）、让 partial observability 可被显式表达（看不见的东西就该体现为高方差）。所以我认为这不是一个“锦上添花”的方向，而是 Critic 真正进入大规模真实机器人 deployment 之前**必须**解决的问题。

参考论文

[1] J. Zhang et al., **Progress Reward Modeling for Robotic Learning: A Comprehensive Survey**, arXiv:2607.21655, 2026.
https://arxiv.org/abs/2607.21655

[2] X. B. Peng et al., **Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning**, arXiv:1910.00177, 2019.
https://arxiv.org/abs/1910.00177

[3] I. Kostrikov, A. Nair, S. Levine, **Offline Reinforcement Learning with Implicit Q-Learning**, arXiv:2110.06169, 2021.
https://arxiv.org/abs/2110.06169

[4] Physical Intelligence et al., **$\pi^{*}_{0.6}$: a VLA That Learns From Experience**, arXiv:2511.14759, 2025.
https://arxiv.org/abs/2511.14759

[5] J. Lv, H. Li et al., **ViVa: A Video-Generative Value Model for Robot Reinforcement Learning**, arXiv:2604.08168, 2026.
https://arxiv.org/abs/2604.08168

[6] W. Xiao et al., **ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning**, arXiv:2606.17011, 2026.
https://arxiv.org/abs/2606.17011

[7] S. Fei et al., **WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning**, arXiv:2607.29613, 2026.
https://arxiv.org/abs/2607.29613

[8] Y. Li et al., **GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation**, arXiv:2512.01801, 2025.
https://arxiv.org/abs/2512.01801

[9] S. Kim et al., **ProgVLA: Progress-Aware Robot Manipulation Skill Learning**, arXiv:2605.28231, 2026.
https://arxiv.org/abs/2605.28231

[10] S. Wang et al., **$\pi$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs**, arXiv:2603.02083, 2026.
https://arxiv.org/abs/2603.02083

