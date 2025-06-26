# 第一章 强化学习中的关键概念

> [!NOTE]
> 本章内容：
> - 强化学习使用的数学语言和数学符号
> - 对强化学习算法的作用进行高层次的解释（先不做代码实现，目标是有一个大概的了解）
> - 算法背后的一些核心数学知识

简而言之，强化学习（RL）是研究代理及其如何通过反复试验进行学习的学科。它形式化地阐述了这样一种观点：对代理的行为进行奖励或惩罚，会增加其在未来重复或放弃该行为的可能性。

**AlphaGo**

下棋的程序是 **代理** 。在训练下棋的程序时，对下的一步好棋进行奖励，对下的一步臭棋进行惩罚，最终训练出一个超越人类的棋手。

**基于人类反馈的强化学习（RLHF）**

大语言模型作为 **代理** 。对大语言模型输出的好的回答进行奖励，对大语言模型输出的不好的回答进行惩罚。进而改变大语言模型的参数（微调），来让大语言模型的输出能够对齐到人类的偏好。

## 关键概念和术语

![](./images/rl_diagram_transparent_bg.png)

![](./images/1.png)

强化学习的主要角色是 **代理** 和 **环境** 。环境是代理生存并与之交互的世界。在交互的每一步，代理都会观察（可能是部分）世界的状态，然后决定采取的行动。环境会随着代理的行动而变化，但也可能自行变化。

代理还会感知来自环境的 **奖励** 信号，这是一个数值，用来告诉代理当前世界状态的好坏。代理的目标是最大化其累积奖励，即 **回报** 。强化学习方法是代理学习行为以实现其目标的方法。

为了更具体地讨论强化学习的作用，我们需要引入一些额外的术语。我们需要讨论

- 状态（State）和观察（Observation）
- 动作空间（Action Space）
- 策略（Policy）
- 轨迹（Trajectory）
- 不同的回报（Return）公式
- 强化学习优化问题
- 价值函数（Value Function）

### 状态和观察

状态 $s$ 是对世界状态的完整描述。状态中不存在任何隐藏于世界之外的信息。 观察 $o$ 是对状态的部分描述，可能会遗漏一些信息。

TODO：添加图片

- 对于棋类，我们可以获取世界状态的完整描述，因为棋盘没有任何隐藏信息。
- 对于超级玛丽游戏，我们只能看到玩家所处的画面，所以只能叫做“观察”。

在深度强化学习中，我们几乎总是用向量、矩阵或张量来表示状态和观察值。例如，视觉观察可以用其像素值的 RGB 矩阵表示；机器人的状态可以用其关节角度和速度表示。

当代理能够观察到环境的完整状态时，我们称该环境是 **完全可观察的** 。当代理只能看到部分观察结果时，我们称该环境是 **部分可观察的** 。

> [!NOTE]
> **必须掌握的概念**
> 强化学习符号有时会将表示状态的符号 $s$ 放在技术上更适合表示观察的符号 $o$ 的位置。具体来说，这种情况发生在讨论代理如何决定某个动作时：我们经常在符号中表示该动作取决于状态，但实际上，由于代理无法访问状态，因此该动作取决于观察。
>
> 具体根据上下文自行判断。

### 动作空间

不同的环境允许不同类型的动作。给定环境中所有有效动作的集合通常称为动作空间 。某些环境，例如围棋，具有 **离散的动作空间** ，其中代理只能进行有限数量的移动。其他环境，例如代理在物理世界中控制机器人的环境，具有 **连续的动作空间** 。在连续空间中，动作是向量、矩阵或者张量。

- 离散动作空间：超级玛丽中玩家只有“起跳”，“蹲下”等有限的几个动作。
- 连续动作空间：自动驾驶，方向盘旋转1度，1.1度，1.2度，......，有无限多种动作。

这种区别对深度强化学习的方法有着相当深远的影响。一些算法只能在一种情况下直接应用，而对于另一种情况则需要进行大量的重新设计。

### 策略

**策略** 是代理用来决定采取哪些行动的规则。它可以是确定性的，在这种情况下通常用 $\mu$ 表示：

$$
a_t=\mu(s_t)
$$

或者策略也可能是随机的，在这种情况下它通常用 $\pi$ 表示：

$$
a_t \sim \pi(\cdot | s_t)
$$

因为策略本质上是代理的大脑，所以用“策略”一词代替“代理”并不罕见，例如说“策略试图最大化奖励”。

在深度强化学习中，我们处理参数化策略：这些策略的输出是可计算函数，取决于一组参数（例如神经网络的权重和偏置），我们可以通过某种优化算法来调整这些参数以改变行为。

我们常常用 $\theta$ 或 $\phi$ 来表示这种策略的参数，然后将其写为策略符号的下标，以突出这种联系：

$$
\begin{split}
a_t &= \mu_{\theta}(s_t) \\
a_t &\sim \pi_{\theta}(\cdot | s_t).
\end{split}
$$

### 确定性策略

```py
pi_net = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, act_dim)
)
```

这将构建一个多层感知机 (MLP) 网络，该网络具有两个大小为 64 的隐藏层和 $\tanh$ 激活函数。如果 `obs` 是一个包含一批观测值的 `Numpy` 数组，则可以使用 `pi_net` 获取一批操作，如下所示：

```py
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)
```

也就是当 `obs_tensor` 张量确定的情况下，MLP 的输出一定是确定的。

### 随机策略

深度强化学习中最常见的两种随机策略是 **分类策略** 和 **对角高斯策略** 。

对于使用和训练随机策略来说，两个关键计算至关重要：

- 从策略中抽样动作，
- 并计算特定动作的对数似然， $\log \pi_{\theta}(a|s)$ 。

接下来，我们将描述如何针对分类策略和对角高斯策略执行这些操作。

> [!NOTE]
> 分类策略
> 分类策略就像一个针对离散动作的分类器。构建分类策略的神经网络的方式与构建分类器的方式相同：输入是观察值，然后是若干层（可能是卷积层或全连接层，具体取决于输入的类型），最后是最后一个线性层，为每个动作提供 `logits` ，最后使用 `softmax` 将 `logits` 转换为概率。
>
> 采样。给定每个动作的概率，PyTorch 等框架内置了采样工具。
>
> 对数似然。将最后一层概率表示为 $P_{\theta}(s)$ 。它是一个向量，其元素数量与动作数量相同，因此我们可以将动作视为该向量的索引。然后，可以通过对向量进行索引来获得动作 $a$ 的对数似然：
>
> $$
> \log \pi_{\theta}(a|s) = \log \left[P_{\theta}(s)\right]_a
> $$
>
> 对角高斯策略
> 多元高斯分布（也可以称之为多元正态分布）由均值向量 $\mu$ 和协方差矩阵 $\Sigma$ 描述。对角高斯分布是一种特殊情况，其协方差矩阵仅在对角线上有元素。因此，我们可以用向量来表示它。
>
> 对角高斯策略始终具有一个从观测值映射到平均动作 $\mu_{\theta}(s)$ 的神经网络。协方差矩阵通常有两种不同的表示方式。
>
> - 第一种方式：只有一个对数标准差向量 $\log \sigma$ ，它不是状态函数： $\log \sigma$ 是独立参数。（PPO 算法的实现就是这样的。）
> - 第二种方式：有一个神经网络，可以将状态映射到对数标准差 $\log \sigma_{\theta}(s)$ 。它可以选择与均值网络共享一些层。
> 请注意，在这两种情况下，我们输出的是对数标准差，而不是直接输出标准差。这是因为对数标准差可以取 $(-\infty, \infty)$ 中的任意值，而标准差必须为非负值。如果不必强制执行这些约束，训练参数会更容易。标准差可以通过对数标准差取幂直接得出，因此用这种方式表示它们不会有任何损失。
> 采样。给定平均动作 $\mu_{\theta}(s)$ 和标准差 $\sigma_{\theta}(s)$ ，以及球面高斯 $( z \sim \mathcal{N}(0, I) )$ 的噪声向量 $z$ ，可以使用以下公式计算动作样本
> $$
> a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z
> $$
> 其中 $\odot$ 表示两个向量的元素乘积。标准框架内置了生成噪声向量的方法，例如 `torch.normal` 。或者，可以构建分布对象，例如通过 `torch.distributions.Normal` ，并使用它们生成样本。（后一种方法的优势在于，这些对象还可以计算对数似然函数。）
> 对数似然。对于均值为 $\mu = \mu_{\theta}(s)$ 、标准差为 $\sigma = \sigma_{\theta}(s)$ 的对角高斯分布， $k$ 维动作 $a$ 的对数似然由下式给出：
> $$
> \log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right)
> $$

### 轨迹

轨迹 $\tau$ 是世界上一系列的状态和动作，

$$
\tau = (s_0, a_0, s_1, a_1, ...)
$$

世界的第一个状态 $s_0$ 是从起始状态分布中随机抽取的，有时表示为 $\rho_0$ ：

$$
s_0 \sim \rho_0(\cdot)
$$

状态转换（即在时间 $t$ , $s_t$ 处的状态与时间 $t+1$ , $s_{t+1}$ 处的状态之间世界所发生的变化）受环境的自然规律支配，并且仅取决于最近的动作 $a_t$ 。它们可以是确定性的，

$$
s_{t+1} = f(s_t, a_t)
$$

或随机的，

$$
s_{t+1} \sim P(\cdot|s_t, a_t)
$$

代理会根据其策略采取相应行动。

### 奖励和回报（Reward and Return）

奖励函数 $R$ 在强化学习中至关重要。它取决于世界的当前状态、刚刚采取的行动以及世界的下一个状态：

$$
r_t = R(s_t, a_t, s_{t+1})
$$

尽管这通常被简化为仅依赖于当前状态 $r_t = R(s_t)$ 或状态-动作对 $r_t = R(s_t,a_t)$ 。

代理的目标是最大化轨迹上的累积奖励，但这实际上可能意味着几件事。我们将用 $R(\tau)$ 来表示所有这些情况，这样一来，上下文就能清楚地表明我们指的是哪种情况，或者这无关紧要（因为相同的方程式适用于所有情况）。

一种回报是 **有限期限的且没有折扣的回报** ，它只是在固定步骤窗口内获得的奖励的总和：

$$
R(\tau) = \sum_{t=0}^T r_t
$$

另一种回报是 **无限期的且有折扣的回报** ，它是代理曾经获得的所有奖励的总和，但会根据未来获得奖励的时间对奖励打折扣。此奖励公式包含一个折扣因子 $\gamma \in (0,1)$ ：

$$
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

但我们为什么要用折扣因子呢？难道我们不是只想获得所有奖励吗？我们当然想，但折扣因子既直观又数学上方便。直观上来说：现在的现金比以后的现金更好。数学上来说：无限期的奖励总和可能不会收敛到一个有限值，而且很难用方程式来处理。但是，有了折扣因子，并且在合理的条件下，无限期的奖励总和就会收敛。

现在就能获得 1 万元，和一百年以后能获得 1 亿元，我们会选择哪一个选项呢？我们会选现在获得 1 万元，因为折扣因子的存在，一百年以后的奖励对现在而言衰减到了几乎为 0 。

虽然在 RL 的数学公式上，这两种回报公式之间的界限非常明显，但深度 RL 实践往往会模糊这条界限——例如​​，我们经常设置算法来优化没有折扣的回报，但在估计价值函数时使用折扣因子。

### 强化学习问题

无论选择何种回报衡量标准（无论是无限期折扣还是有限期不折扣），也无论选择何种策略，RL 中的目标都是选择一种策略，当代理按照该策略行事时，该策略可以最大化 **预期回报** 。

要谈论预期回报，我们首先必须谈论轨迹的概率分布。

假设环境转换和策略都是随机的。在这种情况下， $T$ 步轨迹的概率为：

$$
P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t)
$$

预期回报（无论采用哪种衡量标准）用 $J(\pi)$ 表示，即：

$$
J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = \underset{\tau\sim \pi}E[{R(\tau)}]
$$

强化学习中的核心优化问题可以表示为

$$
\pi^* = \arg \max_{\pi} J(\pi)
$$

其中 $\pi^*$ 为最优策略 。

### 价值函数

了解状态或状态-动作对的 **价值** 通常很有用。我们所说的价值是指从该状态或状态-动作对开始，然后一直按照特定策略行动的预期回报。几乎每种强化学习算法都会以某种方式使用 **价值函数** 。

这里有四个主要值得注意的函数。

1. On-Policy价值函数 $V^{\pi}(s)$ ，如果从状态 $s$ 开始并始终按照策略 $\pi$ 行事，它将给出预期回报：

$$
V^{\pi}(s) = \underset{\tau \sim \pi}E[{R(\tau)\left| s_0 = s\right.}]
$$

2. On-Policy动作-价值函数 $Q^{\pi}(s,a)$ ，如果从状态 $s$ 开始，采取任意动作 $a$ （可能不是来自策略），然后永远按照策略 $\pi$ 采取行动，它将给出预期的回报：

$$
Q^{\pi}(s,a) = \underset{\tau \sim \pi}E[{R(\tau)\left| s_0 = s, a_0 = a\right.}]
$$

3. 最优价值函数 $V^*(s)$ ，如果从状态 $s$ 开始并始终按照环境中的最优策略行事，它将给出预期回报：

$$
V^*(s) = \max_{\pi} \underset{\tau \sim \pi}E[{R(\tau)\left| s_0 = s\right.}]
$$

4. 最优动作价值函数 $Q^*(s,a)$ ，如果从状态 $s$ 开始，采取任意动作 $a$ ，然后永远按照环境中的最优策略采取行动，它将给出预期回报：

$$
Q^*(s,a) = \max_{\pi} \underset{\tau \sim \pi}E[{R(\tau)\left| s_0 = s, a_0 = a\right.}]
$$

当我们讨论价值函数时，如果不考虑时间依赖性，我们仅仅指无限期预期有折扣的收益 。有限期没有折扣的收益的价值函数需要接受时间作为参数。你能想想为什么吗？提示：时间到了会发生什么？

价值函数和动作-价值函数之间有两个经常出现的关键联系：

$$
V^{\pi}(s) = \underset{a\sim \pi}E[{Q^{\pi}(s,a)}]
$$

和

\[
V^*(s) = \max_a Q^* (s,a)
\]

这些关系直接遵循刚刚给出的定义：你能证明它们吗？

### 最优 Q 函数和最优动作

最优动作-价值函数 $Q^*(s,a)$ 与最优策略选择的动作之间存在重要联系。根据定义， $Q^*(s,a)$ 给出了从状态 $s$ 开始，采取（任意）动作 $a$ ，然后始终按照最优策略行动的预期回报。

$s$ 中的最优策略将选择能够最大化从 $s$ 开始的预期回报的行动。因此，如果我们有 $Q^*$ ，我们就可以通过以下方式直接获得最优行动 $a^*(s)$

\[
a^*(s) = \arg \max_a Q^* (s,a)
\]

注意：可能有多个动作可以最大化 $Q^*(s,a)$ ，在这种情况下，所有动作都是最优的，而最优策略可能会随机选择其中任何一个动作。但总有一个最优策略可以确定性地选择一个动作。

### 贝尔曼方程

这四个价值函数都遵循特殊的自洽方程，称为 **贝尔曼方程** 。贝尔曼方程背后的基本思想是：

> 起点的价值是您期望从那里获得的回报，加上下一步的价值。

$$
\begin{align*}
V^{\pi}(s) &= \underset{a \sim \pi \\ {s'\sim P}}E[{r(s,a) + \gamma V^{\pi}(s')}], \\
Q^{\pi}(s,a) &= \underset{s'\sim P}E[{r(s,a) + \gamma \underset{a'\sim \pi}E[{Q^{\pi}(s',a')}}]],
\end{align*}
$$

其中 $s' \sim P$ 是 $s' \sim P(\cdot |s,a)$ 的简写，表示下一个状态 $s'$ 是从环境的转换规则中采样而来的； $a \sim \pi$ 是 $a \sim \pi(\cdot|s)$ 的简写； $a' \sim \pi$ 是 $a' \sim \pi(\cdot|s')$ 的简写。

最优值函数的贝尔曼方程为

$$
\begin{align*}
V^*(s) &= \max_a \underset{s'\sim P}E[{r(s,a) + \gamma V^*(s')}], \\
Q^*(s,a) &= \underset{s'\sim P}E[{r(s,a) + \gamma \max_{a'} Q^*(s',a')}].
\end{align*}
$$

On-Policy价值函数的贝尔曼方程与最优价值函数之间的关键区别在于， $\max$ 在动作上的存在与否。它的出现反映了这样一个事实：每当代理选择其动作时，为了采取最优行动，它必须选择能够带来最高价值的动作。

> “贝尔曼备份”这个术语在强化学习文献中经常出现。一个状态（或状态-动作对）的贝尔曼备份是贝尔曼方程的右边：奖励加下一个值。

### 优势函数

在强化学习中，有时我们不需要描述某个动作的绝对优劣，而只需要描述它平均比其他动作好多少。也就是说，我们想知道该动作的相对优势 。我们用优势函数来精确地表述这个概念。

与策略 $\pi$ 对应的优势函数 $A^{\pi}(s,a)$ 描述了在状态 $s$ 下采取特定行动 $a$ 比根据 $\pi(\cdot|s)$ 随机选择行动（假设你此后一直按照 $\pi$ 行动）要好多少。从数学上讲，优势函数定义为

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
$$

优势函数对于策略梯度方法至关重要。

### 形式化

到目前为止，我们已经非正式地讨论了代理的环境，但如果你尝试深入研究相关文献，你很可能会遇到这种设置的标准数学形式： 马尔可夫决策过程 (MDP)。MDP 是一个 5 元组， $\langle S, A, R, P, \rho_0 \rangle$ ，其中

- $S$ 是所有有效状态的集合，
- $A$ 是所有有效动作的集合，
- $R : S \times A \times S \to \mathbb{R}$ 是奖励函数，其中 $r_t = R(s_t, a_t, s_{t+1})$ ，
- $P : S \times A \to \mathcal{P}(S)$ 是转移概率函数，其中 $P(s'|s,a)$ 表示从状态 $s$ 开始并采取行动 $a$ 时转移到状态 $s'$ 的概率，
- 并且 $\rho_0$ 是起始状态分布。

马尔可夫决策过程这个名称指的是系统遵循马尔可夫特性 ：转换仅取决于最近的状态和动作，而不取决于先前的历史。

# 第二章 强化学习算法的种类

现在我们已经了解了 RL 术语和符号的基础知识，我们可以介绍一些更丰富的内容：现代 RL 中的算法概况，以及算法设计中涉及的各种权衡的描述。

## 强化学习算法的分类

![](./images/rl_algorithms_9_15.svg)

现代强化学习中算法的分类法并非详尽无遗，但非常实用。

本节首先声明：在现代强化学习领域，绘制一个准确、包罗万象的算法分类法非常困难，因为树形结构无法很好地体现算法的模块化。此外，为了使内容能够放在一页纸上，并在导论中易于理解，我们必须省略相当多的高级内容（探索、迁移学习、元学习等）。话虽如此，我们的目标是：

- 强调深度强化学习算法中最基本的设计选择，即学习什么以及如何学习，
- 揭示这些选择中的权衡，
- 并根据这些选择将一些突出的现代算法置于背景中。

## 无模型的 RL VS 基于模型的 RL

强化学习算法中最重要的分支点之一是 **代理是否可以访问（或学习）环境模型** 。我们所说的环境模型是指能够预测状态转换和奖励的函数。

拥有模型的主要优势在于， 它允许代理通过提前思考进行规划 ，预测一系列可能的选择结果，并在各个选项之间做出明确的决定。然后，代理可以将提前规划的结果提炼成学习策略。这种方法的一个著名例子是 AlphaZero 。当这种方法奏效时，与没有模型的方法相比，它可以显著提高采样效率。

主要缺点是， 代理通常无法获得环境的真实模型。 在这种情况下，如果代理想要使用模型，它必须完全依靠经验来学习，这会带来诸多挑战。最大的挑战在于，模型中的偏差可能会被代理利用，导致代理在学习到的模型上表现良好，但在实际环境中却表现不佳（甚至极其糟糕）。模型学习从根本上来说很难，因此即使付出巨大的努力——愿意投入大量时间和计算——也可能无法获得回报。

使用模型的算法称为基于模型的方法，不使用模型的算法称为非模型方法。虽然非模型方法放弃了使用模型可能带来的样本效率提升，但它们往往更易于实现和调整。一般来说，非模型方法比基于模型的方法更受欢迎，并且得到了更广泛的开发和测试。

## 学习什么

强化学习算法中另一个关键的分支点是学习什么。 常见的问题包括：

- 策略，无论是随机的还是确定性的，
- 动作-价值函数（Q 函数），
- 价值函数
- 环境模型

## 无模型强化学习中要学什么

使用无模型 RL 来表示和训练代理主要有两种方法：

**策略优化** 。 此类方法将策略明确表示为 $\pi_{\theta}(a|s)$ 。它们直接通过梯度上升来优化性能目标 $J(\pi_{\theta})$ ，或间接地通过最大化 $J(\pi_{\theta})$ 的局部近似值来优化参数 $\theta$ 。这种优化几乎始终基于策略执行，这意味着每次更新仅使用根据最新版本策略执行时收集的数据。策略优化通常还涉及学习基于策略的价值函数 $V^{\pi}(s)$ 的近似器 $V_{\phi}(s)$ ，该近似器用于确定如何更新策略。

以下是一些策略优化方法的示例：

- A2C / A3C ，通过梯度上升来直接最大化性能，
- 和 PPO ，其更新通过最大化替代目标函数间接地最大化性能，该替代目标函数对更新导致的 $J(\pi_{\theta})$ 变化量给出保守估计。

**Q-Learning**。 这类方法会学习一个近似器 $Q_{\theta}(s,a)$ ，用于最优动作值函数 $Q^*(s,a)$ 。通常，它们使用基于贝尔曼方程的目标函数。这种优化几乎总是以离策略的方式执行，这意味着每次更新都可以使用在训练期间任何时间点收集的数据，而无论代理在获取数据时选择如何探索环境。相应的策略是通过 $Q^*$ 和 $\pi^*$ 之间的联系获得的：Q-Learning 代理采取的动作由下式给出：

$$
a(s) = \arg \max_a Q_{\theta}(s,a).
$$

Q 学习方法的例子包括

- DQN ，一个经典之作，极大地推动了深度强化学习领域的发展，
- 以及 C51 ，它是学习期望为 $Q^*$ 的回报分布的变体。

策略优化与 Q 学习之间的权衡。 策略优化方法的主要优势在于其原则性，即直接针对目标进行优化。 这往往使其稳定可靠。相比之下，Q 学习方法仅通过训练 $Q_{\theta}$ 来满足自洽方程，从而间接优化代理性能。这种学习方法存在多种故障模式，因此稳定性较差。但是，Q 学习方法在实际应用中具有显著更高的采样效率，因为它们可以比策略优化技术更有效地重用数据。

**介于策略优化和Q学习之间的算法** 。策略优化和 Q 学习并非互不相容（在某些情况下，甚至等价 ），并且存在一系列介于两者之间的算法。这些算法能够巧妙地平衡两者的优势和劣势。例如：

- DDPG 是一种同时学习确定性策略和 Q 函数的算法，通过相互改进来提高学习效果。
- 以及 SAC ，它是一种使用随机策略、熵正则化和其他一些技巧来稳定学习并在标准基准上得分高于 DDPG 的变体。

## 基于模型的强化学习需要学习什么

与无模型强化学习不同，基于模型的强化学习并没有少数易于定义的方法集群：存在许多使用模型的正交方法。我们将给出一些示例，但列表远非详尽无遗。在每种情况下，模型可以是给定的，也可以是学习得到的。

背景：纯规划。 最基本的方法从不明确地表示策略，而是使用纯规划技术（例如模型预测控制 (MPC)）来选择动作。在 MPC 中，每次智能体观察环境时，它都会计算一个相对于模型最优的计划，该计划描述了在当前时间窗口之后某个固定时间窗口内执行的所有动作。（规划算法可以通过使用学习到的价值函数来考虑超出时间范围的未来奖励。）然后，智能体会执行该计划的第一个动作，并立即丢弃其余动作。每次准备与环境交互时，它都会计算一个新计划，以避免使用规划时间范围短于预期的计划中的动作。

- MBMF 工作利用学习环境模型在深度 RL 的一些标准基准任务上探索 MPC。

专家迭代。 纯规划的直接后续步骤涉及使用和学习策略 $\pi_{\theta}(a|s)$ 的显式表示。代理在模型中使用规划算法（例如蒙特卡洛树搜索），通过从当前策略中采样来生成规划的候选动作。规划算法生成的动作比单独使用策略生成的动作更好，因此相对于策略而言，它可以说是“专家”。之后，策略会进行更新，以生成更接近规划算法输出的动作。

- ExIt 算法使用这种方法训练深度神经网络来玩 Hex。
- AlphaZero 是这种方法的另一个例子。

无模型方法的数据增强。 使用无模型强化学习算法来训练策略或 Q 函数，但可以 1）在更新代理时用虚构经验来增强真实经验，或 2） 仅使用虚构经验来更新代理。

- 请参阅 MBVE ，了解使用虚构体验增强真实体验的示例。
- 请参阅世界模型 ，了解使用纯粹虚构经验来训练代理的示例，他们称之为“梦中训练”。

将规划循环嵌入策略。 另一种方法是将规划过程作为子程序直接嵌入到策略中——这样完整的规划就成为策略的辅助信息——同时使用任何标准的无模型算法训练策略的输出。关键概念在于，在这个框架下，策略可以学习选择如何以及何时使用这些规划。这使得模型偏差不再是问题，因为如果模型在某些状态下不利于规划，策略可以简单地学习忽略它。

- 请参阅 I2A ，了解代理被赋予这种想象力的示例。

# 第三章 策略优化简介

在本章中，我们将讨论策略优化算法的数学基础，并将相关内容与示例代码联系起来。我们将介绍策略梯度理论中的三个关键结果：

- 描述策略性能相对于策略参数的梯度的最简单方程 ，
- 允许我们从表达式中删除无用的术语的规则，
- 以及允许我们向该表达式添加有用术语的规则。

最后，我们将这些结果结合在一起，并描述基于优势的策略梯度表达式——我们在 Vanilla Policy Gradient 实现中使用的版本。

## 推导最简单的策略梯度

这里，我们考虑随机参数化策略 $\pi_{\theta}$ 的情况。我们的目标是最大化预期收益 $J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{R(\tau)}$ 。为了便于推导，我们取 $R(\tau)$ 来表示有限期限下的未折现收益 ，但无限期限下的折现收益的推导过程几乎相同。

我们希望通过梯度上升来优化策略，例如

$$
\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.
$$

策略性能的梯度 $\nabla_{\theta} J(\pi_{\theta})$ 称为策略梯度 ，以这种方式优化策略的算法称为策略梯度算法。 （例如 Vanilla 策略梯度和 TRPO。PPO 通常被称为策略梯度算法，尽管这种说法略有不准确。）

要实际使用该算法，我们需要一个可以数值计算的策略梯度表达式。这涉及两个步骤：1）推导策略性能的解析梯度，该解析梯度最终呈现为期望值的形式；2）形成该期望值的样本估计值，该样本估计值可以通过有限数量的代理-环境交互步骤的数据计算得出。

在本小节中，我们将找到该表达式的最简形式。在后面的小节中，我们将展示如何改进该最简形式，以获得我们在标准策略梯度实现中实际使用的版本。

我们首先列出一些有助于推导分析梯度的事实。

1. 轨迹的概率。 假设动作来自 $\pi_{\theta}$ ，则轨迹 $\tau = (s_0, a_0, ..., s_{T+1})$ 的概率为

$$
P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).
$$

2. 对数导数技巧。 对数导数技巧基于微积分中的一个简单规则： $\log x$ 对 $x$ 的导数为 $1/x$ 。重新排列并结合链式法则，我们得到：

$$
\nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).
$$

3. 轨迹的对数概率。 轨迹的对数概率就是

$$
\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).
$$

4. 环境函数的梯度。 环境不依赖于 $\theta$ ，因此 $\rho_0(s_0)$ 、 $P(s_{t+1}|s_t, a_t)$ 和 $R(\tau)$ 的梯度为零。

5. 轨迹的梯度对数概率。 轨迹的对数概率的梯度如下

$$
\nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
&= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).
$$

综合起来，我们得出以下结论：

> [!NOTE]
> 基于策略梯度的推导
> ![](./images/2.svg)

这是一个期望，这意味着我们可以用样本均值来估计它。如果我们收集一组轨迹 $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$ ，其中每条轨迹都是通过让代理使用策略 $\pi_{\theta}$ 在环境中行动而获得的，那么策略梯度可以用以下公式来估计：

$$
\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

其中 $|\mathcal{D}|$ 是 $\mathcal{D}$ （此处为 $N$ ）中的轨迹数。

最后一个表达式是我们想要的可计算表达式的最简单版本。假设我们已经以一种允许计算 $\nabla_{\theta} \log \pi_{\theta}(a|s)$ 的方式表示了我们的策略，并且如果我们能够在环境中运行该策略来收集轨迹数据集，那么我们就可以计算策略梯度并采取更新步骤。

## 实现最简单的策略梯度

1. 制定策略网络

```py
# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()
```

此模块构建了使用前馈神经网络分类策略的模块和函数。（请参阅第一部分中的 “随机策略” 部分进行复习。） `logits_net` 模块的输出可用于构建对数概率和动作概率，而 `get_action` 函数则根据从对数计算出的概率对动作进行采样。（注意：此 `get_action` 函数假设只提供一个 `obs` ，因此只有一个整数动作输出。因此它使用了 `.item()` ，用于获取只有一个元素的张量的内容 。）

本例中的很多工作是由 L36 上的 `Categorical` 对象完成的。这是一个 PyTorch Distribution 对象，它封装了一些与概率分布相关的数学函数。具体来说，它包含一个从分布中采样的方法（我们在 L40 上用到）和一个计算给定样本对数概率的方法（我们稍后会用到）。由于 PyTorch 分布对强化学习非常有用，请查看它们的文档来了解它们的工作原理。

> 友情提醒！当我们谈论具有“logits”的分类分布时，我们的意思是，每个结果的概率由 `logits` 的 `Softmax` 函数给出。也就是说，在 `logits` 为 $x_j$ 的分类分布下，动作 $j$ 的概率为：
>
> $$
> p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}
> $$

2. 制定损失函数

```py
# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

在这一块中，我们为策略梯度算法构建了一个“损失”函数。当插入正确的数据时，该损失的梯度等于策略梯度。正确的数据是指根据当前策略执行时收集的一组（状态、动作、权重）三元组，其中状态-动作对的权重是其所属情节的回报。（不过，正如我们将在后面的小节中展示的那样，你也可以为权重插入其他值，这些值同样可以正常工作。）

> [!NOTE]
> 关键点
>
> 尽管我们将其描述为损失函数，但它并非监督学习中典型意义上的损失函数。它与标准损失函数主要有两点区别。
>
> 1. **数据分布取决于参数**。 损失函数通常定义在固定的数据分布上，该分布与我们要优化的参数无关。但这里并非如此，数据必须基于最新的策略进行采样。
>
> 2. **它不衡量性能**。 损失函数通常评估我们关心的性能指标。在这里，我们关心的是预期收益 $J(\pi_{\theta})$ ，但我们的“损失”函数根本无法接近这个值，甚至在期望值上也是如此。这个“损失”函数对我们有用的唯一原因是，当使用当前参数生成的数据对当前参数进行评估时，它的性能梯度为负。
>
> 但在梯度下降的第一步之后，它就与性能不再有任何联系了。这意味着，对于给定的一批数据，最小化这个“损失”函数并不能保证预期收益的提升。你可以将这个损失设为 $-\infty$ ，策略性能可能会大幅下降；事实上，通常情况下确实如此。有时，深度强化学习研究人员可能会将这种结果描述为策略对一批数据的“过拟合”。这是描述性的，但不应从字面上理解，因为它并非泛化误差。
>
> 我们提出这一点是因为机器学习从业者常常将损失函数解读为训练过程中的一个有用信号——“如果损失下降，就万事大吉”。但在策略梯度下降中，这种直觉是错误的，你应该只关心平均收益。损失函数毫无意义。

> [!NOTE]
> 关键点
>
> 这里用来制作 `logp` 张量的方法——调用 PyTorch Categorical 对象的 `log_prob` 方法——可能需要进行一些修改才能与其他类型的分布对象一起使用。
>
> 例如，如果使用正态分布 （对于对角高斯策略），调用 `policy.log_prob(act)` 的输出将提供一个张量，其中包含每个向量值动作的每个分量的单独对数概率。也就是说，输入一个形状为 `(batch, act_dim)` 的张量，并得到一个形状为 `(batch, act_dim)` 的张量，而进行 RL 损失所需的是形状为 `(batch,)` 的张量。在这种情况下，将对动作分量的对数概率求和以获得动作的对数概率。也就是说，将计算：
> ```py
> logp = get_policy(obs).log_prob(act).sum(axis=-1)
> ```

3. 运行一个训练周期

```py
# for training policy
def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens
```

`train_one_epoch()` 函数运行一个“epoch”的策略梯度，我们将其定义为

1. 经验收集步骤（L67-102），其中代理使用最新的策略在环境中执行一定数量的情节，然后
2. 单个策略梯度更新步骤（L104-111）。

算法的主循环只是重复调用 `train_one_epoch()` 。

> 如果你还不熟悉 PyTorch 中的优化，请观察第 104-111 行所示的梯度下降步骤模式。首先，清除梯度缓冲区。然后，计算损失函数。接着，计算损失函数的反向传播；这会将新的梯度累积到梯度缓冲区中。最后，使用优化器进行下一步。



原始策略梯度算法的实现代码。

```py
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    # 此模块构建了使用前馈神经网络分类策略的模块和函数。
    # （请参阅第一章中的 “随机策略” 部分进行复习。）
    # `logits_net` 模块的输出可用于构建对数概率和动作概率，
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        # 本例中的很多工作是由 L36 上的 Categorical 对象完成的。
        # 这是一个 PyTorch Distribution 对象，它封装了一些与概率分布相关的数学函数。
        # 具体来说，它包含一个从分布中采样的方法（我们在 L40 上用到）
        # 和一个计算给定样本对数概率的方法（我们稍后会用到）。
        # 由于 PyTorch 分布对强化学习非常有用，请查看它们的文档来了解它们的工作原理。
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    # 而 `get_action` 函数则根据从对数计算出的概率对动作进行采样。
    # 注意：此 get_action 函数假设只提供一个 obs ，因此只有一个整数动作输出。
    # 因此它使用了 .item() ，用于获取只有一个元素的张量的内容 。
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    # 在这一块中，我们为策略梯度算法构建了一个“损失”函数。
    # 当插入正确的数据时，该损失的梯度等于策略梯度。
    # 正确的数据是指根据当前策略执行时收集的一组（状态、动作、权重）三元组，
    # 其中状态-动作对的权重是其所属情节的回报。
    # （不过，正如我们将在后面的小节中展示的那样，你也可以为权重插入其他值，这些值同样可以正常工作。）
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    # 运行一个训练周期。
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
```

## 预期梯度对数概率（EGLP）引理

Expected Grad-Log-Prob Lemma

在本小节中，我们将推导出一个在策略梯度理论中被广泛使用的中间结果。我们将其称为预期梯度对数概率（EGLP）引理。

EGLP 引理。 假设 $P_{\theta}$ 是随机变量 $x$ 的参数化概率分布。则：

$$
\underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)} = 0.
$$

> 证明：
> 
> 回想一下，所有概率分布都是归一化的 ：
> 
> $$
> \int_x P_{\theta}(x) = 1.
> $$
> 
> 对归一化条件两边取梯度：
> 
> $$
> \nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.
> $$
> 
> 使用对数导数技巧可得到：
> 
> $$
> 0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
> &= \int_x \nabla_{\theta} P_{\theta}(x) \\
> &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
> \therefore 0 &= \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)}.
> $$
> 

## 别让过去分散你的注意力

检查我们最新的策略梯度表达式：

$$
\nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}.
$$

按照这个梯度迈出一步，每个动作的对数概率都会与 $R(\tau)$ （所有获得奖励的总和）成比例地上升。但这没有多大意义。

代理实际上应该只根据行动的后果来强化行动。采取行动之前获得的奖励与该行动的效果无关：只有行动之后获得的奖励才有效。

事实证明，这种直觉在数学上是成立的，我们可以证明策略梯度也可以表示为

$$
\nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}.
$$

在这种形式下，行动只会根据采取行动后获得的奖励而得到强化。

我们将这种形式称为“奖励前进策略梯度”，因为在轨迹中某一点之后的奖励总和，

$$
\hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),
$$

被称为从该点开始的奖励 ，并且该策略梯度表达式取决于状态-动作对的奖励。

> 但如何才能更好呢？ 策略梯度的一个关键问题是，需要多少样本轨迹才能获得低方差的样本估计。我们最初的公式包含了与过去奖励成比例的强化动作项，这些项的均值为零，但方差不为零：因此，它们只会给策略梯度的样本估计增加噪声。通过移除这些项，我们可以减少所需的样本轨迹数量。

此声明的（可选）证明可在 “此处” 找到，它最终取决于 EGLP 引理。

## 实现 Reward-to-Go 策略梯度

与 `1_simple_pg.py` 相比，唯一的变化是我们现在在损失函数中使用了不同的权重。代码修改非常小：我们添加了一个新函数，并更改了另外两行。新函数如下：

```py
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs
```

然后我们对旧的 L91-92 进行调整：

```py
                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len
```

到：

```py
                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))
```

## 策略梯度中的基线

EGLP 引理的一个直接推论是，对于任何仅依赖于状态的函数 $b$ ，

$$
\underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0.
$$

这使我们能够在策略梯度表达式中添加或减去任意数量的类似项，而无需改变其期望值：

$$
\nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}.
$$

任何以此方式使用的函数 $b$ 都称为基线 。

最常见的基线选择是策略值函数 $V^{\pi}(s_t)$ 。回想一下，如果智能体从状态 $s_t$ 开始，然后在其余生中按照策略 $\pi$ 行动，则该函数表示智能体获得的平均回报。

从经验上看，选择 $b(s_t) = V^{\pi}(s_t)$ 具有降低策略梯度样本估计方差的理想效果。这可以实现更快、更稳定的策略学习。从概念角度来看，它也极具吸引力：它编码了这样一种直觉：如果代理获得了预期的结果，它应该对此“感觉”中立。

实际上， $V^{\pi}(s_t)$ 无法精确计算，因此必须进行近似计算。这通常使用神经网络 $V_{\phi}(s_t)$ 来实现，该神经网络与策略同时更新（因此价值网络始终近似于最新策略的价值函数）。

在大多数策略优化算法（包括 VPG、TRPO、PPO 和 A2C）的实现中，学习 $V_{\phi}$ 的最简单方法是最小化均方误差目标：

$$
\phi_k = \arg \min_{\phi} \underE{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2},
$$

其中 $\pi_k$ 是第 $k$ 个周期的策略。这是通过一步或多步梯度下降完成的，从先前的值参数 $\phi_{k-1}$ 开始。

