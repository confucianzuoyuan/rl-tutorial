# 第一章 强化学习中的关键概念

> [!NOTE]
> 本章内容：
> - 强化学习使用的数学语言和数学符号
> - 对强化学习算法的作用进行高层次的解释（先不做代码实现，目标是有一个大概的了解）
> - 算法背后的一些核心数学知识

> [!NOTE]
> 术语表
> - Reinforcement Learning: 强化学习
> - Agent: 智能体
> - Environment: 环境
> - State: 状态
> - Action: 动作
> - Policy: 策略
> - Advantage: 优势
> - Reward: 奖励
> - Return: 回报
> - Observation: 观察

**强化学习** 是研究智能体及其如何通过反复试验进行学习的学科。它形式化地阐述了这样一种观点：对智能体的行为进行奖励或惩罚，会增加智能体在未来重复或放弃该行为的可能性。

> “形式化”的意思是从数学的角度证明了这一点。

## 强化学习的应用

**AlphaGo**

下棋的程序是 **智能体** 。在训练下棋的程序时，对下的一步好棋进行奖励，对下的一步臭棋进行惩罚，最终训练出一个超越人类的AI棋手。

**基于人类反馈的强化学习（RLHF）**

大语言模型作为 **智能体** 。对大语言模型输出的好的回答进行奖励，对大语言模型输出的不好的回答进行惩罚。进而改变大语言模型的参数（也就是“微调”），来让大语言模型的输出能够对齐到人类的偏好。

所谓对齐到人类的偏好，就是按照人类喜欢的方式输出回答。

## 关键概念和术语

![](./images/rl_diagram_transparent_bg.png)

![](./images/1.png)

强化学习的主要角色是 **智能体** 和 **环境** 。环境是智能体生存并与之交互的世界。在交互的每一步，智能体都会观察环境的状态，然后决定采取的行动。环境会随着智能体的行动而变化，但环境也可能自行变化。

智能体可能能够观察到当前环境的完整状态，例如如果下棋的AI程序是智能体的话，那么这个智能体可以获取到当前环境（也就是棋盘）的所有信息。

智能体也可能无法观察到当前环境的完整状态，人作为生活在自然环境中的智能体，是无法获取整个自然环境的完整信息的，只能部分的获取环境的信息。

智能体还会感知来自环境的 **奖励** 信号，这是一个数值，用来告诉智能体当前环境的状态的好坏。智能体的目标是最大化智能体获得的累积奖励，也就是 **回报** 。强化学习方法是智能体学习行为以实现其目标的方法。

为了更具体地讨论强化学习的作用，我们需要引入一些额外的术语。我们需要讨论

- 状态（State）和观察（Observation）
- 动作空间（Action Space）
- 策略（Policy）
- 轨迹（Trajectory）
- 不同的回报（Return）公式
- 强化学习优化问题
- 价值函数（Value Function）

## 状态和观察

状态 $s$ 是对环境的状态的完整描述。状态中不存在任何隐藏于环境之外的信息。观察 $o$ 是对状态的部分描述，可能会遗漏一些信息。

- 对于棋类，下棋的AI程序或者人类作为智能体，可以获取环境状态的完整描述，因为棋盘没有任何隐藏信息。
- 对于超级玛丽游戏，玩家作为智能体只能看到当前超级玛丽所处的画面，无法获取其它的游戏画面（例如无法看到其它关卡的游戏画面）。由于无法获取游戏的所有信息，所以只能叫做“观察”。

如果将深度学习技术和强化学习技术结合使用，则称为 **深度强化学习** 。号称人工智能中，最具有吸引力的话题。

> 具体如何结合起来，后面我们会看到。

在深度强化学习中，我们几乎总是用向量、矩阵或张量来表示状态和观察值。例如，视觉观察可以用其像素值的 RGB 矩阵表示；机器人的状态可以用其关节角度和速度表示。

当智能体能够观察到环境的完整状态时，我们称该环境是 **完全可观察的** 。当智能体只能看到部分观察结果时，我们称该环境是 **部分可观察的** 。

> [!NOTE]
> **笔记**
>
> 强化学习符号有时会将表示状态的符号 $s$ 放在技术上更适合表示观察的符号 $o$ 的位置。具体来说，这种情况发生在讨论智能体如何决定某个动作时：我们经常在符号中表示该动作取决于状态，但实际上，由于智能体无法访问状态，因此智能体的动作取决于观察。

## 动作空间

不同的环境允许不同类型的动作。给定环境中所有有效动作的集合通常称为动作空间 。某些环境，例如围棋，具有 **离散的动作空间** ，其中智能体只能进行有限数量的移动。其他环境，例如智能体在物理世界中控制机器人的环境，具有 **连续的动作空间** 。在连续空间中，动作是向量、矩阵或者张量。

- 离散动作空间：超级玛丽中玩家只有“起跳”，“蹲下”等有限的几个动作。
- 连续动作空间：自动驾驶，方向盘旋转1度，1.1度，1.2度，......，有无限多种动作。

这种区别对深度强化学习的方法有着相当深远的影响。一些算法只能在一种情况下直接应用，而对于另一种情况则需要进行大量的重新设计。

## 策略

**策略** 会告诉智能体在每一个状态应该采取什么样的动作。

在数学上，策略可以通过条件概率来描述。我们常用 $\pi(a|s)$ 来表示在状态 $s$ 采取动作 $a$ 的概率（注意这里的 $\pi$ 不是圆周率）。这个概率对每一个状态和每一个动作都有定义。

也就是在环境处于某个状态时，智能体采取某个动作的概率是多少。

![](./images/3.png)

因为策略本质上是智能体的大脑，所以用“策略”一词代替“智能体”并不罕见，例如说“策略试图最大化奖励”。

策略本质上是一个函数，接收状态作为输入，输出各种动作的概率。所以策略函数可以是一个深度学习模型。

如果我们将 DeepSeek 大模型看作一个策略的话，那么状态是：用户输入的提示词。由于大模型的算法是接收用户的输入，并预测下一个词（predict next token, pnt）。所以动作其实就是大模型根据提示词预测出来的下一个词的概率分布。然后根据温度值的不同，可能选择不同的词作为下一个词。

由于深度学习模型有很多很多的参数，所以我们需要将深度学习的参数也放到公式里面。

所以公式就变成了：

$$
\pi_\theta(a|s)
$$

这里的 $\theta$ 表示深度学习模型中的那些权重和偏置。

> 强化学习中的数学符号很多，不要被迷惑，要理解本质！

下面举一个完整的例子，把之前的概念穿起来，理解一下。

下图展示了一个网格世界的例子。其中有一个智能体在网格中移动。在每个时刻，智能体只能占据一个单元格。白色单元格代表可以进入的区域，红色单元格代表禁止进入的区域，绿色单元格代表目标区域。智能体的任务是从一个初始区域出发，最终到达目标区域。

![](./images/4.png)

如果智能体知道网格世界的地图，那么规划一条能到达目标单元格的路径其实并不难。然而，如果智能体事先不知道有关环境的任何信息，这个任务就变得有挑战性了。此时，智能体需要和环境交互，通过获取经验来找到一个好的策略。

**状态和动作**

在网格世界中，状态对应了智能体所在单元格的位置。网格世界有 9 个单元格，所以也对应了 9 个状态，表示为 $s_1, s_2, ..., s_9$ 。所有状态的集合被称为状态空间，表示为 $\mathcal{S} = \left\{s_1, s_2, ..., s_9\right\}$ 。

在网格世界中，智能体在每一个状态有 5 个可选的动作：向上移动、向右移动、向下移动、向左移动、保持不动。这 5 个动作分别表示为： $a_1, a_2, ..., a_5$ 。所有动作的集合被称为动作空间，表示为 $\mathcal{A} = \left\{ a_1, a_2, ..., a_5 \right\}$ 。

![](./images/5.png)

**状态转移**

当执行一个动作时，智能体可能从一个状态转移到了另一个状态，这样的过程被称为状态转移。例如，如果智能体当前时刻处在状态 $s_1$ 并且执行动作 $a_2$ （即向右移动），那么智能体会在下一个时刻移动到状态 $s_2$ ，这个过程可以表示为：

$$
s_1 \xrightarrow{a_2} s_2
$$

在数学上，状态转移的过程可以用条件概率来描述。例如，状态 $s_1$ 和动作 $a_2$ 对应的状态转移可以用如下条件概率来描述：

$$
\begin{split}
p(s_1|s_1,a_2) &= 0 \\
p(s_2|s_1,a_2) &= 1 \\
p(s_3|s_1,a_2) &= 0 \\
p(s_4|s_1,a_2) &= 0 \\
p(s_5|s_1,a_2) &= 0 \\
\end{split}
$$

上面的条件概率告诉我们：当在状态 $s_1$ 采取动作 $a_2$ 时，智能体转移到状态 $s_2$ 的概率是 1 ，而转移到其他任意状态的概率是 0 。因此，在 $s_1$ 采取 $a_2$ 时，一定会导致智能体转移到 $s_2$ 。

策略会告诉智能体在每一个状态应该采取什么样的动作。

从直观上，策略可以通过箭头来描述。如果智能体执行某一个策略，那么它会从初始状态生成一条轨迹。

![](./images/6.png)

在数学上，策略可以通过条件概率来描述。我们常用 $\pi(a|s)$ 来表示在状态 $s$ 采取动作 $a$ 的概率。在上面的图中，状态 $s_1$ 对应的策略是：

$$
\begin{split}
\pi(a_1|s_1) &= 0 \\
\pi(a_2|s_1) &= 1 \\
\pi(a_3|s_1) &= 0 \\
\pi(a_4|s_1) &= 0 \\
\pi(a_5|s_1) &= 0 \\
\end{split}
$$

上面的条件概率表示在状态 $s_1$ 采取动作 $a_2$ 的概率为 1 ，而采取其他任意动作的概率都为 0 。其他的状态也可以用类似的条件概率来描述对应的策略。

上面的例子中的策略是确定性的。策略也可能是随机性的。例如下图，给出了一个随机策略：在状态 $s_1$ ，智能体有 0.5 的概率采取向右的动作，有 0.5 的概率采取向下的动作。此时在状态 $s_1$ 的策略是：

$$
\begin{split}
\pi(a_1|s_1) &= 0 \\
\pi(a_2|s_1) &= 0.5 \\
\pi(a_3|s_1) &= 0.5 \\
\pi(a_4|s_1) &= 0 \\
\pi(a_5|s_1) &= 0 \\
\end{split}
$$

![](./images/7.png)

## 奖励

奖励是强化学习中最独特的概念之一。

在一个状态执行一个动作后，智能体会获得奖励 $r$ 。 $r$ 是一个实数，它是状态 $s$ 和动作 $a$ 的函数，可以写成 $r(s,a)$ 。其值可以是正数、负数或者零。不同的奖励值对智能体最终学习到的策略有不同的影响。一般来说，正的奖励表示我们鼓励智能体采取相应的动作：负的奖励表示我们不鼓励智能体采取该动作。另外，如果 $r$ 是负数，此时称之为“惩罚”更为合适，不过我们一般不加区分的统一称之为“奖励”。

在网格世界的例子中，我们可以设置如下奖励：

- 如果智能体试图越过四周边界，设 $r_{boundary} = -1$
- 如果智能体试图进入禁止区域，设 $r_{forbidden} = -1$
- 如果智能体到达了目标区域，设 $r_{target} = +1$
- 在其他情况下，智能体获得的奖励为 $r_{other} = 0$

要注意的是，当智能体到达目标状态 $s_9$ 之后，它也许会持续执行策略，进而持续获得奖励。例如，如果智能体在 $s_9$ 采取动作 $a_5$ （保持不动），下一个状态依然是 $s_0$ ，此时会继续获取奖励 $r_{target} = +1$ 。如果智能体在 $s_9$ 执行动作 $a_2$ （向右移动），会试图越过右侧边界，因此会被反弹回来，此时下一个状态也是 $s_9$ ，但奖励是 $r_{boundary} = -1$ 。

奖励实际上是人机交互的一个重要手段:我们可以设置合适的奖励来引导智能体按照我们的预期来运动。例如，通过上述奖励设置，智能体会尽可能避免越过边界、避免进入禁止区域、力争进入目标区域。设计合适的奖励来实现我们的意图是强化学习中的一个重要环节。然而对于复杂的任务，这一环节可能并不简单，它需要用户能很好地理解所给定的任务。尽管如此，奖励的设计可能仍然比使用其他专业工具来设计策略更容易，这也是强化学习受众比较广的原因之一。

当智能体位于一个状态，并采取一个动作时，就可以获得对应的奖励。

如果我们每次处于一个状态时，都采取最大奖励的动作，那么最后能到达目标区域吗？或者说，能找到最好的策略吗？答案是否定的。

因为每次处于一个状态，并采取一个动作时，获取的奖励叫做“即时奖励”。也就是采取动作后立即获得的奖励。

如果要找到一个好的策略，必须考虑更加长远的“总奖励”。每次都选择奖励最大的动作，不一定能带来最大的总奖励。

> 大家可以结合人生来理解一下这个问题。人们并不总是会考虑长远利益，更容易被短期利益冲昏头脑。
> 在深度学习中，“梯度下降”算法只会找到局部最小值，因为深度学习解决的问题是“凹优化”，而非“凸优化”。是同样的道理，所以需要引入各种防止过拟合的方法，例如Dropout等等。

为了描述一般化的奖励过程，我们可以使用条件概率： $p(r|s,a)$ 来表示在状态 $s$ 采取动作 $a$ 后得到奖励 $r$ 的概率。例如，上面的例子里面，对于状态 $s_1$ ，有：

$$
p(r=-1|s_1,a_1) = 1 \\ p(r\neq-1|s_1,a_1) = 0
$$

上面的条件概率，其实描述的是“确定性奖励”，当然就有“随机性奖励”。

例如学生考试，如果努力学习，则会获得正的奖励，但得到的分数可能是随机的，90分，91分，... 因为影响分数的因素太多了。

## 轨迹、回报、回合

一条轨迹（trajectory）指的是一个“状态-动作-奖励”的链条。例如之前例子中的策略，智能体从 $s_1$ 出发会得到如下轨迹：

$$
s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9
$$

沿着一条轨迹，智能体会得到一系列的即时奖励，这些即时奖励之和被称为回报（return）。例如，上面的轨迹对应的回报是：

$$
return = 0+0+0+1=1
$$

回报由即时奖励和未来奖励组成。这里，即时奖励实在初始状态执行动作后立刻获得的奖励：未来奖励指的是离开初始状态后获得的奖励之和。例如，上述轨迹对应的即时奖励是 0 ，但是未来奖励是 1 ，因此总奖励是 1 。另外，回报也成为总奖励，或者累积奖励。



## 强化学习问题

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

## 价值函数

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

$$
V^*(s) = \max_a Q^* (s,a)
$$

这些关系直接遵循刚刚给出的定义：你能证明它们吗？

### 最优 Q 函数和最优动作

最优动作-价值函数 $Q^*(s,a)$ 与最优策略选择的动作之间存在重要联系。根据定义， $Q^*(s,a)$ 给出了从状态 $s$ 开始，采取（任意）动作 $a$ ，然后始终按照最优策略行动的预期回报。

$s$ 中的最优策略将选择能够最大化从 $s$ 开始的预期回报的行动。因此，如果我们有 $Q^*$ ，我们就可以通过以下方式直接获得最优行动 $a^*(s)$

$$
a^*(s) = \arg \max_a Q^* (s,a)
$$

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

## 优势函数

在强化学习中，有时我们不需要描述某个动作的绝对优劣，而只需要描述它平均比其他动作好多少。也就是说，我们想知道该动作的相对优势 。我们用优势函数来精确地表述这个概念。

与策略 $\pi$ 对应的优势函数 $A^{\pi}(s,a)$ 描述了在状态 $s$ 下采取特定行动 $a$ 比根据 $\pi(\cdot|s)$ 随机选择行动（假设你此后一直按照 $\pi$ 行动）要好多少。从数学上讲，优势函数定义为

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
$$

优势函数对于策略梯度方法至关重要。

## 形式化

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

这里，我们考虑随机参数化策略 $\pi_{\theta}$ 的情况。我们的目标是最大化预期收益 $J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}E[{R(\tau)}]$ 。为了便于推导，我们取 $R(\tau)$ 来表示有限期限下的没有折扣的收益 ，但无限期限下的有折扣的收益的推导过程几乎相同。

我们希望通过 **梯度上升** 来优化策略，例如

$$
\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.
$$

策略性能的梯度 $\nabla_{\theta} J(\pi_{\theta})$ 称为 **策略梯度** ，以这种方式优化策略的算法称为 **策略梯度算法** 。

- 最简单的策略梯度算法
- PPO（近端策略优化算法）
- DPO（直接策略优化算法）
- GRPO（组相对策略优化算法）

要实际使用该算法，我们需要一个可以数值计算的策略梯度表达式。这涉及两个步骤：

1. 推导策略性能的解析梯度，该解析梯度最终呈现为期望值的形式；
2. 形成该期望值的样本估计值，该样本估计值可以通过有限数量的代理-环境交互步骤的数据计算得出。

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
\begin{split}
\nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
&= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).
\end{split}
$$

综合起来，我们得出以下结论：

> [!NOTE]
> 基于策略梯度的推导
>
> ![](./images/2.svg)

这是一个期望，这意味着我们可以用样本均值来估计它。如果我们收集一组轨迹 $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$ ，其中每条轨迹都是通过让代理使用策略 $\pi_{\theta}$ 在环境中行动而获得的，那么策略梯度可以用以下公式来估计：

$$
\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),
$$

其中 $|\mathcal{D}|$ 是 $\mathcal{D}$ （此处为 $N$ ）中的轨迹数。

最后一个表达式是我们想要的可计算表达式的最简单版本。假设我们已经以一种允许计算 $\nabla_{\theta} \log \pi_{\theta}(a|s)$ 的方式表示了我们的策略，并且如果我们能够在环境中运行该策略来收集轨迹数据集，那么我们就可以计算策略梯度并采取更新步骤。

# 第四章 最简单的策略梯度算法

1. 实现策略网络

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

此模块构建了使用前馈神经网络分类策略的模块和函数。（请参阅第一章中的 “随机策略” 部分进行复习。） `logits_net` 模块的输出可用于构建对数概率和动作概率，而 `get_action` 函数则根据从对数计算出的概率对动作进行采样。（注意：此 `get_action` 函数假设只提供一个 `obs` ，因此只有一个整数动作输出。因此它使用了 `.item()` ，用于获取只有一个元素的张量的内容 。）

本例中的很多工作是由 L36 上的 `Categorical` 对象完成的。这是一个 PyTorch Distribution 对象，它封装了一些与概率分布相关的数学函数。具体来说，它包含一个从分布中采样的方法（我们在 L40 上用到）和一个计算给定样本对数概率的方法（我们稍后会用到）。由于 PyTorch 分布对强化学习非常有用，请查看它们的文档来了解它们的工作原理。

> [!NOTE]
>
> 友情提醒！当我们谈论具有“logits”的分类分布时，我们的意思是，每个结果的概率由 `logits` 的 `Softmax` 函数给出。也就是说，在 `logits` 为 $x_j$ 的分类分布下，动作 $j$ 的概率为：
>
> $$
> p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}
> $$

2. 实现损失函数

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
\underset{x \sim P_{\theta}}E[{\nabla_{\theta} \log P_{\theta}(x)}] = 0.
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
> \begin{split}
> 0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
> &= \int_x \nabla_{\theta} P_{\theta}(x) \\
> &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
> \therefore 0 &= \underset{x \sim P_{\theta}}E{\nabla_{\theta} \log P_{\theta}(x)}.
> \end{split}
> $$
> 

## 别让过去分散你的注意力

检查我们最新的策略梯度表达式：

$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}E[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}].
$$

按照这个梯度迈出一步，每个动作的对数概率都会与 $R(\tau)$ （所有获得奖励的总和）成比例地上升。但这没有多大意义。

代理实际上应该只根据行动的后果来强化行动。采取行动之前获得的奖励与该行动的效果无关：只有行动之后获得的奖励才有效。

事实证明，这种直觉在数学上是成立的，我们可以证明策略梯度也可以表示为

$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}E[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}].
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
\underset{a_t \sim \pi_{\theta}}E[{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)}] = 0.
$$

这使我们能够在策略梯度表达式中添加或减去任意数量的类似项，而无需改变其期望值：

$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}E[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}]
$$

任何以此方式使用的函数 $b$ 都称为基线 。

最常见的基线选择是策略值函数 $V^{\pi}(s_t)$ 。回想一下，如果智能体从状态 $s_t$ 开始，然后在其余生中按照策略 $\pi$ 行动，则该函数表示智能体获得的平均回报。

从经验上看，选择 $b(s_t) = V^{\pi}(s_t)$ 具有降低策略梯度样本估计方差的理想效果。这可以实现更快、更稳定的策略学习。从概念角度来看，它也极具吸引力：它编码了这样一种直觉：如果代理获得了预期的结果，它应该对此“感觉”中立。

实际上， $V^{\pi}(s_t)$ 无法精确计算，因此必须进行近似计算。这通常使用神经网络 $V_{\phi}(s_t)$ 来实现，该神经网络与策略同时更新（因此价值网络始终近似于最新策略的价值函数）。

在大多数策略优化算法（包括 VPG、TRPO、PPO 和 A2C）的实现中，学习 $V_{\phi}$ 的最简单方法是最小化均方误差目标：

$$
\phi_k = \arg \min_{\phi} \underset{s_t, \hat{R}_t \sim \pi_k}E[{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2}]
$$

其中 $\pi_k$ 是第 $k$ 个周期的策略。这是通过一步或多步梯度下降完成的，从先前的值参数 $\phi_{k-1}$ 开始。

## 策略梯度的其他形式

到目前为止，我们已经看到策略梯度具有一般形式

$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}E[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t}]
$$

其中 $\Phi_t$ 可以是以下任意一项

$$
\Phi_t = R(\tau),
$$

或者

$$
\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),
$$

或者

$$
\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t).
$$

尽管方差不同，所有这些选择都会导致相同的策略梯度预期值。事实证明，还有两个更有效的权重选择 $\Phi_t$ ，了解这些选择很重要。

1. 在线策略动作价值函数。 选择

$$
\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)
$$

也是有效的。请参阅此页面 （可选）获取此声明的证明。

2. 优势函数。 回想一下， 一个动作的优势 （定义为 $A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)$ ）描述了它相对于其他动作（相对于当前策略）平均而言有多好或多差。这个选择，

$$
\Phi_t = A^{\pi_{\theta}}(s_t, a_t)
$$

也是有效的。证明是，它等价于使用 $\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$ ，然后使用价值函数基线，我们始终可以自由地这样做。

使用优势函数来制定策略梯度是非常常见的，并且存在许多不同的方法来估计不同算法所使用的优势函数。
