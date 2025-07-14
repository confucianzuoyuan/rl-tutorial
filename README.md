# 第一章 强化学习中的关键概念

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
> - RLHF: Reinforcement Learning From Human Feedback, 基于人类反馈的强化学习

**强化学习** 是研究智能体及其如何通过反复试验进行学习的学科。强化学习从数学的角度阐述了这样一种观点：对智能体的行为进行奖励或惩罚，会增加智能体在未来重复或放弃该行为的可能性。

## 强化学习的应用

**AlphaGo**

下棋的程序是 **智能体** 。在训练下棋的程序时，对下的一步好棋进行奖励，对下的一步臭棋进行惩罚，最终训练出一个超越人类的AI棋手。

**基于人类反馈的强化学习（RLHF）**

大语言模型作为 **智能体** 。对大语言模型输出的好的回答进行奖励，对大语言模型输出的不好的回答进行惩罚。进而改变大语言模型的参数（也就是“微调”），来让大语言模型的输出能够对齐到人类的偏好。

所谓对齐到人类的偏好，就是按照人类喜欢的方式输出回答。

## 强化学习中的关键概念和术语

![](./images/rl_diagram_transparent_bg.png)

![](./images/1.png)

强化学习的主要角色是 **智能体** 和 **环境** 。环境是智能体生存并与之交互的世界。在交互的每一步，智能体都会观察环境的状态，然后决定采取的行动。环境会随着智能体的行动而变化，但环境也可能自行变化。

智能体可能能够观察到当前环境的完整状态，例如如果下棋的AI程序是智能体的话，那么这个智能体可以获取到当前环境（也就是棋盘）的所有信息。

智能体也可能无法观察到当前环境的完整状态，人作为生活在自然环境中的智能体，是无法获取整个自然环境的完整信息的，只能部分的获取环境的信息。

智能体还会感知来自环境的 **奖励** 信号，这是一个数值，用来告诉智能体当前环境的状态的好坏。智能体的目标是最大化智能体获得的累积奖励，也就是 **回报** 。强化学习方法是智能体学习行为以实现其目标的方法。

我们大概知道了强化学习主要做的事情是什么，接下来，我们分别讨论一些概念和术语。

## 状态和观察

状态 $s$ 是对环境的状态的完整描述。状态中不存在任何隐藏于环境之外的信息。观察 $o$ 是对状态的部分描述，可能会遗漏一些环境的信息。

- 对于棋类，下棋的AI程序或者人类作为智能体，可以获取环境状态的完整描述，因为棋盘没有任何隐藏信息。
- 对于超级玛丽游戏，玩家作为智能体只能看到当前超级玛丽所处的画面，无法获取其它的游戏画面（例如无法看到其它关卡的游戏画面）。由于无法获取游戏环境的所有信息，所以只能叫做“观察”。

如果将深度学习技术和强化学习技术结合使用，则称为 **深度强化学习** 。号称人工智能中，最具有吸引力的话题。也被认为是通往 AGI 的唯一的一条路。

> 具体如何结合起来，后面我们会看到。

在深度强化学习中，我们几乎总是用向量、矩阵或张量来表示状态和观察到的值。例如，视觉观察到的值可以用传感器采集到的图像的像素值的 `RGB` 矩阵表示；机器人的状态可以用其关节角度和速度表示。

当智能体能够观察到环境的完整状态时，我们称该环境是 **完全可观察的** 。当智能体只能看到部分观察结果时，我们称该环境是 **部分可观察的** 。

## 动作空间(Action Space)

不同的环境允许不同类型的动作。给定环境中所有有效动作的集合通常称为动作空间 。某些环境，例如象棋，具有 **离散的动作空间** ，其中智能体针对某个棋子只能使用有限的动作（走法），例如马走日，象走田等等。其他环境，例如智能体在物理世界中控制机器人的环境，具有 **连续的动作空间** 。在连续空间中，动作是向量、矩阵或者张量。

- 离散动作空间：超级玛丽中玩家只有“起跳”，“蹲下”等有限的几个动作。
- 连续动作空间：自动驾驶，方向盘旋转1度，1.1度，1.2度，......，有无限多种动作。

这种区别对深度强化学习的方法有着相当深远的影响。一些算法只能在一种情况下直接应用，而对于另一种情况则需要进行大量的重新设计。

## 策略(Policy)

**策略** 会告诉智能体在每一个状态应该采取什么样的动作。

> 日常生活中有什么例子？
>
> 我是一个智能体，我的策略是：当环境处于没有老师的情况下，采取打游戏的动作，当环境处于有老师的情况下，采取写代码的动作。

在数学上，策略可以通过条件概率来描述。我们常用 $\pi(a|s)$ 来表示在状态 $s$ 采取动作 $a$ 的概率（注意这里的 $\pi$ 不是圆周率）。这个概率对每一个状态和每一个动作都有定义。

也就是在环境处于某个状态时，智能体采取某个动作的概率是多少。

![](./images/3.png)

因为策略本质上是智能体的大脑，所以用“策略”一词代替“智能体”并不罕见，例如说“策略试图最大化奖励”，意思是“智能体试图最大化奖励”。

策略本质上是一个函数，接收状态作为输入，输出各种动作的概率。所以策略函数当然可以是一个深度学习模型。

如果我们将 DeepSeek 大模型看作一个智能体或者策略的话，那么状态是：用户输入的提示词（可能还有之前的上下文）。由于大模型的算法是接收用户的输入，并预测下一个词（predict next token, pnt）。所以动作其实就是大模型根据提示词预测出来的下一个词的概率分布。然后根据温度值的不同，可能选择不同的词作为下一个词。

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

奖励实际上是人机交互的一个重要手段：我们可以设置合适的奖励来引导智能体按照我们的预期来运动。例如，通过上述奖励设置，智能体会尽可能避免越过边界、避免进入禁止区域、力争进入目标区域。设计合适的奖励来实现我们的意图是强化学习中的一个重要环节。然而对于复杂的任务，这一环节可能并不简单，它需要用户能很好地理解所给定的任务。尽管如此，奖励的设计可能仍然比使用其他专业工具来设计策略更容易，这也是强化学习受众比较广的原因之一。

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

回报由即时奖励和未来奖励组成。这里，即时奖励实在初始状态执行动作后立刻获得的奖励：未来奖励指的是离开初始状态后获得的奖励之和。例如，上述轨迹对应的即时奖励是 0 ，但是未来奖励是 1 ，因此总奖励是 1 。另外，回报也称为“总奖励”，或者“累积奖励”。

![](./images/8.png)

回报可以用于评价一个策略的好坏。例如，上面图中两个策略，可以分别计算两条轨迹对应的回报，进而判断哪个策略更好。具体来说，左边的策略的回报为 1 。右边策略的回报为 0 。所以我们知道左边的策略更好，这符合我们的直觉。因为右边的策略经过了禁止区域。

刚刚我们提到的轨迹都是有限长的。而轨迹也可以是无限长的。如果上图左边的策略到达状态 $s_9$ 之后，并没有停止，而是继续执行策略。那么如果策略是停在 $s_9$ 不动。那么智能体会不断的获得 “+1” 奖励。所对应的轨迹也是无限长的。

$$
s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9 \xrightarrow[r=1]{a_5} s_9 \xrightarrow[r=1]{a_5} s_9 \xrightarrow[r=1]{a_5} s_9 \xrightarrow[r=1]{a_5} s_9 \ldots
$$

此时，如果我们直接把这条轨迹上所有的奖励求和，来计算回报。那么得到的是：

$$
return = 0 + 0 + 0 + 1 + 1 + 1 + 1 + \cdots = \infty
$$

这里因为轨迹是无限长的，所以计算的回报会发散到无穷。此时，我们需要引入折扣回报的概念。令 $\gamma \in (0,1)$ 为折扣因子。折扣回报是所有折扣奖励的综合，即为不同时刻得到的奖励添加相应的折扣再求和：

$$
\begin{split}
\text{discounted return} &= 0 + \gamma 0 + \gamma^2 0 + \gamma^3 1 + \gamma^4 1 + \gamma^5 1 + \ldots \\
&= \gamma^3(1+\gamma+\gamma^2+\gamma^3+\ldots) \\
&= \gamma^3(\frac{1}{1-\gamma})
\end{split}
$$

由于 $\gamma \in (0, 1)$ ，所以折扣回报的值就不再是无穷的了。而是一个有限值。

折扣因子的引入具有以下用途。第一，它允许考虑无限长的轨迹，而不用担心回报会发散到无穷；第二，折扣因子可以用来调整对近期或远期奖励的重视程度。具体来说，如果 $\gamma$ 接近 0 ，则智能体会更加重视近期奖励，最后所得到的策略也会比较短视。如果 $\gamma$ 接近 1 ，则智能体会更加重视远期奖励，最后所得到的策略也会更具有远见，例如敢于冒险在近期获得负面奖励来获得更大的未来奖励。

当执行一个策略进而与环境交互时，智能体从初始状态开始到终止状态(terminal state)停止的过程被称为一个回合(episode)或尝试(trial)。这里的Episode有多种翻译，例如回合、情节、集、轮等，其中“回合”能比较好地描述其内涵。不过，它应该与神经网络训练过程中的回合(epoch)加以区分。

回合和轨迹在概念上非常类似：回合通常被认为是一条有限长的轨迹。如果一个任务最多有有限步，那么这样的任务称为回合制任务(episodic task)。如果一个任务没有终止状态，则意味着智能体与环境的交互永不停止，这种任务被称为持续性任务(continuing task)。为了在数学上可以不加区分地对待这两种任务，我们可以把回合制任务转换为持续性任务。为此，我们只需要合理定义智能体在到达终止状态后的状态和动作等元素即可。具体来说，在回合制任务中到达终止状态后，我们有如下两种方式将其转换为持续性任务。

第一，我们可以将终止状态视为一个特殊状态，即专门设计其动作空间或状态转移，从而使智能体永远停留在此状态，这样的状态被称为吸收状态(absorbing state)，即一旦达到这样的状态就会一直停留在该状态。例如，对于目标状态 $s_9$ ，我们可以指定其动作空间为 $A(s_9) = \{a_5\}$ ，即到达这个状态后唯一可执行的动作就是原地不动。

第二，我们可以将终止状态视为一个普通状态，即将其与其他状态一视同仁，此时智能体可能会离开该状态并再次回来。由于每次到达 $s_9$ 都可以获得 $r=1$ 的正奖励，可以预期的是智能体最终会学会永远停留在 $s_9$ 以获得更多的奖励。值得注意的是，将回合制任务转换为持续性任务需要使用折扣因子，以避免回报趋于无穷。

## 马尔可夫决策过程

前面几节通过例子直观地介绍了强化学习中的基本概念。本节将在马尔可夫决策过程（Markov decision process, MDP）的框架下以更加正式的方式介绍这些概念。

马尔可夫决策过程是描述随机动态系统的一般框架，它并不局限于强化学习，而是强化学习需要依赖于这个框架。马尔可夫决策过程涉及以下关键要素。

- 集合：
    - 状态空间：所有状态的集合，记为 $S$ 。
    - 动作空间：与每个状态 $s\in S$ 相关联的所有动作的集合，记为 $A(s)$ 。
    - 奖励集合：与 $(s,a)$ 相关联的所有奖励的集合，记为 $R(s,a)$ 。

- 模型：
    - 状态转移概率：在状态 $s$ 采取动作 $a$ 时，智能体转移到状态 $s{'}$ 的概率为 $p(s'|s,a')$ 。对于任意 $(s,a)$ ，都有 $\sum_{s'\in S} p(s'|s,a)=1$ 。
    - 奖励概率：在状态 $s$ 采取动作 $a$ 时，智能体获得奖励 $r$ 的概率是 $p(r|s,a)$ 。对于任意 $(s,a)$ ，都有 $\sum_{r\in{R(s,a)}}p(r|s,a)=1$ 成立。

- 策略：在状态 $s$ ，智能体采取动作 $a$ 的概率是 $\pi(a|s)$ 。对于任意 $s\in S$，都有 $\sum_{a\in{A(s)}}\pi(a|s)=1$ 。

- 马尔可夫性质：马尔可夫性质（Markov Property）指的是随机过程中的无记忆性质，它在数学上表示为：

$$
p(s_{t+1}|s_t,a_t,s_{t-1},a_{a-1},\ldots,s_0,a_0) = p(s_{t+1}|s_t,a_t) \\
p(r_{t+1}|s_t,a_t,s_{t-1},a_{a-1},\ldots,s_0,a_0) = p(r_{t+1}|s_t,a_t)
$$

其中 $t$ 表示当前时刻，$t+1$ 表示下一个时刻。上面的式子表示下一个状态和奖励仅依赖于当前时刻的状态和动作，而与之前时刻的状态和动作无关。

强化学习的过程涉及智能体与环境的交互，智能体之外的一切都被视为环境(environment)。第一，智能体是一个感知者，例如具有眼睛能够感知并理解当前的状态;第二，智能体是一个决策者，例如具有大脑能够做出决策，知道在什么状态应该采取什么行动;第三，智能体是一个执行者，例如具有操作机构来执行策略所指示的动作，从而改变状态并得到奖励。

## 价值函数

强化学习（Reinforcement Learning, RL）中的**价值函数**是核心概念之一，它用来衡量在某个状态或状态-动作对下，智能体（Agent）未来能够获得的期望回报。价值函数帮助智能体评估当前策略的好坏，从而指导策略的改进。

### 1. 价值函数的定义

#### 状态价值函数（State-Value Function）

状态价值函数记作 $V^\pi(s)$，表示在策略 $\pi$ 下，从状态 $s$ 出发，智能体能获得的**期望累计回报**：

\[
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \mid S_0 = s \right]
\]

- $s$：当前状态
- $\pi$：策略，定义为在某状态下采取某动作的概率分布
- $\gamma \in [0,1]$：折扣因子，权衡未来奖励的重要性
- $R_{t+1}$：在时间步 $t$ 采取动作后获得的即时奖励

#### 状态-动作价值函数（Action-Value Function）

状态-动作价值函数记作 $Q^\pi(s,a)$，表示在策略 $\pi$ 下，从状态 $s$ 采取动作 $a$ 后，智能体能获得的期望累计回报：

\[
Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \mid S_0 = s, A_0 = a \right]
\]

### 2. 价值函数的作用

- **评价策略**：价值函数反映了在当前策略下，某状态或状态-动作对的“好坏”，即未来能获得多少奖励。
- **指导决策**：通过比较不同动作的价值，智能体可以选择价值更高的动作，从而优化策略。
- **算法基础**：许多强化学习算法（如动态规划、蒙特卡洛方法、时序差分学习）都基于估计和更新价值函数。

### 3. 最优价值函数

最优状态价值函数 $V^*(s)$ 和最优状态-动作价值函数 $Q^*(s,a)$ 定义为在所有可能策略中能获得的最大期望回报：

\[
V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s,a) = \max_\pi Q^\pi(s,a)
\]

### 4. 直观理解

- **价值函数告诉你“现在的状态值多少钱”**，即如果你从这里开始，按照某种策略行动，预计能获得多少奖励。
- **状态-动作价值函数告诉你“做某个动作值多少钱”**，帮助你判断当前动作是否值得。

## 强化学习问题

我们再来讨论一下前面的概念。首先轨迹的数学符号是 $\tau$ 。是 “状态” 和 “动作” 的序列。可以表示为以下公式：

$$
\tau = (s_0,a_0,s_1,a_1,\ldots)
$$

那么轨迹的最开始的状态 $s_0$ 是怎么来的呢？是从 “开始状态的分布” 随机采样的来的。开始状态的分布有时候会表示为：

$$
s_0 \sim \rho_0(\cdot)
$$

当环境处于状态 $s_t$ ，并且智能体采取动作 $a_t$ 的情况下，环境会跳转到状态 $s_{t+1}$ 。

当然 $s_{t+1}$ 是不确定的，也就是说智能体采取了某个行动，环境跳转到的状态可能是确定的，也可能是不确定的。统一表示为：

$$
s_{t+1} \sim P(\cdot|s_t, a_t)
$$

动作来自于智能体采取的策略。

轨迹（trajectories）经常被称作“回合”（episodes）或者“抛出”（rollouts）。

我们再来讨论一下“奖励和回报”。

奖励函数 $R$ 在强化学习中至关重要。它取决于环境的当前状态、刚刚采取的行动以及环境的下一个状态：

$$
r_t = R(s_t, a_t, s_{t+1})
$$

尽管这通常被简化为仅依赖于当前状态 $r_t = R(s_t)$ 或“状态-动作对” $r_t = R(s_t,a_t)$ 。

智能体的目标是最大化轨迹上的累积奖励，但这实际上可能意味着几件事。我们将用 $R(\tau)$ 来表示所有这些情况，这样一来，上下文就能清楚地表明我们指的是哪种情况，或者这无关紧要（因为相同的方程式适用于所有情况）。

一种回报是“有限期限的没有折扣的回报” ，它只是在固定步骤窗口内获得的奖励的总和：

$$
R(\tau) = \sum_{t=0}^T r_t
$$

另一种回报是“无限期的有折扣的回报” ，它是智能体曾经获得的所有奖励的总和，但会根据未来获得奖励的时间进行折现。此奖励公式包含一个折扣因子 $\gamma \in (0,1)$ ：

$$
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

但我们为什么要用折扣因子呢？难道我们不是只想获得所有奖励吗？我们当然想，但折扣因子既直观又数学上方便。直观上来说：现在的现金比以后的现金更好。数学上来说：无限期的奖励总和可能不会收敛到一个有限值，而且很难用方程式来处理。但是，有了折扣因子，并且在合理的条件下，无限期的奖励总和就会收敛。

无论选择何种回报衡量标准（无论是无限期折扣还是有限期不折扣），也无论选择何种策略，RL 中的目标都是选择一种策略，当智能体按照该策略行事时，该策略可以最大化 **预期回报** 。

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

## 优势函数

在强化学习中，有时我们不需要描述某个动作的绝对优劣，而只需要描述它平均比其他动作好多少。也就是说，我们想知道该动作的相对优势 。我们用优势函数来精确地表述这个概念。

与策略 $\pi$ 对应的优势函数 $A^{\pi}(s,a)$ 描述了在状态 $s$ 下采取特定行动 $a$ 比根据 $\pi(\cdot|s)$ 随机选择行动（假设你此后一直按照 $\pi$ 行动）要好多少。从数学上讲，优势函数定义为

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
$$

优势函数对于策略梯度方法至关重要。

# 第二章 策略优化简介

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

# 第三章 最简单的策略梯度算法

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

这段代码是强化学习中经典的 **REINFORCE** 算法计算策略梯度损失的部分，具体含义如下：

```python
loss = 0
for log_prob, R in zip(log_probs, discounted_rewards):
    loss -= log_prob * R  # REINFORCE loss
```

---

## 逐项解释

- `log_probs`：一个列表，包含智能体在一条轨迹（episode）中每一步选择动作的对数概率 \(\log \pi_\theta(a_t|s_t)\)。
- `discounted_rewards`：对应每一步的折扣累计奖励 \(R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}\)，反映从当前步开始未来的总回报。
- `zip(log_probs, discounted_rewards)`：将每一步的对数概率和对应的折扣奖励配对。

---

## REINFORCE 损失函数含义

REINFORCE 算法的目标是最大化期望回报，等价于最小化下面的损失：

\[
L(\theta) = - \sum_t \log \pi_\theta(a_t|s_t) \cdot R_t
\]

- 这里 \(\theta\) 是策略参数。
- \(\log \pi_\theta(a_t|s_t)\) 是策略在状态 \(s_t\) 下选择动作 \(a_t\) 的对数概率。
- \(R_t\) 是从时间步 \(t\) 开始的折扣累计奖励。

---

## 代码中实现

- `loss` 初始为0。
- 对每一步，计算 \(- \log \pi_\theta(a_t|s_t) \times R_t\)，累加到 `loss`。
- 负号是因为我们用梯度下降去最小化 `loss`，而实际上是最大化期望回报。

---

## 作用

- 通过对 `loss` 求梯度并反向传播，更新策略参数，使得高回报的动作概率被提升，低回报的动作概率被降低。
- 这是策略梯度方法的核心。

---

# 第四章 PPO（近端策略优化）

# 第五章 DPO（直接策略优化）

# 第六章 KTO

# 第七章 GRPO