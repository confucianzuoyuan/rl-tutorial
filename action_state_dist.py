import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# 定义一个简单的策略网络，输入状态，输出动作概率
class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimplePolicy, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        logits = self.fc(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

# 假设状态维度为4，动作空间大小为3
state_dim = 4
action_dim = 3

# 创建策略网络实例
policy = SimplePolicy(state_dim, action_dim)

# 构造一个示例状态张量（可以是环境返回的状态）
state = torch.tensor([0.5, -0.2, 0.1, 0.0], dtype=torch.float32)

# 计算动作概率
action_probs = policy(state)

# 根据动作概率构造离散分布
dist = Categorical(action_probs)

# 采样动作
# 根据概率分布随机抽样，动作 a 被选中的概率就是 P(a)
# 这会引入探索性，智能体不会总是选择同一个动作，有助于强化学习训练。
# 如果每次都选择概率最大的动作，则会丧失探索性。
action = dist.sample()

# 打印关键数据
print("状态 state:", state)
print("动作概率 action_probs:", action_probs)
print("采样动作 action:", action.item())