import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 策略网络，输入状态，输出动作概率（2个动作）


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


def train():
    env = gym.make('CartPole-v1')
    # 状态空间
    # - 小车位置
    # - 小车速度
    # - 杆的角度
    # - 杆的角速度
    state_dim = env.observation_space.shape[0]
    # 动作空间
    # - 向左推力
    # - 向右推力
    # 推力是gym内部设置的固定值
    action_dim = env.action_space.n

    # Pi(a_t|s_t)
    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99  # 折扣因子

    max_episodes = 1000
    print_interval = 20

    for episode in range(max_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0)  # shape: [1, state_dim]
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            # log(P(action)) 计算动作action的对数概率
            log_prob = dist.log_prob(action)

            # 倒立摆的下一个状态 next_state
            # 当前这一步获得的奖励 reward
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            # 超过最大步数或者倒立摆倒下（失败），则 done = True
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # 计算折扣回报
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        # 标准化奖励，稳定训练
        discounted_rewards = (
            discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 计算损失函数
        loss = 0
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss -= log_prob * R  # REINFORCE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % print_interval == 0:
            print(f"Episode {episode+1}\tTotal reward: {sum(rewards):.2f}")
    torch.save(policy.state_dict(), "policy.pth")

# 训练函数省略，假设训练完成后有policy和env


def test(policy, env, episodes=3):
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Test Episode {ep+1}: total reward = {total_reward}")
    env.close()


if __name__ == "__main__":
    # 训练
    # train()

    # 创建环境和策略网络，加载训练好的模型参数（如果保存了）
    env = gym.make('CartPole-v1', render_mode='human')
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    # 如果有保存模型，可以加载：
    policy.load_state_dict(torch.load("policy.pth"))

    test(policy, env)
