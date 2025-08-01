import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
import os
from collections import deque
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor-Critic网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        for layer in [self.fc1, self.fc2, self.fc3, self.mean_layer, self.log_std_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制log_std范围
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # 重参数化采样
        z = normal.rsample()
        action = torch.tanh(z) * self.action_bound
        
        # 计算log概率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        # 预分配numpy数组以提高效率
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        index = self.position % self.capacity
        
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[index] = (state, action, reward, next_state, done)
        self.position += 1
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        return (torch.FloatTensor(batch_states).to(device),
                torch.FloatTensor(batch_actions).to(device),
                torch.FloatTensor(batch_rewards).unsqueeze(1).to(device),
                torch.FloatTensor(batch_next_states).to(device),
                torch.FloatTensor(batch_dones).unsqueeze(1).to(device))
    
    def __len__(self):
        return len(self.buffer)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound=1.0, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1000000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 网络
        self.actor = Actor(state_dim, action_dim, action_bound=action_bound).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        
        # 复制目标网络参数
        self._update_target_networks(tau=1.0)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # 学习步数
        self.learn_step = 0
        
    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if evaluate:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 更新Critic网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor网络
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新目标网络
        self._update_target_networks()
        
        self.learn_step += 1
    
    def save_model(self, path, episode=None):
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'learn_step': self.learn_step
        }
        # 如果提供了 episode，则保存它
        if episode is not None:
            save_dict['episode'] = episode
            
        torch.save(save_dict, path)
    
    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            self.learn_step = checkpoint['learn_step']
            print(f"Model loaded from {path}")
            # 如果保存了 episode，则返回它
            return checkpoint.get('episode', None)
        else:
            print(f"No model found at {path}")
            return None

# 训练函数
def train_ant():
    # 创建环境
    env = gym.make("Ant-v5", render_mode=None)  # 训练时不需要渲染
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bound: {action_bound}")
    
    # 创建智能体
    agent = SACAgent(state_dim, action_dim, action_bound=action_bound)
    
    start_episode = 0 # 默认从第0轮开始
    # 加载最新的模型（如果存在）
    loaded_episode = agent.load_model("latest_model.pth")
    if loaded_episode is not None:
        # 如果成功加载了模型，则从保存的 episode 之后开始
        start_episode = loaded_episode + 1
        print(f"从 episode {start_episode} 继续训练")
    
    # 训练参数
    max_episodes = 10000
    max_steps = 1000
    batch_size = 256
    save_frequency = 500
    best_reward = -np.inf
    
    # 创建保存目录
    os.makedirs("models", exist_ok=True)
    
    # 奖励记录
    reward_history = deque(maxlen=100)
    
    print("开始训练...")
    
    for episode in range(start_episode, max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # 选择动作 - 不再使用前10轮随机探索，始终使用智能体策略
            # if episode < 10:  # 移除或注释掉这个条件
            #     action = np.random.uniform(-action_bound, action_bound, action_dim)
            # else:
            action = agent.select_action(state) # 直接使用智能体选择动作
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # 检查是否结束
            if terminated or truncated:
                break
        
        # 学习
        # if len(agent.memory) >= batch_size and episode >= 10: # 移除 episode >= 10 的限制
        if len(agent.memory) >= batch_size:
            for _ in range(min(episode_steps, 100)):  # 每个episode最多学习100次
                agent.learn(batch_size)
        
        # 记录奖励
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        
        # 打印信息
        print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, "
              f"Avg Reward: {avg_reward:.2f}, Steps: {episode_steps}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model("models/best_model.pth")
            print(f"新最佳模型已保存! 奖励: {best_reward:.2f}")
        
        # 定期保存检查点
        if (episode + 1) % save_frequency == 0:
            agent.save_model(f"models/checkpoint_{episode+1}.pth")
            print(f"检查点 {episode+1} 已保存")
        
        # 保存最新模型，并保存当前 episode 计数
        agent.save_model("latest_model.pth", episode)
    
    env.close()
    print("训练完成!")

# 测试函数
def test_ant():
    # 创建环境（启用渲染）
    env = gym.make("Ant-v5", render_mode="human")
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    # 创建智能体
    agent = SACAgent(state_dim, action_dim, action_bound=action_bound)
    
    # 加载最佳模型
    agent.load_model("models/best_model.pth")
    
    print("开始测试...")
    
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        print(f"测试回合 {episode + 1} 开始")
        
        while not done:
            # 选择确定性动作
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # 渲染环境
            env.render()
            
            # 检查是否结束
            done = terminated or truncated
            
            # 打印信息
            print(f"Reward: {reward:.4f}, x_position: {info.get('x_position', 0):.4f}, "
                  f"y_position: {info.get('y_position', 0):.4f}", end='\r')
            
            # 控制渲染速度
            time.sleep(0.05)
        
        print(f"\n测试回合 {episode + 1} 完成，总奖励: {episode_reward:.2f}\n")
    
    env.close()
    print("测试完成!")

if __name__ == "__main__":
    # 训练模型
    train_ant()
    
    # 测试模型
    # test_ant() # 可以选择性地运行测试