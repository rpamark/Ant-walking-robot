import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor网络（与训练时保持一致）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        z = normal.rsample()
        action = torch.tanh(z) * self.action_bound
        
        return action, None

# SAC Agent（仅用于测试）
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound=1.0):
        self.actor = Actor(state_dim, action_dim, action_bound=action_bound).to(device)
        self.actor.eval()  # 设置为评估模式
    
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=device)
            if 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            else:
                self.actor.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def select_action(self, state, evaluate=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

def test_ant_3d_render():
    # 创建环境，启用3D渲染
    env = gym.make("Ant-v5", render_mode="human")
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    print(f"Environment Info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Action bound: {action_bound}")
    
    # 创建智能体
    agent = SACAgent(state_dim, action_dim, action_bound=action_bound)
    
    # 加载最佳模型
    model_path = "models/best_model.pth"
    if not agent.load_model(model_path):
        print("Failed to load model. Exiting.")
        env.close()
        return
    
    print("开始3D渲染测试...")
    print("按Ctrl+C退出测试")
    
    try:
        # 运行多个测试回合
        for episode in range(10):
            state, info = env.reset(seed=42 + episode)  # 使用不同种子
            episode_reward = 0
            step_count = 0
            done = False
            
            print(f"\n=== 测试回合 {episode + 1} 开始 ===")
            
            while not done:
                try:
                    # 选择确定性动作（评估模式）
                    action = agent.select_action(state, evaluate=True)
                    
                    # 执行动作
                    state, reward, terminated, truncated, step_info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    # 渲染环境（3D可视化）
                    env.render()
                    
                    # 实时显示信息
                    x_pos = step_info.get('x_position', 0)
                    y_pos = step_info.get('y_position', 0)
                    z_pos = step_info.get('z_position', 0) if 'z_position' in step_info else 0
                    
                    print(f"\r步骤: {step_count:3d} | "
                          f"即时奖励: {reward:7.2f} | "
                          f"累计奖励: {episode_reward:8.2f} | "
                          f"位置: ({x_pos:6.2f}, {y_pos:6.2f}, {z_pos:6.2f})", 
                          end='', flush=True)
                    
                    # 控制渲染速度，使动作更流畅
                    time.sleep(0.01)
                    
                    # 检查是否结束
                    done = terminated or truncated
                    
                except KeyboardInterrupt:
                    print(f"\n\n用户中断测试")
                    done = True
                    break
            
            print(f"\n回合 {episode + 1} 完成:")
            print(f"  总步数: {step_count}")
            print(f"  总奖励: {episode_reward:.2f}")
            print(f"  最终位置: ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})")
            
            if done:
                break
                
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    
    finally:
        # 关闭环境
        env.close()
        print("\n环境已关闭")

def test_ant_multiple_episodes():
    """测试多个回合并显示统计信息"""
    # 创建环境
    env = gym.make("Ant-v5", render_mode="human")
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    # 创建智能体
    agent = SACAgent(state_dim, action_dim, action_bound=action_bound)
    
    # 加载最佳模型
    model_path = "models/best_model.pth"
    if not agent.load_model(model_path):
        print("无法加载模型文件")
        env.close()
        return
    
    print("=== 多回合测试模式 ===")
    
    # 统计信息
    total_episodes = 15
    rewards = []
    steps_list = []
    
    try:
        for episode in range(total_episodes):
            state, _ = env.reset(seed=100 + episode)
            episode_reward = 0
            step_count = 0
            done = False
            
            print(f"\n回合 {episode + 1}/{total_episodes} 开始")
            
            while not done and step_count < 1000:  # 限制最大步数
                # 选择动作
                action = agent.select_action(state, evaluate=True)
                
                # 执行动作
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 渲染
                env.render()
                
                # 显示进度
                if step_count % 50 == 0:
                    x_pos = info.get('x_position', 0)
                    y_pos = info.get('y_position', 0)
                    print(f"  步数: {step_count}, 奖励: {episode_reward:.2f}, "
                          f"位置: ({x_pos:.2f}, {y_pos:.2f})")
                
                # 控制速度
                time.sleep(0.01)
                
                # 检查结束条件
                done = terminated or truncated
            
            # 记录统计信息
            rewards.append(episode_reward)
            steps_list.append(step_count)
            
            print(f"回合 {episode + 1} 结束:")
            print(f"  步数: {step_count}")
            print(f"  奖励: {episode_reward:.2f}")
        
        # 显示最终统计
        print("\n=== 测试统计 ===")
        print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"最高奖励: {np.max(rewards):.2f}")
        print(f"最低奖励: {np.min(rewards):.2f}")
        print(f"平均步数: {np.mean(steps_list):.1f}")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        env.close()
        print("测试完成")

if __name__ == "__main__":
    print("Ant-v5 3D渲染测试程序")
    print("选择测试模式:")
    print("1. 连续测试模式")
    print("2. 多回合统计模式")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            test_ant_3d_render()
        elif choice == "2":
            test_ant_multiple_episodes()
        else:
            print("无效选择，运行默认测试模式")
            test_ant_3d_render()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行错误: {e}")