import gymnasium as gym
import numpy as np
import time

# 创建环境，启用渲染
env = gym.make("Ant-v5", render_mode="human")

# 重置环境，获取初始观测
observation, info = env.reset(seed=42)

# 运行多个 episode
for episode in range(50):  # 运行5个回合
    print(f"Episode {episode + 1} started.")
    observation, info = env.reset()
    done = False

    while not done:
        # 随机采样动作：每个关节施加 [-1, 1] 范围内的 torque
        action = np.random.uniform(low=-1.0, high=1.0, size=env.action_space.shape)

        # 在环境中执行动作
        observation, reward, terminated, truncated, info = env.step(action)

        # 渲染环境
        env.render()

        # 判断是否结束
        done = terminated or truncated

        # 可选：打印 reward 和位置信息
        print(f"Reward: {reward:.4f}, x_position: {info['x_position']:.4f}, y_position: {info['y_position']:.4f}")
        time.sleep(0.1)

    print(f"Episode {episode + 1} finished.\n")

# 关闭环境
env.close()