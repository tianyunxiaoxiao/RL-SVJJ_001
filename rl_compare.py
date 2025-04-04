# (4) 强化学习模型对比
# === rl_compare.py ===
"""
用于对比和训练强化学习模型的脚本文件。
包含DQN、PPO、TD3、Actor-Critic等强化学习算法的实现。
使用stable-baselines3库实现真实的强化学习模型。
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RL_COMPARE_CONFIG

# 自定义环境，用于强化学习模型训练
class SimpleEnvForRL(gym.Env):
    """
    一个简易的环境示例：对每个样本做一个动作(0=call,1=put)，并返回一个简单的reward
    适配gymnasium接口，以便与stable-baselines3兼容
    """
    def __init__(self, X, y_call, y_put):
        super(SimpleEnvForRL, self).__init__()
        self.X = X
        self.y_call = y_call
        self.y_put = y_put
        self.n_samples = X.shape[0]
        self.index = 0
        
        # 定义动作空间：离散动作空间，2个动作 (0=call, 1=put)
        self.action_space = spaces.Discrete(2)
        
        # 定义观察空间：连续观察空间，维度为X的特征数
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # 重置环境
        super().reset(seed=seed)
        self.index = 0
        return self.X[self.index].astype(np.float32), {}  # 返回初始观察和空信息字典

    def step(self, action):
        """
        执行一步动作并返回结果
        reward = - |真实call - X[index][0]| (若action=0) 
                或  - |真实put  - X[index][1]| (若action=1)
        """
        if action == 0:
            reward = -abs(self.y_call[self.index] - self.X[self.index][0])
        else:
            reward = -abs(self.y_put[self.index] - self.X[self.index][1])

        self.index += 1
        done = (self.index >= self.n_samples)
        terminated = done
        truncated = False
        
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32) if done else self.X[self.index].astype(np.float32)
        return obs, reward, terminated, truncated, {}

# 自定义Actor-Critic模型（使用PyTorch实现）
class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCriticModel, self).__init__()
        # 共享网络层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        state_values = self.critic(shared_features)
        return action_probs, state_values

class ActorCriticAgent:
    def __init__(self, input_dim, n_actions, hidden_dim=128, lr=0.001, gamma=0.99, entropy_weight=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticModel(input_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        
        # 存储经验
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, state_value = self.model(state)
        
        # 创建概率分布
        dist = Categorical(action_probs)
        
        # 根据概率采样动作
        action = dist.sample()
        
        # 保存log概率和value
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        
        # 计算熵以鼓励探索
        entropy = dist.entropy()
        self.entropies.append(entropy)
        
        return action.item()
    
    def update(self, next_value=0):
        # 计算折扣回报
        returns = []
        R = next_value
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        
        # 转换为tensor
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values)
        entropies = torch.stack(self.entropies)
        
        # 计算优势函数
        advantage = returns - values.detach()
        
        # 计算Actor损失（策略梯度）
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # 计算Critic损失（值函数估计）
        critic_loss = F.mse_loss(values, returns)
        
        # 计算熵损失（鼓励探索）
        entropy_loss = -entropies.mean()
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss + self.entropy_weight * entropy_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空缓存
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
        return loss.item()

# 使用Actor-Critic自定义实现进行训练
def train_actor_critic(env, episodes=1000, hidden_dim=128, lr=0.001, gamma=0.99):
    """
    使用自定义实现的Actor-Critic算法训练模型
    """
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ActorCriticAgent(input_dim, n_actions, hidden_dim, lr, gamma)
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储奖励
            agent.rewards.append(reward)
            episode_reward += reward
            
            # 更新状态
            state = next_state
            
            # 如果回合结束，更新策略
        if done:
                agent.update()
        
        total_rewards.append(episode_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}, 平均奖励: {avg_reward:.2f}")
    
    return agent, total_rewards

# 创建用于训练的环境包装器
def make_env(X, y_call, y_put):
    def _init():
        env = SimpleEnvForRL(X, y_call, y_put)
        return env
    return _init

# 使用stable-baselines3训练DQN模型
def train_dqn(env, timesteps=10000, learning_rate=0.0001, buffer_size=10000, batch_size=64):
    """
    使用DQN算法训练模型
    """
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=1000,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1
    )
    
    model.learn(total_timesteps=timesteps)
    return model

# 使用stable-baselines3训练PPO模型
def train_ppo(env, timesteps=10000, learning_rate=0.0003, n_steps=128, batch_size=64):
    """
    使用PPO算法训练模型
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        verbose=1
    )
    
    model.learn(total_timesteps=timesteps)
    return model

# 使用stable-baselines3训练TD3模型
def train_td3(env, timesteps=10000, learning_rate=0.0003, buffer_size=10000, batch_size=100):
    """
    使用TD3算法训练模型，适用于连续动作空间
    注意：TD3需要连续动作空间，与我们的离散动作环境不兼容
    此处仅作为示例，实际调用时可能需要修改环境
    """
    # 创建一个连续动作空间的环境包装器
    class ContinuousEnvWrapper(gym.Wrapper):
        def __init__(self, env):
            super(ContinuousEnvWrapper, self).__init__(env)
            # 将离散动作空间转换为连续动作空间
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
        def step(self, action):
            # 将连续动作映射到离散动作
            discrete_action = 0 if action[0] < 0 else 1
            return self.env.step(discrete_action)
    
    # 包装环境
    continuous_env = ContinuousEnvWrapper(env)
    
    model = TD3(
        "MlpPolicy",
        continuous_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=100,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        action_noise=None,
        policy_kwargs=dict(net_arch=[400, 300]),
        verbose=1
    )
    
    model.learn(total_timesteps=timesteps)
    return model

# 使用stable-baselines3训练SAC模型(作为Actor-Critic的替代)
def train_sac(env, timesteps=10000, learning_rate=0.0003, buffer_size=10000, batch_size=64):
    """
    使用SAC算法训练模型，这是一种更高级的Actor-Critic变体
    注意：SAC需要连续动作空间，与我们的离散动作环境不兼容
    此处仅作为示例，实际调用时可能需要修改环境
    """
    # 创建一个连续动作空间的环境包装器
    class ContinuousEnvWrapper(gym.Wrapper):
        def __init__(self, env):
            super(ContinuousEnvWrapper, self).__init__(env)
            # 将离散动作空间转换为连续动作空间
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
        def step(self, action):
            # 将连续动作映射到离散动作
            discrete_action = 0 if action[0] < 0 else 1
            return self.env.step(discrete_action)
    
    # 包装环境
    continuous_env = ContinuousEnvWrapper(env)
    
    model = SAC(
        "MlpPolicy",
        continuous_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=100,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_update_interval=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        verbose=1
    )
    
    model.learn(total_timesteps=timesteps)
    return model

# 评估强化学习模型
def evaluate_rl_model(model, env, n_eval_episodes=10):
    """
    评估训练好的强化学习模型
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward
    }

# 保存训练结果和图表
def save_results(model_name, results, base_dir="results"):
    """
    保存训练结果和图表
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(f"{base_dir}/{model_name}_results.csv", index=False)
    
    # 绘制奖励曲线
    if 'rewards' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['rewards'])
        plt.title(f"{model_name} 训练奖励")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.savefig(f"{base_dir}/{model_name}_reward_curve.png")
        plt.close()

# 运行多次训练和评估
def run_multiple_times_rl(env_factory, X, y_call, y_put, model_types=None, 
                          runs=None, timesteps=None, max_train_time=None):
    """
    运行多次训练和评估，比较不同强化学习算法的性能
    
    参数:
        env_factory: 环境工厂函数
        X, y_call, y_put: 训练数据
        model_types: 要评估的模型类型列表，如果为None则使用配置文件中enabled=True的算法
        runs: 每个模型运行次数，如果为None则使用配置文件设置
        timesteps: 训练步数限制，如果为None则使用配置文件设置
        max_train_time: 最大训练时间限制(秒)，如果为None则使用配置文件设置
        
    返回:
        results: 所有模型的评估结果
    """
    # 从配置文件加载默认参数
    config = RL_COMPARE_CONFIG
    
    # 如果参数为None，则使用配置文件中的设置
    if runs is None:
        runs = config["training"]["runs"]
    if timesteps is None:
        timesteps = config["training"]["timesteps"]
    if max_train_time is None:
        max_train_time = config["training"]["max_train_time"]
    
    # 如果model_types为None，则使用配置中enabled=True的算法
    if model_types is None:
        model_types = [algo_type for algo_type in ["dqn", "ppo", "td3", "sac", "actor_critic"] 
                      if algo_type in config and config[algo_type]["enabled"]]
    
    results = {}
    
    # 算法信息和参考文献
    algorithm_info = config["algorithm_info"]
    
    print("=== 强化学习算法比较 ===")
    print(f"统一评估条件: {timesteps} 环境步数")
    if max_train_time:
        print(f"最大训练时间限制: {max_train_time} 秒")
    print(f"每个算法运行 {runs} 次\n")
    
    for model_type in model_types:
        print(f"开始训练 {model_type.upper()} 模型...")
        print(f"算法: {algorithm_info[model_type]['full_name']}")
        print(f"参考: {algorithm_info[model_type]['reference']}")
        print(f"实现库: {algorithm_info[model_type]['library']}")
        
        model_results = {
            'time': [],
            'reward': [],
            'std_reward': [],
            'steps_completed': []
        }
        
        for run in range(runs):
            print(f"\n运行 {run+1}/{runs}...")
            # 创建环境
            if model_type in ['td3', 'sac']:
                # 对于需要连续动作空间的算法，创建包装环境
                env = SimpleEnvForRL(X, y_call, y_put)
                
                # 包装环境
                class ContinuousEnvWrapper(gym.Wrapper):
                    def __init__(self, env):
                        super(ContinuousEnvWrapper, self).__init__(env)
                        # 将离散动作空间转换为连续动作空间
                        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        
                    def step(self, action):
                        # 将连续动作映射到离散动作
                        discrete_action = 0 if action[0] < 0 else 1
                        return self.env.step(discrete_action)
                
                env = ContinuousEnvWrapper(env)
                env = Monitor(env)  # 监控环境以记录奖励
            else:
                env = SimpleEnvForRL(X, y_call, y_put)
                env = Monitor(env)  # 监控环境以记录奖励
            
        start_t = time.time()
            steps_completed = 0
            
            # 训练模型，并支持时间限制
            try:
                if model_type == 'dqn':
                    # 从配置加载DQN参数
                    dqn_config = config["dqn"]
                    model = DQN(
                        "MlpPolicy",
                        env,
                        learning_rate=dqn_config["learning_rate"],
                        buffer_size=dqn_config["buffer_size"],
                        batch_size=dqn_config["batch_size"],
                        learning_starts=dqn_config["learning_starts"],
                        target_update_interval=dqn_config["target_update_interval"],
                        exploration_fraction=dqn_config["exploration_fraction"],
                        exploration_final_eps=dqn_config["exploration_final_eps"],
                        gamma=dqn_config["gamma"],
                        policy_kwargs=dict(net_arch=dqn_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 支持时间限制训练
                    if max_train_time:
                        # 每次训练少量步数，检查时间限制
                        remaining_steps = timesteps
                        while remaining_steps > 0 and (time.time() - start_t) < max_train_time:
                            steps_to_train = min(1000, remaining_steps)
                            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
                            steps_completed += steps_to_train
                            remaining_steps -= steps_to_train
                    else:
                        # 标准步数限制训练
                        model.learn(total_timesteps=timesteps)
                        steps_completed = timesteps
                
                elif model_type == 'ppo':
                    # 从配置加载PPO参数
                    ppo_config = config["ppo"]
                    model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=ppo_config["learning_rate"],
                        n_steps=ppo_config["n_steps"],
                        batch_size=ppo_config["batch_size"],
                        n_epochs=ppo_config["n_epochs"],
                        gamma=ppo_config["gamma"],
                        gae_lambda=ppo_config["gae_lambda"],
                        clip_range=ppo_config["clip_range"],
                        clip_range_vf=ppo_config["clip_range_vf"],
                        ent_coef=ppo_config["ent_coef"],
                        vf_coef=ppo_config["vf_coef"],
                        max_grad_norm=ppo_config["max_grad_norm"],
                        policy_kwargs=dict(net_arch=ppo_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 支持时间限制训练
                    if max_train_time:
                        remaining_steps = timesteps
                        while remaining_steps > 0 and (time.time() - start_t) < max_train_time:
                            steps_to_train = min(1000, remaining_steps)
                            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
                            steps_completed += steps_to_train
                            remaining_steps -= steps_to_train
                    else:
                        model.learn(total_timesteps=timesteps)
                        steps_completed = timesteps
                
                elif model_type == 'td3':
                    # 从配置加载TD3参数
                    td3_config = config["td3"]
                    model = TD3(
                        "MlpPolicy",
                        env,
                        learning_rate=td3_config["learning_rate"],
                        buffer_size=td3_config["buffer_size"],
                        batch_size=td3_config["batch_size"],
                        learning_starts=td3_config["learning_starts"],
                        gamma=td3_config["gamma"],
                        tau=td3_config["tau"],
                        policy_delay=td3_config["policy_delay"],
                        policy_kwargs=dict(net_arch=td3_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 支持时间限制训练
                    if max_train_time:
                        remaining_steps = timesteps
                        while remaining_steps > 0 and (time.time() - start_t) < max_train_time:
                            steps_to_train = min(1000, remaining_steps)
                            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
                            steps_completed += steps_to_train
                            remaining_steps -= steps_to_train
                    else:
                        model.learn(total_timesteps=timesteps)
                        steps_completed = timesteps
                
                elif model_type == 'sac':
                    # 从配置加载SAC参数
                    sac_config = config["sac"]
                    model = SAC(
                        "MlpPolicy",
                        env,
                        learning_rate=sac_config["learning_rate"],
                        buffer_size=sac_config["buffer_size"],
                        batch_size=sac_config["batch_size"],
                        learning_starts=sac_config["learning_starts"],
                        gamma=sac_config["gamma"],
                        tau=sac_config["tau"],
                        ent_coef=sac_config["ent_coef"],
                        target_update_interval=sac_config["target_update_interval"],
                        policy_kwargs=dict(net_arch=sac_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 支持时间限制训练
                    if max_train_time:
                        remaining_steps = timesteps
                        while remaining_steps > 0 and (time.time() - start_t) < max_train_time:
                            steps_to_train = min(1000, remaining_steps)
                            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
                            steps_completed += steps_to_train
                            remaining_steps -= steps_to_train
                    else:
                        model.learn(total_timesteps=timesteps)
                        steps_completed = timesteps
                
                elif model_type == 'actor_critic':
                    # 从配置加载Actor-Critic参数
                    ac_config = config["actor_critic"]
                    
                    # 自定义Actor-Critic实现
                    env.reset()
                    
                    # 对于自定义实现，我们使用环境步数而不是episode数量作为度量
                    # 假设每个episode平均10步，调整为与其他算法相同的步数
                    total_steps = 0
                    total_episodes = 0
                    rewards = []
                    
                    # 初始化agent
                    input_dim = env.observation_space.shape[0]
                    n_actions = env.action_space.n
                    agent = ActorCriticAgent(
                        input_dim=input_dim, 
                        n_actions=n_actions, 
                        hidden_dim=ac_config["hidden_dim"], 
                        lr=ac_config["learning_rate"], 
                        gamma=ac_config["gamma"],
                        entropy_weight=ac_config["entropy_weight"]
                    )
                    
                    # 训练循环
                    while total_steps < timesteps:
                        if max_train_time and (time.time() - start_t) >= max_train_time:
                            break
                            
                        state, _ = env.reset()
                        episode_reward = 0
                        done = False
                        episode_steps = 0
                        
                        while not done:
                            # 选择动作
                            action = agent.select_action(state)
                            
                            # 执行动作
                            next_state, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            
                            # 存储奖励
                            agent.rewards.append(reward)
                            episode_reward += reward
                            
                            # 更新状态
                            state = next_state
                            
                            # 更新步数计数
                            episode_steps += 1
                            total_steps += 1
                            
                            # 如果达到步数限制，提前退出
                            if total_steps >= timesteps:
                                break
                                
                            # 如果达到时间限制，提前退出
                            if max_train_time and (time.time() - start_t) >= max_train_time:
                                break
                        
                        # 如果回合结束，更新策略
                        if agent.rewards:
                            agent.update()
                        
                        # 记录回合奖励
                        rewards.append(episode_reward)
                        total_episodes += 1
                        
                        # 定期打印进度
                        if total_episodes % 10 == 0:
                            avg_reward = np.mean(rewards[-10:])
                            print(f"Episode {total_episodes}, 步数: {total_steps}/{timesteps}, 平均奖励: {avg_reward:.2f}")
                    
                    model = None  # 自定义实现没有stable-baselines3兼容的模型
                    steps_completed = total_steps
            
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                # 记录已完成的步数
                model_results['steps_completed'].append(steps_completed)
                # 跳过当前运行并继续下一个
                continue
            
        elapsed = time.time() - start_t
            model_results['time'].append(elapsed)
            model_results['steps_completed'].append(steps_completed)
            
            # 评估模型性能
            if model_type == 'actor_critic':
                # 自定义实现的评估
                final_rewards = rewards[-min(100, len(rewards)):]
                mean_reward = np.mean(final_rewards)
                std_reward = np.std(final_rewards)
                
                # 保存额外指标
                model_results['avg_q'] = 0.0  # 自定义实现没有Q值
                model_results['avg_entropy'] = 0.0  # 熵值，如果有记录可以添加
            else:
                # stable-baselines3模型的评估
                try:
                    eval_result = evaluate_rl_model(model, env)
                    mean_reward = eval_result['mean_reward']
                    std_reward = eval_result['std_reward']
                    
                    # 保存额外指标
                    if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
                        # 一些可能的额外指标，取决于算法
                        if 'train/q_values_mean' in model.logger.name_to_value:
                            model_results['avg_q'] = model.logger.name_to_value['train/q_values_mean']
                        if 'train/entropy_loss' in model.logger.name_to_value:
                            model_results['avg_entropy'] = model.logger.name_to_value['train/entropy_loss']
                except Exception as e:
                    print(f"评估过程中发生错误: {e}")
                    mean_reward = 0
                    std_reward = 0
            
            model_results['reward'].append(mean_reward)
            model_results['std_reward'].append(std_reward)
            
            print(f"运行 {run+1} 完成:")
            print(f"  - 训练时间: {elapsed:.2f}秒")
            print(f"  - 完成步数: {steps_completed}/{timesteps}")
            print(f"  - 平均奖励: {mean_reward:.2f}±{std_reward:.2f}")
        
        # 计算模型的平均性能
        if model_results['time']:  # 确保列表不为空
            model_summary = {
                'time_mean': np.mean(model_results['time']),
                'time_std': np.std(model_results['time']),
                'reward_mean': np.mean(model_results['reward']),
                'reward_std': np.mean(model_results['std_reward']),
                'steps_completed_mean': np.mean(model_results['steps_completed']),
                'steps_completed_percent': np.mean(model_results['steps_completed']) / timesteps * 100
            }
            
            # 添加可能存在的额外指标
            if 'avg_q' in model_results:
                model_summary['avg_q'] = model_results['avg_q']
            if 'avg_entropy' in model_results:
                model_summary['avg_entropy'] = model_results['avg_entropy']
            
            results[model_type] = {
                'runs': model_results,
                'summary': model_summary,
                'info': algorithm_info[model_type]
            }
            
            print(f"\n{model_type.upper()} 模型摘要:")
            print(f"  - 平均训练时间: {model_summary['time_mean']:.2f}±{model_summary['time_std']:.2f}秒")
            print(f"  - 平均完成步数: {model_summary['steps_completed_mean']:.0f} ({model_summary['steps_completed_percent']:.1f}%)")
            print(f"  - 平均奖励: {model_summary['reward_mean']:.2f}±{model_summary['reward_std']:.2f}")
            print("-" * 50)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 从配置文件获取环境设置
    n_samples = RL_COMPARE_CONFIG["env"]["n_samples"]
    n_features = RL_COMPARE_CONFIG["env"]["n_features"]
    
    # 创建示例数据
    X = np.random.randn(n_samples, n_features)
    y_call = X[:, 0] + 0.1 * np.random.randn(n_samples)
    y_put = X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # 从配置文件获取训练参数
    TIMESTEPS = RL_COMPARE_CONFIG["training"]["timesteps"]
    MAX_TRAIN_TIME = RL_COMPARE_CONFIG["training"]["max_train_time"]
    RUNS = RL_COMPARE_CONFIG["training"]["runs"]
    
    # 从配置文件获取要启用的模型类型
    enabled_models = [model for model in ["dqn", "ppo", "td3", "sac", "actor_critic"] 
                     if model in RL_COMPARE_CONFIG and RL_COMPARE_CONFIG[model]["enabled"]]
    
    print("=== 强化学习算法比较 (统一评估条件) ===")
    print(f"环境步数限制: {TIMESTEPS}")
    print(f"时间限制: {MAX_TRAIN_TIME} 秒")
    print(f"每个算法运行 {RUNS} 次")
    print(f"启用的算法: {', '.join(enabled_models).upper()}")
    
    # 创建结果输出目录
    results_dir = RL_COMPARE_CONFIG["results"]["dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行多个强化学习模型进行比较
    results = run_multiple_times_rl(
        SimpleEnvForRL, X, y_call, y_put,
        model_types=enabled_models,
        runs=RUNS,
        timesteps=TIMESTEPS,
        max_train_time=MAX_TRAIN_TIME
    )
    
    # 保存结果为CSV
    if RL_COMPARE_CONFIG["results"]["save_csv"]:
        results_df = []
        for model_type, model_data in results.items():
            summary = model_data['summary']
            info = model_data.get('info', {})
            row = {
                'model_type': model_type,
                'full_name': info.get('full_name', model_type),
                'reference': info.get('reference', ''),
                'library': info.get('library', ''),
                'time_mean': summary['time_mean'],
                'time_std': summary['time_std'],
                'reward_mean': summary['reward_mean'],
                'reward_std': summary['reward_std'],
                'steps_completed_mean': summary.get('steps_completed_mean', TIMESTEPS),
                'steps_completed_percent': summary.get('steps_completed_percent', 100.0)
            }
            # 添加可能存在的额外指标
            if 'avg_q' in summary:
                row['avg_q'] = summary['avg_q']
            if 'avg_entropy' in summary:
                row['avg_entropy'] = summary['avg_entropy']
            
            results_df.append(row)
        
        # 创建DataFrame并保存
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(f"{results_dir}/rl_algorithms_comparison.csv", index=False)
        print(f"\n结果已保存到 {results_dir}/rl_algorithms_comparison.csv")
    
    # 打印更详细的比较结果
    print("\n=== 各模型性能比较 ===")
    print(f"{'算法':<15} {'训练时间(秒)':<20} {'步数完成率':<15} {'平均奖励':<20} {'参考文献'}")
    print("-" * 100)
    for model_type, model_data in results.items():
        summary = model_data['summary']
        info = model_data.get('info', {})
        print(f"{info.get('full_name', model_type):<15} "
              f"{summary['time_mean']:.2f}±{summary['time_std']:.2f} 秒  "
              f"{summary.get('steps_completed_percent', 100.0):.1f}%        "
              f"{summary['reward_mean']:.4f}±{summary['reward_std']:.4f}    "
              f"{info.get('reference', '')[:60]}")
    
    # 可视化比较结果
    if RL_COMPARE_CONFIG["results"]["save_plots"]:
        try:
            # 设置样式
            vis_config = RL_COMPARE_CONFIG["visualization"]
            sns.set(style=vis_config["style"])
            
            # 创建比较图表
            fig, axs = plt.subplots(2, 2, figsize=vis_config["figsize"])
            
            # 提取模型名称和指标
            model_names = [model_data['info'].get('full_name', model_type) for model_type, model_data in results.items()]
            training_times = [model_data['summary']['time_mean'] for _, model_data in results.items()]
            time_stds = [model_data['summary']['time_std'] for _, model_data in results.items()]
            rewards = [model_data['summary']['reward_mean'] for _, model_data in results.items()]
            reward_stds = [model_data['summary']['reward_std'] for _, model_data in results.items()]
            steps_percent = [model_data['summary'].get('steps_completed_percent', 100.0) for _, model_data in results.items()]
            
            # 训练时间比较
            axs[0, 0].bar(model_names, training_times, yerr=time_stds, capsize=5, color=vis_config["colors"]["time"])
            axs[0, 0].set_title('训练时间比较')
            axs[0, 0].set_ylabel('时间 (秒)')
            axs[0, 0].tick_params(axis='x', rotation=45)
            
            # 平均奖励比较
            axs[0, 1].bar(model_names, rewards, yerr=reward_stds, capsize=5, color=vis_config["colors"]["reward"])
            axs[0, 1].set_title('平均奖励比较')
            axs[0, 1].set_ylabel('奖励')
            axs[0, 1].tick_params(axis='x', rotation=45)
            
            # 步数完成率比较
            axs[1, 0].bar(model_names, steps_percent, color=vis_config["colors"]["steps"])
            axs[1, 0].set_title('训练步数完成率')
            axs[1, 0].set_ylabel('完成率 (%)')
            axs[1, 0].set_ylim(0, 100)
            axs[1, 0].tick_params(axis='x', rotation=45)
            
            # 时间效率比较（奖励/时间）
            efficiency = [r/t if t > 0 else 0 for r, t in zip(rewards, training_times)]
            axs[1, 1].bar(model_names, efficiency, color=vis_config["colors"]["efficiency"])
            axs[1, 1].set_title('时间效率 (奖励/时间)')
            axs[1, 1].set_ylabel('效率')
            axs[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/rl_algorithms_comparison.{vis_config['save_format']}", dpi=vis_config["dpi"], bbox_inches='tight')
            print(f"比较图表已保存到 {results_dir}/rl_algorithms_comparison.{vis_config['save_format']}")
        except Exception as e:
            print(f"创建可视化图表时出错: {e}")

# 添加RL迁移学习功能
def transfer_rl_model(pretrained_model, env_factory, X, y_call, y_put, 
                     model_type, transfer_params=None, timesteps=None, 
                     verbose=True):
    """
    将预训练的RL模型迁移到新环境
    
    参数:
        pretrained_model: 预训练的RL模型
        env_factory: 环境工厂函数
        X, y_call, y_put: 新环境的数据
        model_type: 模型类型，如'dqn', 'ppo'等
        transfer_params: 迁移学习参数
        timesteps: 迁移学习的时间步数
        verbose: 是否显示详细输出
        
    返回:
        transferred_model: 迁移学习后的模型
        metrics: 性能指标
    """
    if transfer_params is None:
        transfer_params = config.RL_COMPARE_CONFIG["transfer"]
    
    # 如果没有指定时间步数，使用配置中的值
    if timesteps is None:
        timesteps = transfer_params["finetune_timesteps"]
    
    # 创建新环境
    env = SimpleEnvForRL(X, y_call, y_put)
    env = Monitor(env)  # 监控环境以记录奖励
    
    # 调整学习率
    if model_type == 'dqn':
        finetune_lr = transfer_params["finetune_lr"]
        if hasattr(pretrained_model, 'learning_rate'):
            pretrained_model.learning_rate = finetune_lr
        
        # 调整探索率
        if hasattr(pretrained_model, 'exploration_schedule'):
            pretrained_model.exploration_schedule.initial_p = transfer_params["exploration_fraction"]
    
    elif model_type == 'ppo':
        finetune_lr = transfer_params["finetune_lr"]
        if hasattr(pretrained_model, 'learning_rate'):
            pretrained_model.learning_rate = finetune_lr
    
    elif model_type == 'actor_critic':
        # 对于自定义的Actor-Critic实现
        finetune_lr = transfer_params["finetune_lr"]
        if hasattr(pretrained_model, 'optimizer') and hasattr(pretrained_model.optimizer, 'param_groups'):
            for param_group in pretrained_model.optimizer.param_groups:
                param_group['lr'] = finetune_lr
    
    # 迁移学习
    print(f"开始{model_type.upper()}模型的迁移学习，时间步数: {timesteps}")
    start_time = time.time()
    
    try:
        # 不同模型类型的迁移学习方法
        if model_type in ['dqn', 'ppo', 'td3', 'sac']:
            # 在新环境上继续训练
            pretrained_model.set_env(env)
            pretrained_model.learn(total_timesteps=timesteps)
        
        elif model_type == 'actor_critic':
            # 自定义Actor-Critic迁移学习
            env.reset()
            total_steps = 0
            total_episodes = 0
            rewards = []
            
            while total_steps < timesteps:
                state, _ = env.reset()
                episode_reward = 0
                done = False
                episode_steps = 0
                
                while not done:
                    # 选择动作
                    action = pretrained_model.select_action(state)
                    
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # 存储奖励
                    pretrained_model.rewards.append(reward)
                    episode_reward += reward
                    
                    # 更新状态
                    state = next_state
                    
                    # 更新步数计数
                    episode_steps += 1
                    total_steps += 1
                    
                    # 如果达到步数限制，提前退出
                    if total_steps >= timesteps:
                        break
                
                # 如果回合结束，更新策略
                if pretrained_model.rewards:
                    pretrained_model.update()
                
                # 记录回合奖励
                rewards.append(episode_reward)
                total_episodes += 1
            
    except Exception as e:
        print(f"迁移学习过程中发生错误: {e}")
    
    # 计算训练时间
    training_time = time.time() - start_time
    
    # 评估迁移后的模型
    mean_reward, std_reward = evaluate_rl_model(pretrained_model, env, n_eval_episodes=10)
    
    metrics = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'timesteps': timesteps
    }
    
    print(f"迁移学习完成，耗时: {training_time:.2f}秒，平均奖励: {mean_reward:.4f}±{std_reward:.4f}")
    
    return pretrained_model, metrics


def run_rl_transfer_learning(source_data, target_data, model_types=None, 
                           runs=3, pretrain_timesteps=None, finetune_timesteps=None, 
                           max_train_time=None, verbose=True):
    """
    运行RL迁移学习实验
    
    参数:
        source_data: 源域数据字典，包含'X_train', 'y_train'等键
        target_data: 目标域数据字典，包含'X_train', 'y_train'等键
        model_types: 要使用的模型类型列表
        runs: 每个模型运行次数
        pretrain_timesteps: 预训练时间步数
        finetune_timesteps: 微调时间步数
        max_train_time: 最大训练时间(秒)
        verbose: 是否显示详细输出
        
    返回:
        results: 实验结果字典
    """
    # 从配置中加载默认参数
    config_data = config.RL_COMPARE_CONFIG
    
    # 如果未指定模型类型，则使用配置中启用的模型
    if model_types is None:
        model_types = [m for m in ["dqn", "ppo", "actor_critic"] 
                     if config_data[m]["enabled"]]
    
    # 设置时间步数
    if pretrain_timesteps is None:
        pretrain_timesteps = config_data["training"]["timesteps"]
    if finetune_timesteps is None:
        finetune_timesteps = config_data["transfer"]["finetune_timesteps"]
    if max_train_time is None:
        max_train_time = config_data["training"]["max_train_time"]
    
    # 提取数据
    X_source = source_data["X_train"]
    y_call_source = source_data["y_train"][:, 0]
    y_put_source = source_data["y_train"][:, 1]
    
    X_target = target_data["X_train"]
    y_call_target = target_data["y_train"][:, 0]
    y_put_target = target_data["y_train"][:, 1]
    
    # 创建结果字典
    results = {}
    
    # 算法信息
    algorithm_info = config_data["algorithm_info"]
    
    print("=== 强化学习迁移学习实验 ===")
    print(f"预训练步数: {pretrain_timesteps}")
    print(f"微调步数: {finetune_timesteps}")
    if max_train_time:
        print(f"最大训练时间限制: {max_train_time}秒")
    print(f"每个算法运行{runs}次\n")
    
    for model_type in model_types:
        print(f"\n开始训练和迁移 {model_type.upper()} 模型...")
        print(f"算法: {algorithm_info[model_type]['full_name']}")
        print(f"参考: {algorithm_info[model_type]['reference']}")
        
        model_results = {
            'pretrain': {'reward': [], 'std_reward': [], 'time': []},
            'finetune': {'reward': [], 'std_reward': [], 'time': []},
            'direct': {'reward': [], 'std_reward': [], 'time': []}
        }
        
        for run in range(runs):
            print(f"\n运行 {run+1}/{runs}...")
            
            # 1. 在源域上预训练模型
            print("预训练阶段...")
            env_source = SimpleEnvForRL(X_source, y_call_source, y_put_source)
            env_source = Monitor(env_source)
            
            start_time = time.time()
            pretrained_model = None
            
            try:
                if model_type == 'dqn':
                    # 从配置加载DQN参数
                    dqn_config = config_data["dqn"]
                    pretrained_model = DQN(
                        "MlpPolicy",
                        env_source,
                        learning_rate=dqn_config["learning_rate"],
                        buffer_size=dqn_config["buffer_size"],
                        batch_size=dqn_config["batch_size"],
                        learning_starts=dqn_config["learning_starts"],
                        target_update_interval=dqn_config["target_update_interval"],
                        exploration_fraction=dqn_config["exploration_fraction"],
                        exploration_final_eps=dqn_config["exploration_final_eps"],
                        gamma=dqn_config["gamma"],
                        policy_kwargs=dict(net_arch=dqn_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 训练模型
                    pretrained_model.learn(total_timesteps=pretrain_timesteps)
                
                elif model_type == 'ppo':
                    # 从配置加载PPO参数
                    ppo_config = config_data["ppo"]
                    pretrained_model = PPO(
                        "MlpPolicy",
                        env_source,
                        learning_rate=ppo_config["learning_rate"],
                        n_steps=ppo_config["n_steps"],
                        batch_size=ppo_config["batch_size"],
                        n_epochs=ppo_config["n_epochs"],
                        gamma=ppo_config["gamma"],
                        gae_lambda=ppo_config["gae_lambda"],
                        clip_range=ppo_config["clip_range"],
                        clip_range_vf=ppo_config["clip_range_vf"],
                        ent_coef=ppo_config["ent_coef"],
                        vf_coef=ppo_config["vf_coef"],
                        max_grad_norm=ppo_config["max_grad_norm"],
                        policy_kwargs=dict(net_arch=ppo_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 训练模型
                    pretrained_model.learn(total_timesteps=pretrain_timesteps)
                
                elif model_type == 'actor_critic':
                    # 自定义Actor-Critic实现
                    ac_config = config_data["actor_critic"]
                    
                    # 初始化agent
                    input_dim = env_source.observation_space.shape[0]
                    n_actions = env_source.action_space.n
                    pretrained_model = ActorCriticAgent(
                        input_dim=input_dim, 
                        n_actions=n_actions, 
                        hidden_dim=ac_config["hidden_dim"], 
                        lr=ac_config["learning_rate"], 
                        gamma=ac_config["gamma"],
                        entropy_weight=ac_config["entropy_weight"]
                    )
                    
                    # 训练循环
                    total_steps = 0
                    total_episodes = 0
                    rewards = []
                    
                    while total_steps < pretrain_timesteps:
                        if max_train_time and (time.time() - start_time) >= max_train_time:
                            break
                            
                        state, _ = env_source.reset()
                        episode_reward = 0
                        done = False
                        
                        while not done:
                            # 选择动作
                            action = pretrained_model.select_action(state)
                            
                            # 执行动作
                            next_state, reward, terminated, truncated, _ = env_source.step(action)
                            done = terminated or truncated
                            
                            # 存储奖励
                            pretrained_model.rewards.append(reward)
                            episode_reward += reward
                            
                            # 更新状态
                            state = next_state
                            
                            # 更新步数计数
                            total_steps += 1
                            
                            # 如果达到步数限制，提前退出
                            if total_steps >= pretrain_timesteps:
                                break
                        
                        # 如果回合结束，更新策略
                        if pretrained_model.rewards:
                            pretrained_model.update()
                        
                        # 记录回合奖励
                        rewards.append(episode_reward)
                        total_episodes += 1
            
            except Exception as e:
                print(f"预训练过程中发生错误: {e}")
                continue
            
            pretrain_time = time.time() - start_time
            
            # 评估预训练模型
            mean_reward, std_reward = evaluate_rl_model(pretrained_model, env_source, n_eval_episodes=10)
            
            model_results['pretrain']['reward'].append(mean_reward)
            model_results['pretrain']['std_reward'].append(std_reward)
            model_results['pretrain']['time'].append(pretrain_time)
            
            print(f"预训练完成，耗时: {pretrain_time:.2f}秒")
            print(f"源域评估: 平均奖励 = {mean_reward:.4f} ± {std_reward:.4f}")
            
            # 2. 迁移学习到目标域
            print("\n迁移学习阶段...")
            
            # 创建目标域环境
            env_target = SimpleEnvForRL(X_target, y_call_target, y_put_target)
            env_target = Monitor(env_target)
            
            # 进行迁移学习
            transferred_model, transfer_metrics = transfer_rl_model(
                pretrained_model,
                None,  # 不需要env_factory
                X_target, y_call_target, y_put_target,
                model_type,
                config_data["transfer"],
                finetune_timesteps,
                verbose
            )
            
            model_results['finetune']['reward'].append(transfer_metrics['mean_reward'])
            model_results['finetune']['std_reward'].append(transfer_metrics['std_reward'])
            model_results['finetune']['time'].append(transfer_metrics['training_time'])
            
            # 3. 直接在目标域上训练模型作为基线
            print("\n直接训练阶段...")
            
            start_time = time.time()
            direct_model = None
            
            try:
                if model_type == 'dqn':
                    # 从配置加载DQN参数
                    dqn_config = config_data["dqn"]
                    direct_model = DQN(
                        "MlpPolicy",
                        env_target,
                        learning_rate=dqn_config["learning_rate"],
                        buffer_size=dqn_config["buffer_size"],
                        batch_size=dqn_config["batch_size"],
                        learning_starts=dqn_config["learning_starts"],
                        target_update_interval=dqn_config["target_update_interval"],
                        exploration_fraction=dqn_config["exploration_fraction"],
                        exploration_final_eps=dqn_config["exploration_final_eps"],
                        gamma=dqn_config["gamma"],
                        policy_kwargs=dict(net_arch=dqn_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 训练模型
                    direct_model.learn(total_timesteps=finetune_timesteps)
                
                elif model_type == 'ppo':
                    # 从配置加载PPO参数
                    ppo_config = config_data["ppo"]
                    direct_model = PPO(
                        "MlpPolicy",
                        env_target,
                        learning_rate=ppo_config["learning_rate"],
                        n_steps=ppo_config["n_steps"],
                        batch_size=ppo_config["batch_size"],
                        n_epochs=ppo_config["n_epochs"],
                        gamma=ppo_config["gamma"],
                        gae_lambda=ppo_config["gae_lambda"],
                        clip_range=ppo_config["clip_range"],
                        clip_range_vf=ppo_config["clip_range_vf"],
                        ent_coef=ppo_config["ent_coef"],
                        vf_coef=ppo_config["vf_coef"],
                        max_grad_norm=ppo_config["max_grad_norm"],
                        policy_kwargs=dict(net_arch=ppo_config["net_arch"]),
                        verbose=0
                    )
                    
                    # 训练模型
                    direct_model.learn(total_timesteps=finetune_timesteps)
                
                elif model_type == 'actor_critic':
                    # 自定义Actor-Critic实现
                    ac_config = config_data["actor_critic"]
                    
                    # 初始化agent
                    input_dim = env_target.observation_space.shape[0]
                    n_actions = env_target.action_space.n
                    direct_model = ActorCriticAgent(
                        input_dim=input_dim, 
                        n_actions=n_actions, 
                        hidden_dim=ac_config["hidden_dim"], 
                        lr=ac_config["learning_rate"], 
                        gamma=ac_config["gamma"],
                        entropy_weight=ac_config["entropy_weight"]
                    )
                    
                    # 训练循环
                    total_steps = 0
                    total_episodes = 0
                    rewards = []
                    
                    while total_steps < finetune_timesteps:
                        if max_train_time and (time.time() - start_time) >= max_train_time:
                            break
                            
                        state, _ = env_target.reset()
                        episode_reward = 0
                        done = False
                        
                        while not done:
                            # 选择动作
                            action = direct_model.select_action(state)
                            
                            # 执行动作
                            next_state, reward, terminated, truncated, _ = env_target.step(action)
                            done = terminated or truncated
                            
                            # 存储奖励
                            direct_model.rewards.append(reward)
                            episode_reward += reward
                            
                            # 更新状态
                            state = next_state
                            
                            # 更新步数计数
                            total_steps += 1
                            
                            # 如果达到步数限制，提前退出
                            if total_steps >= finetune_timesteps:
                                break
                        
                        # 如果回合结束，更新策略
                        if direct_model.rewards:
                            direct_model.update()
                        
                        # 记录回合奖励
                        rewards.append(episode_reward)
                        total_episodes += 1
            
            except Exception as e:
                print(f"直接训练过程中发生错误: {e}")
                continue
            
            direct_time = time.time() - start_time
            
            # 评估直接训练模型
            mean_reward, std_reward = evaluate_rl_model(direct_model, env_target, n_eval_episodes=10)
            
            model_results['direct']['reward'].append(mean_reward)
            model_results['direct']['std_reward'].append(std_reward)
            model_results['direct']['time'].append(direct_time)
            
            print(f"直接训练完成，耗时: {direct_time:.2f}秒")
            print(f"目标域评估: 平均奖励 = {mean_reward:.4f} ± {std_reward:.4f}")
        
        # 计算平均结果
        for phase in ['pretrain', 'finetune', 'direct']:
            for metric in ['reward', 'std_reward', 'time']:
                if model_results[phase][metric]:
                    model_results[phase][f'{metric}_mean'] = np.mean(model_results[phase][metric])
                    model_results[phase][f'{metric}_std'] = np.std(model_results[phase][metric])
        
        # 计算迁移学习改进
        if 'reward_mean' in model_results['finetune'] and 'reward_mean' in model_results['direct']:
            finetune_reward = model_results['finetune']['reward_mean']
            direct_reward = model_results['direct']['reward_mean']
            
            # 对于奖励，更高更好，所以计算提升百分比
            improvement = (finetune_reward - direct_reward) / abs(direct_reward) * 100
            model_results['improvement'] = {'reward_percent': improvement}
            
            print(f"\n{model_type.upper()} 迁移学习结果:")
            print(f"预训练奖励: {model_results['pretrain']['reward_mean']:.4f} ± {model_results['pretrain']['std_reward_mean']:.4f}")
            print(f"迁移学习奖励: {finetune_reward:.4f} ± {model_results['finetune']['std_reward_mean']:.4f}")
            print(f"直接训练奖励: {direct_reward:.4f} ± {model_results['direct']['std_reward_mean']:.4f}")
            
            if improvement > 0:
                print(f"迁移学习提升了奖励 +{improvement:.2f}%")
            else:
                print(f"迁移学习降低了奖励 {improvement:.2f}%")
        
        # 保存结果
        results[model_type] = model_results
    
    return results


def save_rl_transfer_results(results, output_dir=None):
    """
    保存RL迁移学习结果
    
    参数:
        results: 结果字典
        output_dir: 输出目录
        
    返回:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_DIR, "rl_transfer")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    rows = []
    
    for model_type, model_results in results.items():
        pretrain = model_results.get("pretrain", {})
        finetune = model_results.get("finetune", {})
        direct = model_results.get("direct", {})
        improvement = model_results.get("improvement", {})
        
        row = {
            "model": model_type,
            "pretrain_reward": pretrain.get("reward_mean", float('nan')),
            "pretrain_reward_std": pretrain.get("reward_std", float('nan')),
            "pretrain_time": pretrain.get("time_mean", float('nan')),
            
            "finetune_reward": finetune.get("reward_mean", float('nan')),
            "finetune_reward_std": finetune.get("reward_std", float('nan')),
            "finetune_time": finetune.get("time_mean", float('nan')),
            
            "direct_reward": direct.get("reward_mean", float('nan')),
            "direct_reward_std": direct.get("reward_std", float('nan')),
            "direct_time": direct.get("time_mean", float('nan')),
            
            "reward_improvement_pct": improvement.get("reward_percent", float('nan'))
        }
        
        rows.append(row)
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(rows)
    
    # 按改进百分比排序
    results_df = results_df.sort_values(by="reward_improvement_pct", ascending=False)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"rl_transfer_results_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    results_df.to_csv(file_path, index=False)
    
    print(f"RL迁移学习结果已保存至: {file_path}")
    return file_path
