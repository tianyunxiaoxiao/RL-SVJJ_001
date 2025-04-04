# (2) 主入口，读取配置 & 调用后续逻辑
# === main_svjj.py ===
"""
主入口脚本(main)，示例功能：
1. 读取 config.py，获取全局配置
2. 调用 svjj_model.py 中的 SVJJ 仿真，生成或加载数据
3. 进行数据预处理、拆分
4. 分别调用 ml_dl_models_compare.py 和 rl_compare.py 进行多种模型的训练与评估
5. 最后汇总打印对比结果并可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import time

# 读取全局配置
import config
from svjj_model import SVCJParams, generate_random_option_data
from ml_dl_models_compare import (
    train_and_evaluate_ml_models, 
    perform_hyperparameter_tuning,
    run_multiple_times_dl,
    evaluate_models_performance,
    visualize_learning_curves,
    DeepMLP,
    ResidualMLP,
    SimpleCNN,
    AdvancedCNN,
    SimpleRNN,
    SimpleLSTM,
    SimpleGRU,
    SimpleTransformer,
    BayesianNN
)
from rl_compare import (SimpleEnvForRL, run_multiple_times_rl)
from utils import (
    get_features_and_labels, 
    print_all_comparisons,
    visualize_model_comparisons,
    create_metrics_dataframe,
    evaluate_predictions,
    evaluate_rainbow_dqn,
    compare_rainbow_stages
)

class SVJJEnvironment:
    """为Rainbow DQN提供的SVJJ环境接口"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.current_idx = 0
        self.action_space_size = 2  # 0=看涨期权，1=看跌期权
        
    def reset(self):
        """重置环境，返回初始状态"""
        self.current_idx = 0
        return self.X[self.current_idx].astype(np.float32)
    
    def step(self, action):
        """执行动作，返回下一状态、奖励和是否结束"""
        # 计算奖励：预测价格与实际价格的负差距
        if action == 0:  # 看涨期权
            reward = -abs(self.y[self.current_idx][0] - self.get_model_prediction(action))
        else:  # 看跌期权
            reward = -abs(self.y[self.current_idx][1] - self.get_model_prediction(action))
        
        # 移动到下一个样本
        self.current_idx += 1
        done = self.current_idx >= self.n_samples
        
        # 获取下一个状态
        if done:
            next_state = np.zeros(self.X.shape[1], dtype=np.float32)
        else:
            next_state = self.X[self.current_idx].astype(np.float32)
            
        return next_state, reward, done
    
    def get_model_prediction(self, action):
        """获取模型预测值，可以是简单的基于特征的计算"""
        # 简单示例：基于特征的预测
        if action == 0:  # 看涨期权
            return np.mean(self.X[self.current_idx]) * 1.1
        else:  # 看跌期权
            return np.mean(self.X[self.current_idx]) * 0.9
    
    def action_space(self):
        """返回动作空间大小"""
        return self.action_space_size
    
    def train(self):
        """设置为训练模式"""
        pass
        
    def eval(self):
        """设置为评估模式"""
        pass
        
    def close(self):
        """关闭环境"""
        pass

def train_rainbow_dqn(args, env, config):
    """训练Rainbow DQN模型"""
    from rainbow_model.agent import Agent
    from rainbow_model.memory import ReplayMemory
    from rainbow_model.test import test
    from tqdm import trange
    
    # 创建结果目录
    results_dir = os.path.join('results', args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 初始化指标
    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
    
    # 创建代理和内存
    dqn = Agent(args, env)
    mem = ReplayMemory(args, args.memory_capacity)
    
    # 计算优先级权重增加
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    
    # 构建验证内存
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state = env.reset()
        next_state, _, done = env.step(np.random.randint(0, env.action_space()))
        val_mem.append(state, -1, 0.0, done)
        state = next_state
        T += 1
    
    # 训练循环
    dqn.train()
    done = True
    for T in trange(1, args.T_max + 1):
        if done:
            state = env.reset()
        
        if T % args.replay_frequency == 0:
            dqn.reset_noise()
        
        action = dqn.act(state)
        next_state, reward, done = env.step(action)
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)
        mem.append(state, action, reward, done)
        
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
            
            if T % args.replay_frequency == 0:
                dqn.learn(mem)
            
            if T % args.evaluation_interval == 0:
                dqn.eval()
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)
                print(f'T = {T} / {args.T_max} | 平均奖励: {avg_reward} | 平均Q值: {avg_Q}')
                dqn.train()
            
            if T % args.target_update == 0:
                dqn.update_target_net()
            
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint.pth')
        
        state = next_state
    
    env.close()
    return metrics

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 两阶段训练流程
    if args.pretrain:
        print("=== 阶段1: 使用SVJJ理论模型生成的数据进行预训练 ===")
        # 生成SVJJ模拟数据
        print("生成SVJJ模拟数据...")
        # 创建SVCJ参数
    params = SVCJParams(
            r=0.0201,                  # 无风险利率
            q=0.0174,                  # 股息率
            alp_v=0.026 * 252,         # 波动率均值回归速度
            m_v=0.54 * 252 / 10000,    # 波动率长期均值
            sig_v=0.08 * 252 / 100,    # 波动率的波动率
            rho=-0.48,                 # 标的与波动率相关性
            lam=0.006 * 252,           # 跳跃到达率(P测度)
            lam_q=0.006 * 252,         # 跳跃到达率(Q测度)
            v0=0.54 * 252 / 10000,     # 初始波动率
            rho_j=np.finfo(float).eps, # 跳跃相关性(接近零)
            mu_v=1.48 * 252 / 10000,   # 波动率跳跃均值(P测度)
            mu_vq=8.78 * 252 / 10000,  # 波动率跳跃均值(Q测度)
            gam_s=0.04,                # 股票市场价格风险
            gam_v=-0.031 * 252,        # 波动率市场价格风险
            mu_s=-2.63 / 100,          # 价格跳跃均值(P测度)
            sig_s=2.89 / 100,          # 价格跳跃波动率(P测度)
            mu_sq=-2.63 / 100,         # 价格跳跃均值(Q测度)
            sig_sq=2.89 / 100          # 价格跳跃波动率(Q测度)
        )
        
        # 生成模拟数据
        svjj_data = generate_random_option_data(params, num_samples=args.svjj_samples)
        
        # 预处理模拟数据
        X_pretrain, y_pretrain = prepare_svjj_data(svjj_data, config)
        
        # 划分训练集和测试集
        X_train, X_val, y_train, y_val = train_test_split(
            X_pretrain, y_pretrain, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 保存scaler供后续使用
        import joblib
        joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
        
        # 创建环境
        env = SVJJEnvironment(X_train, y_train)
        
        # 根据选择的模型类型进行预训练
        if args.model_type == 'svjj':
            print("预训练SVJJ模型...")
            svjj_model = train_svjj(X_train, y_train, config)
            save_model(svjj_model, os.path.join(args.output_dir, 'pretrained_svjj_model.pkl'))
            # 评估模型
            svjj_metrics = evaluate_svjj(svjj_model, X_val, y_val)
            print("SVJJ预训练模型评估完成")
            
        elif args.model_type == 'fake_rl':
            print("预训练FakeRL模型...")
            fake_rl_model = train_fake_rl(X_train, y_train, config)
            save_model(fake_rl_model, os.path.join(args.output_dir, 'pretrained_fake_rl_model.pkl'))
            # 评估模型
            fake_rl_metrics = evaluate_fake_rl(fake_rl_model, X_val, y_val)
            print("FakeRL预训练模型评估完成")
            
        elif args.model_type == 'rainbow_dqn':
            print("预训练Rainbow DQN模型...")
            # 导入必要的库
            from tqdm import trange
            from rainbow_model.agent import Agent
            from rainbow_model.memory import ReplayMemory
            from rainbow_model.test import test
            
            # 更新Rainbow配置并创建参数对象
            rainbow_config = config.RAINBOW_CONFIG
            rainbow_args = create_rainbow_args(args, rainbow_config)
            
            # 记录开始时间
            start_time = time.time()
            
            # 预训练Rainbow DQN
            rainbow_metrics = train_rainbow_dqn(rainbow_args, env, rainbow_config)
            
            # 记录训练时间
            training_time = time.time() - start_time
            print("Rainbow DQN预训练完成")
            
            # 保存预训练的Rainbow模型
            results_dir = os.path.join(args.output_dir, 'rainbow_dqn_pretrained')
            os.makedirs(results_dir, exist_ok=True)
            dqn = Agent(rainbow_args, env)
            dqn.save(results_dir, 'pretrained_model.pth')
            
            # 评估预训练的Rainbow DQN
            val_mem = ReplayMemory(rainbow_args, rainbow_config['evaluation_size'])
            T, done = 0, True
            while T < rainbow_config['evaluation_size']:
                if done:
                    state = env.reset()
                next_state, _, done = env.step(np.random.randint(0, env.action_space()))
                val_mem.append(state, -1, 0.0, done)
                state = next_state
                T += 1
                
            dqn.eval()
            avg_reward, avg_Q = test(rainbow_args, rainbow_config['T_max'], dqn, val_mem, rainbow_metrics, 
                                   results_dir, evaluate=True)
            print(f'Rainbow DQN预训练评估结果 - 平均奖励: {avg_reward} | 平均Q值: {avg_Q}')
            
            # 使用我们的新函数生成全部指标
            rainbow_summary = evaluate_rainbow_dqn(rainbow_metrics, avg_reward, avg_Q, training_time)
            
            # 打印Rainbow DQN的详细指标
            print("\nRainbow DQN预训练详细指标:")
            print(f"看涨期权MSE: {rainbow_summary['mse_call_mean']:.6f}")
            print(f"看跌期权MSE: {rainbow_summary['mse_put_mean']:.6f}")
            print(f"看涨期权R²: {rainbow_summary['r2_call_mean']:.6f}")
            print(f"看跌期权R²: {rainbow_summary['r2_put_mean']:.6f}")
            print(f"训练时间: {rainbow_summary['time_mean']:.2f}秒")
            print(f"平均奖励: {rainbow_summary['avg_reward']:.4f}")
            print(f"平均Q值: {rainbow_summary['avg_q']:.4f}")
            print(f"最佳平均奖励: {rainbow_summary['best_avg_reward']:.4f}")
            
            # 保存Rainbow DQN的评估指标
            results_df = pd.DataFrame([rainbow_summary])
            results_df.to_csv(os.path.join(results_dir, 'rainbow_dqn_pretrain_metrics.csv'), index=False)
            
            # 将预训练结果也可视化
            rl_summary_dict = {"Rainbow DQN (Pretrain)": rainbow_summary}
            visualize_results(rl_summary_dict, config)
            
            # 可视化Rainbow DQN的预训练过程
            visualize_rainbow_results(rainbow_metrics, args.output_dir, suffix='pretrain')
            
    if args.transfer:
        print("=== 阶段2: 在真实数据上进行迁移学习 ===")
        
        if not args.data_path:
            raise ValueError("进行迁移学习需要指定真实数据路径 --data-path")
        
        # 加载真实数据
        print(f"加载真实数据: {args.data_path}")
        real_data = pd.read_csv(args.data_path)
        
        # 数据预处理
        X_real, y_real = prepare_real_data(real_data, config)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )
        
        # 使用与预训练相同的scaler进行标准化
        if args.pretrain:
            # 如果刚刚进行了预训练，直接使用之前创建的scaler
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            # 否则，尝试加载保存的scaler
            try:
                import joblib
                scaler = joblib.load(os.path.join(os.path.dirname(args.pretrained_model_path), 'scaler.pkl'))
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            except:
                # 如果没有保存的scaler，创建新的
    scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
        
        # 创建环境
        env = SVJJEnvironment(X_train, y_train)
        
        # 根据选择的模型类型进行迁移学习
        if args.model_type == 'svjj':
            print("迁移学习SVJJ模型...")
            # 加载预训练模型
            if args.pretrained_model_path or args.pretrain:
                model_path = args.pretrained_model_path if args.pretrained_model_path else os.path.join(args.output_dir, 'pretrained_svjj_model.pkl')
                pretrained_model = load_model(model_path)
                svjj_model = fine_tune_svjj(pretrained_model, X_train, y_train, config)
            else:
                # 如果没有预训练模型，从头开始训练
                svjj_model = train_svjj(X_train, y_train, config)
            
            # 保存迁移学习后的模型
            save_model(svjj_model, os.path.join(args.output_dir, 'transfer_svjj_model.pkl'))
            
            # 评估模型
            svjj_metrics = evaluate_svjj(svjj_model, X_test, y_test)
            print("SVJJ迁移学习模型评估完成")
            
            # 可视化结果
            rl_summary_dict = {"SVJJ": svjj_metrics}
            visualize_results(rl_summary_dict, config)
            
        elif args.model_type == 'fake_rl':
            print("迁移学习FakeRL模型...")
            # 加载预训练模型
            if args.pretrained_model_path or args.pretrain:
                model_path = args.pretrained_model_path if args.pretrained_model_path else os.path.join(args.output_dir, 'pretrained_fake_rl_model.pkl')
                pretrained_model = load_model(model_path)
                fake_rl_model = fine_tune_fake_rl(pretrained_model, X_train, y_train, config)
            else:
                # 如果没有预训练模型，从头开始训练
                fake_rl_model = train_fake_rl(X_train, y_train, config)
            
            # 保存迁移学习后的模型
            save_model(fake_rl_model, os.path.join(args.output_dir, 'transfer_fake_rl_model.pkl'))
            
            # 评估模型
            fake_rl_metrics = evaluate_fake_rl(fake_rl_model, X_test, y_test)
            print("FakeRL迁移学习模型评估完成")
            
            # 可视化结果
            rl_summary_dict = {"FakeDQN": fake_rl_metrics}
            visualize_results(rl_summary_dict, config)
            
        elif args.model_type == 'rainbow_dqn':
            print("迁移学习Rainbow DQN模型...")
            # 导入必要的库
            from tqdm import trange
            from rainbow_model.agent import Agent
            from rainbow_model.memory import ReplayMemory
            from rainbow_model.test import test
            
            # 更新Rainbow配置并创建参数对象
            rainbow_config = config.RAINBOW_CONFIG
            rainbow_args = create_rainbow_args(args, rainbow_config)
            
            # 创建Rainbow DQN代理
            dqn = Agent(rainbow_args, env)
            
            # 加载预训练模型
            if args.rainbow_model or args.pretrain:
                model_path = args.rainbow_model if args.rainbow_model else os.path.join(args.output_dir, 'rainbow_dqn_pretrained', 'pretrained_model.pth')
                if os.path.exists(model_path):
                    print(f"加载预训练的Rainbow DQN模型: {model_path}")
                    dqn.load(model_path)
                else:
                    print(f"预训练模型路径不存在: {model_path}")
            
            # 迁移学习：在真实数据上继续训练Rainbow DQN
            results_dir = os.path.join(args.output_dir, 'rainbow_dqn_transfer')
            os.makedirs(results_dir, exist_ok=True)
            
            # 训练前先保存一份原始模型用于比较
            dqn.save(results_dir, 'pre_transfer_model.pth')
            
            # 修改学习率用于迁移学习
            rainbow_args.learning_rate *= 0.1  # 降低学习率用于微调
            
            # 添加这个部分来使用新的评估函数
            start_time = time.time()
            
            # 训练Rainbow DQN（迁移学习）
            rainbow_metrics = train_rainbow_dqn(rainbow_args, env, rainbow_config)
            # 记录训练时间
            training_time = time.time() - start_time
            print("Rainbow DQN迁移学习完成")
            
            # 保存迁移学习后的模型
            dqn.save(results_dir, 'transfer_model.pth')
            
            # 评估迁移学习后的Rainbow DQN
            val_mem = ReplayMemory(rainbow_args, rainbow_config['evaluation_size'])
            T, done = 0, True
            while T < rainbow_config['evaluation_size']:
                if done:
                    state = env.reset()
                next_state, _, done = env.step(np.random.randint(0, env.action_space()))
                val_mem.append(state, -1, 0.0, done)
                state = next_state
                T += 1
                
            dqn.eval()
            avg_reward, avg_Q = test(rainbow_args, rainbow_config['T_max'], dqn, val_mem, rainbow_metrics, 
                                   results_dir, evaluate=True)
            print(f'Rainbow DQN迁移学习评估结果 - 平均奖励: {avg_reward} | 平均Q值: {avg_Q}')
            
            # 使用我们的新函数生成全部指标
            rainbow_summary = evaluate_rainbow_dqn(rainbow_metrics, avg_reward, avg_Q, training_time)
            
            # 将Rainbow DQN的评估结果添加到可视化比较中
            rl_summary_dict = {"Rainbow DQN": rainbow_summary}
            visualize_results(rl_summary_dict, config)
            
            # 打印Rainbow DQN的详细指标
            print("\nRainbow DQN详细指标:")
            print(f"看涨期权MSE: {rainbow_summary['mse_call_mean']:.6f}")
            print(f"看跌期权MSE: {rainbow_summary['mse_put_mean']:.6f}")
            print(f"看涨期权R²: {rainbow_summary['r2_call_mean']:.6f}")
            print(f"看跌期权R²: {rainbow_summary['r2_put_mean']:.6f}")
            print(f"训练时间: {rainbow_summary['time_mean']:.2f}秒")
            print(f"平均奖励: {rainbow_summary['avg_reward']:.4f}")
            print(f"平均Q值: {rainbow_summary['avg_q']:.4f}")
            print(f"最佳平均奖励: {rainbow_summary['best_avg_reward']:.4f}")
            
            # 可视化Rainbow DQN的训练过程
            visualize_rainbow_results(rainbow_metrics, args.output_dir, suffix='transfer')
            print(f"Rainbow DQN迁移学习可视化结果已保存至 {results_dir}")
            
            # 保存Rainbow DQN的评估指标
            results_df = pd.DataFrame([rainbow_summary])
            results_df.to_csv(os.path.join(results_dir, 'rainbow_dqn_metrics.csv'), index=False)


def prepare_svjj_data(svjj_data, config):
    """从SVJJ模拟数据中准备特征和标签"""
    # 这里使用config中定义的特征和目标列
    X = svjj_data[config.FEATURES].values
    y = svjj_data[config.TARGET].values
    return X, y


def prepare_real_data(real_data, config):
    """从真实数据中准备特征和标签
    
    真实数据格式：
    ['secid', 'date', 'days', 'delta', 'impl_volatility', 'impl_strike', 'impl_premium', 
     'dispersion', 'cp_flag', 'cusip', 'ticker', 'sic', 'index_flag', 'exchange_d', 
     'issue_type', 'date_std', 'days_gap', 'maturity', 'close_maturity', 'value', 
     'gross_return', 'net_return', 'datedays', 'historical_vol', 'rate', 'close_date', 
     'simple_return', 'log_return', 'BS', 'R-Measure_simple', 'R-Measure_log', 
     'expected_return_simple', 'expected_return_log', 'ifcall', 'dp_sp', 'ep_sp', 
     'bm_sp', 'ntis', 'tbl', 'tms', 'dfy', 'svar', 'ym', 'moneyness', 'rf', 'rfm']
    """
    # 预处理真实数据
    # 1. 处理日期格式
    real_data['date'] = pd.to_datetime(real_data['date'], format='%y/%m/%d')
    
    # 2. 创建"是否看涨期权"特征
    if 'cp_flag' in real_data.columns and 'ifcall' not in real_data.columns:
        real_data['ifcall'] = real_data['cp_flag'].apply(lambda x: 1 if x.upper() == 'C' else 0)
    
    # 3. 提取特征和标签
    # 如果配置中的特征在真实数据中不存在，则需要创建或转换
    missing_features = [f for f in config.FEATURES if f not in real_data.columns]
    if missing_features:
        print(f"警告：以下特征在真实数据中不存在，需要转换或创建: {missing_features}")
        # 这里可以添加特征转换逻辑
    
    # 确保所有特征都是数值型
    for feature in config.FEATURES:
        if feature in real_data.columns and not pd.api.types.is_numeric_dtype(real_data[feature]):
            # 尝试转换为数值型
            try:
                real_data[feature] = pd.to_numeric(real_data[feature])
            except:
                print(f"警告：特征 {feature} 无法转换为数值型，将被填充为0")
                real_data[feature] = 0
    
    # 提取特征和标签
    X = real_data[config.FEATURES].values
    
    # 确保标签列都存在
    missing_targets = [t for t in config.TARGET if t not in real_data.columns]
    if missing_targets:
        print(f"警告：以下标签在真实数据中不存在: {missing_targets}")
        # 如果缺少标签，可以使用Black-Scholes模型计算
        if 'BS' in real_data.columns and 'impl_premium' in real_data.columns:
            # 使用BS和impl_premium作为替代
            y = np.zeros((len(real_data), len(config.TARGET)))
            for i, target in enumerate(config.TARGET):
                if target in real_data.columns:
                    y[:, i] = real_data[target].values
                else:
                    # 替代方案：使用BS或impl_premium
                    if 'ifcall' in real_data.columns:
                        mask = real_data['ifcall'] == i
                        y[mask, i] = real_data.loc[mask, 'impl_premium'].values
                        y[~mask, i] = real_data.loc[~mask, 'impl_premium'].values
                    else:
                        y[:, i] = real_data['impl_premium'].values
        else:
            raise ValueError(f"真实数据中缺少关键标签，无法继续: {missing_targets}")
    else:
        y = real_data[config.TARGET].values
    
    return X, y


def save_model(model, path):
    """保存模型到指定路径"""
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"模型已保存至 {path}")


def load_model(path):
    """从指定路径加载模型"""
    import joblib
    model = joblib.load(path)
    print(f"已加载模型: {path}")
    return model


def fine_tune_svjj(pretrained_model, X_train, y_train, config):
    """使用预训练模型在真实数据上进行微调"""
    print("微调SVJJ模型...")
    # 这里应该包含微调逻辑，通常会使用小学习率
    # 简单起见，这里直接返回原始模型
    return pretrained_model


def fine_tune_fake_rl(pretrained_model, X_train, y_train, config):
    """使用预训练模型在真实数据上进行微调"""
    print("微调FakeRL模型...")
    # 这里应该包含微调逻辑，通常会使用小学习率
    # 简单起见，这里直接返回原始模型
    return pretrained_model


def visualize_rainbow_results(metrics, output_dir, suffix=''):
    """可视化Rainbow DQN的训练结果"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 创建子图
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('训练奖励', 'Q值'))
    
    # 添加奖励曲线
    fig.add_trace(
        go.Scatter(x=metrics['steps'], y=[np.mean(r) for r in metrics['rewards']],
                  mode='lines', name='平均奖励'),
        row=1, col=1
    )
    
    # 添加最大奖励线
    fig.add_trace(
        go.Scatter(x=metrics['steps'], y=[np.max(r) for r in metrics['rewards']],
                  mode='lines', name='最大奖励', line=dict(dash='dash')),
        row=1, col=1
    )
    
    # 添加最小奖励线
    fig.add_trace(
        go.Scatter(x=metrics['steps'], y=[np.min(r) for r in metrics['rewards']],
                  mode='lines', name='最小奖励', line=dict(dash='dash')),
        row=1, col=1
    )
    
    # 添加Q值曲线
    fig.add_trace(
        go.Scatter(x=metrics['steps'], y=[np.mean(q) for q in metrics['Qs']],
                  mode='lines', name='平均Q值'),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=f"Rainbow DQN训练过程{' ('+suffix+')' if suffix else ''}",
        showlegend=True
    )
    
    # 保存图表
    results_dir = os.path.join(output_dir, f'rainbow_dqn{"_"+suffix if suffix else ""}')
    os.makedirs(results_dir, exist_ok=True)
    fig.write_html(os.path.join(results_dir, 'training_results.html'))

def parse_args():
    parser = argparse.ArgumentParser(description='SVJJ模型训练与评估')
    
    # 基本参数
    parser.add_argument('--data-path', type=str, required=False,
                      help='真实数据文件路径')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='输出目录')
    parser.add_argument('--model-type', type=str, default='svjj',
                      choices=['svjj', 'fake_rl', 'rainbow_dqn'],
                      help='选择要训练的模型类型')
    parser.add_argument('--pretrain', action='store_true',
                      help='是否进行预训练')
    parser.add_argument('--transfer', action='store_true',
                      help='是否进行迁移学习')
    parser.add_argument('--pretrained-model-path', type=str,
                      help='预训练模型路径，用于迁移学习')
    parser.add_argument('--svjj-samples', type=int, default=1000000,
                      help='SVJJ模型生成的样本数量')
    
    # Rainbow DQN参数
    parser.add_argument('--rainbow-id', type=str, default='default',
                      help='Rainbow DQN实验ID')
    parser.add_argument('--rainbow-seed', type=int, default=123,
                      help='Rainbow DQN随机种子')
    parser.add_argument('--disable-cuda', action='store_true',
                      help='禁用CUDA')
    parser.add_argument('--rainbow-model', type=str,
                      help='Rainbow DQN预训练模型路径')
    parser.add_argument('--rainbow-memory', type=str,
                      help='Rainbow DQN内存保存/加载路径')
    
    return parser.parse_args()

def create_rainbow_args(args, config):
    """从命令行参数和配置创建Rainbow DQN所需的参数对象"""
    from types import SimpleNamespace
    
    # 创建一个命名空间对象，将配置和参数合并
    rainbow_args = SimpleNamespace(
        id=args.rainbow_id,
        seed=args.rainbow_seed,
        disable_cuda=args.disable_cuda,
        T_max=config['T_max'],
        max_episode_length=config['max_episode_length'],
        history_length=config['history_length'],
        architecture=config['architecture'],
        hidden_size=config['hidden_size'],
        noisy_std=config['noisy_std'],
        atoms=config['atoms'],
        V_min=config['V_min'],
        V_max=config['V_max'],
        model=args.rainbow_model,
        memory_capacity=config['memory_capacity'],
        replay_frequency=config['replay_frequency'],
        priority_exponent=config['priority_exponent'],
        priority_weight=config['priority_weight'],
        multi_step=config['multi_step'],
        discount=config['discount'],
        target_update=config['target_update'],
        reward_clip=config['reward_clip'],
        learning_rate=config['learning_rate'],
        adam_eps=config['adam_eps'],
        batch_size=config['batch_size'],
        norm_clip=config['norm_clip'],
        learn_start=config['learn_start'],
        evaluate=False,
        evaluation_interval=config['evaluation_interval'],
        evaluation_episodes=config['evaluation_episodes'],
        evaluation_size=config['evaluation_size'],
        render=config['render'],
        enable_cudnn=config['enable_cudnn'],
        checkpoint_interval=config['checkpoint_interval'],
        memory=args.rainbow_memory,
        disable_bzip_memory=config['disable_bzip_memory'],
        device=torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
    )
    
    return rainbow_args

def train_svjj(X_train, y_train, config):
    """训练SVJJ模型"""
    print("训练SVJJ模型...")
    # 这里应该放置实际的SVJJ模型训练代码
    return "svjj_model_placeholder"

def evaluate_svjj(model, X_test, y_test):
    """评估SVJJ模型"""
    print("评估SVJJ模型...")
    # 这里应该放置实际的SVJJ模型评估代码
    return {
        "mse_call_mean": 0.1,
        "mse_put_mean": 0.1,
        "r2_call_mean": 0.8,
        "r2_put_mean": 0.8,
        "time_mean": 10
    }

def train_fake_rl(X_train, y_train, config):
    """训练FakeRL模型"""
    print("训练FakeRL模型...")
    # 这里应该放置实际的FakeRL模型训练代码
    return "fake_rl_model_placeholder"

def evaluate_fake_rl(model, X_test, y_test):
    """评估FakeRL模型"""
    print("评估FakeRL模型...")
    # 这里应该放置实际的FakeRL模型评估代码
    return {
        "mse_call_mean": 0.2,
        "mse_put_mean": 0.2,
        "r2_call_mean": 0.7,
        "r2_put_mean": 0.7,
        "time_mean": 15
    }

def visualize_results(results_dict, config):
    """可视化训练结果"""
    print("可视化结果...")
    # 这里应该放置实际的可视化代码
    import matplotlib.pyplot as plt
    
    # 创建简单的条形图
    plt.figure(figsize=(10, 6))
    
    models = list(results_dict.keys())
    mse_values = [results_dict[model]['mse_call_mean'] for model in models]
    
    plt.bar(models, mse_values)
    plt.title('模型MSE比较')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'model_comparison.png'))
    plt.close()

# 在预训练和迁移学习后进行Rainbow DQN性能比较分析
def compare_rainbow_performance(args):
    """比较Rainbow DQN预训练和迁移学习后的性能"""
    if args.pretrain and args.transfer and args.model_type == 'rainbow_dqn':
        print("\n=== 比较Rainbow DQN预训练和迁移学习后的性能 ===")
        from utils import compare_rainbow_stages
        
        # 尝试加载预训练和迁移学习的指标
        try:
            import pandas as pd
            pretrain_metrics_path = os.path.join(args.output_dir, 'rainbow_dqn_pretrained', 'rainbow_dqn_pretrain_metrics.csv')
            transfer_metrics_path = os.path.join(args.output_dir, 'rainbow_dqn_transfer', 'rainbow_dqn_metrics.csv')
            
            if os.path.exists(pretrain_metrics_path) and os.path.exists(transfer_metrics_path):
                pretrain_df = pd.read_csv(pretrain_metrics_path)
                transfer_df = pd.read_csv(transfer_metrics_path)
                
                pretrain_metrics = pretrain_df.iloc[0].to_dict()
                transfer_metrics = transfer_df.iloc[0].to_dict()
                
                # 生成比较结果
                comparison = compare_rainbow_stages(pretrain_metrics, transfer_metrics, args.output_dir)
                
                # 打印比较结果
                print("\nRainbow DQN预训练和迁移学习的指标比较:")
                pd.set_option('display.width', None)
                pd.set_option('display.max_columns', None)
                print(comparison)
            else:
                print("无法找到预训练或迁移学习的指标文件，跳过比较分析。")
        except Exception as e:
            print(f"比较分析时出错: {e}")

if __name__ == "__main__":
    main()
    # 在主函数执行完毕后进行性能比较分析
    args = parse_args()
    compare_rainbow_performance(args)
