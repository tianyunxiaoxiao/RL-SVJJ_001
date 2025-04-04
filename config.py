# (1) 全局配置文件
# === config.py ===
"""
配置文件：
1. 存放环境变量(示例DATA_PATH等)
2. 统一管理输入数据结构(哪些列是特征，哪些列是标签)
3. 可以在此定义不同模型的参数
4. 配置评估指标和可视化参数
5. 新增：迁移学习相关配置
"""

import os

# (A) 一些通用环境配置(数据路径等)
# 你可根据实际情况改为读取 .env 或 argparse，这里直接写死为示例
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATED_DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_data.csv")  # 模拟数据路径
REAL_DATA_PATH = "/Users/a1/Desktop/ML999.csv"  # 真实数据路径 
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 存放结果的目录
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")  # 存放图表的目录
MODELS_DIR = os.path.join(RESULTS_DIR, "models")  # 保存模型的目录
TRANSFER_DIR = os.path.join(RESULTS_DIR, "transfer_models")  # 保存迁移模型的目录

# 创建必要的目录
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRANSFER_DIR, exist_ok=True)

# (B) 数据结构配置
SIMULATED_DATA_CONFIG = {
    "features": ["S", "V", "K", "tau", "moneyness"],  # 模拟数据的特征列
    "labels": ["call_price", "put_price"],            # 模拟数据的标签列
    "test_size": 0.2,                                # 测试集比例
    "random_state": 42                               # 随机种子
}

# 真实数据配置
REAL_DATA_CONFIG = {
    "features": ["impl_strike", "impl_volatility", "days", "moneyness", "rate"],  # 真实数据的特征列
    "labels": ["impl_premium"],                                                  # 真实数据的标签列
    "feature_mapping": {  # 从模拟数据到真实数据的特征映射
        "K": "impl_strike",
        "V": "impl_volatility", 
        "tau": "days",
        "moneyness": "moneyness",
        "S": "rate"  # 这只是示例映射，可能需要调整
    },
    "label_mapping": {  # 从模拟数据到真实数据的标签映射
        "call_price": "impl_premium",  # 当ifcall=1时
        "put_price": "impl_premium"    # 当ifcall=0时
    },
    "test_size": 0.2,
    "random_state": 42,
    "date_format": "%y/%m/%d",         # 日期格式，如"98/01/02"
    "target_column": "ifcall"          # 判断是看涨还是看跌期权的列
}

# (C) 迁移学习配置
TRANSFER_LEARNING_CONFIG = {
    "enabled": True,                    # 是否启用迁移学习
    "pretraining_samples": 1000000,     # 预训练数据样本数量
    "fine_tuning_ratio": 0.8,           # 微调时解冻层的比例
    "learning_rate_decay": 0.1,         # 迁移学习时学习率衰减
    "batch_size_reduce": 0.5,           # 迁移学习时批量大小减少比例
    "freeze_layers": False,             # 是否冻结部分层(仅适用于深度学习模型)
    "epochs_pretrain": 100,             # 预训练轮数
    "epochs_finetune": 50,              # 微调轮数
    "save_pretrained": True             # 是否保存预训练模型
}

# (D) 评估指标配置
METRICS_CONFIG = {
    "standard_metrics": ["mse", "rmse", "mae", "r2"],  # 标准评估指标
    "advanced_metrics": ["median_ae", "max_error", "explained_variance"],  # 高级评估指标
    "relative_metrics": ["mape", "rmae", "rmse_pct"],  # 相对评估指标
    "option_specific_metrics": [                        # 期权特定指标
        "time_value_rmse", "time_value_mae",
        "in_money_rmse", "in_money_mae",
        "out_money_rmse", "out_money_mae"
    ]
}

# (E) 传统机器学习模型参数
ML_MODELS_PARAMS = {
    "n_runs": 5,                          # 重复训练次数
    "verbose": True,                      # 是否打印详细信息
    
    # 各模型超参数
    "LinearRegression": {},               # 线性回归无需额外参数
    "Lasso": {"alpha": 0.01, "max_iter": 10000},
    "Ridge": {"alpha": 0.1},
    "ElasticNet": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 10000},
    "SGDRegressor": {"max_iter": 1000, "tol": 1e-4},
    "DecisionTree": {"max_depth": 10, "min_samples_split": 5},
    "RandomForest": {"n_estimators": 100, "max_depth": 10, "n_jobs": -1},
    "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
    "SVR-Linear": {"kernel": "linear", "C": 1.0, "epsilon": 0.1},
    "SVR-RBF": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
    
    # 超参数调优设置
    "tuning": {
        "enabled": False,                 # 是否启用超参数调优
        "tuning_type": "random",          # "grid"或"random"
        "n_iter": 10,                     # RandomizedSearchCV的迭代次数
        "cv": 3,                          # 交叉验证折数
        "verbose": 1                      # 详细程度
    },
    
    # 迁移学习特定参数
    "transfer": {
        "warm_start_models": ["RandomForest", "GradientBoosting"],  # 支持warm_start的模型
        "recalibration_models": ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]  # 支持重新校准的模型
    }
}

# (F) 深度学习模型参数
DL_MODELS_PARAMS = {
    "common": {
        "epochs": 100,                    # 训练轮数
        "batch_size": 64,                 # 批量大小
        "weight_decay": 1e-4,             # L2正则化系数
        "patience": 10,                    # 早停耐心值
        "lr_scheduler": "plateau",        # 学习率调度器类型：'plateau', 'cosine', None
        "device": "cpu",                  # 训练设备：'cpu', 'cuda'
        "verbose": True,                  # 是否打印训练进度
        "runs": 3,                        # 每个模型运行次数
        
        # 迁移学习特定参数
        "transfer": {
            "freeze_ratio": 0.5,          # 迁移学习时冻结层的比例
            "finetune_lr": 1e-4,          # 微调学习率
            "finetune_wd": 1e-5           # 微调权重衰减
        }
    },
    
    "DeepMLP": {
        "hidden_layers": [64, 128, 64],   # 隐藏层维度
        "dropout_rate": 0.2,              # Dropout比率
        "lr": 1e-3                        # 学习率
    },
    
    "ResidualMLP": {
        "hidden_dim": 64,                 # 隐藏层维度
        "blocks": 3,                      # 残差块数量
        "dropout_rate": 0.2,              # Dropout比率
        "lr": 1e-3                        # 学习率
    },
    
    "SimpleCNN": {
        "channels": [16, 32],             # 卷积通道数
        "kernel_size": 3,                 # 卷积核大小
        "dropout_rate": 0.2,              # Dropout比率
        "lr": 1e-3                        # 学习率
    },
    
    "AdvancedCNN": {
        "channels": [16, 32, 64, 128],    # 卷积通道数
        "kernel_size": 3,                 # 卷积核大小
        "dropout_rate": 0.2,              # Dropout比率
        "lr": 1e-3                        # 学习率
    },
    
    "SimpleRNN": {
        "hidden_dim": 64,                 # 隐藏层维度
        "num_layers": 2,                  # RNN层数
        "dropout_rate": 0.2,              # Dropout比率
        "bidirectional": True,            # 是否双向
        "lr": 1e-3                        # 学习率
    },
    
    "SimpleLSTM": {
        "hidden_dim": 64,                 # 隐藏层维度
        "num_layers": 2,                  # LSTM层数
        "dropout_rate": 0.2,              # Dropout比率
        "bidirectional": True,            # 是否双向
        "lr": 1e-3                        # 学习率
    },
    
    "SimpleGRU": {
        "hidden_dim": 64,                 # 隐藏层维度
        "num_layers": 2,                  # GRU层数
        "dropout_rate": 0.2,              # Dropout比率
        "bidirectional": True,            # 是否双向
        "lr": 1e-3                        # 学习率
    },
    
    "SimpleTransformer": {
        "d_model": 64,                    # 模型维度
        "nhead": 4,                       # 注意力头数
        "num_layers": 2,                  # Transformer层数
        "dim_feedforward": 128,           # 前馈网络维度
        "dropout_rate": 0.1,              # Dropout比率
        "lr": 1e-3                        # 学习率
    },
    
    "BayesianNN": {
        "hidden_dims": [64, 32],          # 隐藏层维度
        "prior_sigma_1": 0.1,             # 先验分布参数1
        "prior_sigma_2": 0.001,           # 先验分布参数2
        "prior_pi": 0.5,                  # 先验分布参数pi
        "lr": 1e-3                        # 学习率
    }
}

# (G) 强化学习配置
RL_COMPARE_CONFIG = {
    "algorithm_info": {
        "dqn": {
            "full_name": "Deep Q-Network",
            "reference": "Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.",
            "library": "stable-baselines3"
        },
        "ppo": {
            "full_name": "Proximal Policy Optimization",
            "reference": "Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.",
            "library": "stable-baselines3"
        },
        "td3": {
            "full_name": "Twin Delayed DDPG",
            "reference": "Fujimoto, S. et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML 2018.",
            "library": "stable-baselines3"
        },
        "sac": {
            "full_name": "Soft Actor-Critic",
            "reference": "Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018.",
            "library": "stable-baselines3"
        },
        "actor_critic": {
            "full_name": "Actor-Critic (Custom)",
            "reference": "Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. NIPS 2000.",
            "library": "custom implementation"
        }
    },
    
    "env": {
        "n_samples": 10000,
        "n_features": 5
    },
    
    "training": {
        "timesteps": 10000,
        "max_train_time": 300,  # 每个算法最多训练5分钟
        "runs": 3
    },
    
    "results": {
        "dir": "results/rl",
        "save_csv": True,
        "save_plots": True
    },
    
    "visualization": {
        "figsize": (15, 10),
        "dpi": 100,
        "style": "seaborn-whitegrid",
        "colors": {
            "time": "skyblue",
            "reward": "salmon",
            "steps": "lightgreen",
            "efficiency": "mediumpurple"
        },
        "save_format": "png"
    },
    
    # 启用的RL算法
    "dqn": {
        "enabled": True,
        "learning_rate": 0.0003,
        "buffer_size": 10000,
        "batch_size": 64,
        "learning_starts": 1000,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
        "net_arch": [64, 64]
    },
    
    "ppo": {
        "enabled": True,
        "learning_rate": 0.0003,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
    },
    
    "td3": {
        "enabled": False,  # 默认禁用，除非需要连续动作空间
        "learning_rate": 0.0003,
        "buffer_size": 10000,
        "batch_size": 100,
        "learning_starts": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "net_arch": [400, 300]
    },
    
    "sac": {
        "enabled": False,  # 默认禁用，除非需要连续动作空间
        "learning_rate": 0.0003,
        "buffer_size": 10000,
        "batch_size": 64,
        "learning_starts": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "net_arch": [dict(pi=[256, 256], qf=[256, 256])]
    },
    
    "actor_critic": {
        "enabled": True,
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "entropy_weight": 0.01
    },
    
    # 迁移学习特定配置
    "transfer": {
        "enabled": True,
        "finetune_timesteps": 5000,
        "finetune_lr": 0.0001,
        "exploration_fraction": 0.1,
        "buffer_reuse_ratio": 0.2  # 在微调时保留的预训练经验比例
    }
}

# Rainbow DQN配置
RAINBOW_CONFIG = {
    # 基本训练参数
    'id': 'rainbow_svjj',            # 实验名称
    'seed': 123,                     # 随机种子
    'disable_cuda': False,           # 是否禁用CUDA
    'T_max': int(1e5),               # 训练步数 (简化为10万步)
    'max_episode_length': int(1e4),  # 最大episode长度
    'history_length': 4,             # 连续状态数量
    'architecture': 'data-efficient', # 网络架构: 'canonical' 或 'data-efficient'
    'hidden_size': 128,              # 网络隐藏层大小 (简化为128)
    'noisy_std': 0.1,                # 噪声线性层的初始标准差
    
    # 分布式Q学习参数
    'atoms': 51,                     # 值分布的离散化大小
    'V_min': -10,                    # 值分布支撑的最小值
    'V_max': 10,                     # 值分布支撑的最大值
    
    # 经验回放参数
    'memory_capacity': int(1e4),     # 经验回放缓冲区容量 (简化为1万)
    'replay_frequency': 4,           # 从内存采样的频率
    'priority_exponent': 0.5,        # 优先经验回放指数
    'priority_weight': 0.4,          # 初始优先经验回放重要性采样权重
    
    # 多步学习参数
    'multi_step': 3,                 # 多步回报的步数
    'discount': 0.99,                # 折扣因子
    
    # 网络更新参数
    'target_update': int(1e3),       # 目标网络更新步数 (简化为1000步)
    'reward_clip': 1,                # 奖励裁剪
    'learning_rate': 0.0001,         # 学习率 (增大以加速训练)
    'adam_eps': 1.5e-4,              # Adam epsilon
    'batch_size': 32,                # 批量大小
    'norm_clip': 10,                 # 梯度裁剪的L2范数
    'learn_start': int(1e3),         # 开始训练的步数 (简化为1000步)
    
    # 评估参数
    'evaluation_interval': 5000,     # 评估间隔 (简化为5000步)
    'evaluation_episodes': 10,       # 评估episode数量
    'evaluation_size': 500,          # 用于验证Q的转换数量
    
    # 其他参数
    'render': False,                 # 是否显示屏幕
    'enable_cudnn': True,            # 是否启用cuDNN
    'checkpoint_interval': 5000,     # 模型检查点间隔 (简化为5000步)
    'disable_bzip_memory': True,     # 是否不压缩内存文件
    
    # 特定于SVJJ环境的参数
    'model': None,                   # 预训练模型路径
    'memory': None,                  # 内存文件路径
    
    # 迁移学习特定配置
    'transfer': {
        'enabled': True,
        'finetune_steps': int(5e4),   # 微调步数
        'finetune_lr': 5e-5,          # 微调学习率
        'priority_exponent': 0.4,     # 微调时的优先经验回放指数
        'memory_reuse_ratio': 0.2     # 保留的预训练经验比例
    }
}

# (H) 可视化配置
VISUALIZATION_CONFIG = {
    "figsize": (15, 10),                  # 图表大小
    "save_format": "png",                 # 保存格式：'png', 'pdf', 'svg'
    "dpi": 300,                           # 分辨率
    "style": "seaborn-whitegrid",         # 样式
    "color_palette": "viridis",           # 色彩方案
    "font_family": "DejaVu Sans",         # 字体
    "font_size": 12,                      # 字体大小
    "title_font_size": 16,                # 标题字体大小
    "legend_font_size": 10,               # 图例字体大小
    "show_grid": True,                    # 是否显示网格
    "use_tex": False,                     # 是否使用TeX渲染
    
    # 迁移学习可视化配置
    "transfer_learning": {
        "compare_metrics": ["rmse", "mae", "r2"],  # 用于比较的指标
        "color_pretrain": "blue",            # 预训练模型颜色
        "color_finetune": "red",             # 微调模型颜色
        "color_direct": "green",             # 直接训练模型颜色
        "alpha": 0.7,                        # 透明度
        "bar_width": 0.25,                   # 柱状图宽度
        "annotate_values": True,             # 是否标注数值
        "include_improvements": True         # 是否显示改进百分比
    }
}
