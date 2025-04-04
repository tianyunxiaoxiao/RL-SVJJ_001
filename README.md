# Rainbow SVJJ 项目

这个项目实现了一套完整的期权定价模型比较框架，包括传统机器学习、深度学习、强化学习和Rainbow DQN模型。项目的主要目标是比较不同类型的算法在期权定价任务上的表现。

## 项目结构

```
├── config.py                  # 配置文件，存储所有超参数和路径
├── main.py                    # 主入口脚本
├── main_svjj.py               # 替代入口脚本（更复杂的功能）
├── ml_dl_models_compare.py    # 传统ML和DL模型实现和比较
├── rl_compare.py              # 强化学习模型比较
├── svjj_model.py              # SVJJ模型定义和数据生成
├── utils.py                   # 工具函数
├── data/                      # 数据目录
│   └── simulated_data.csv     # 模拟数据
├── results/                   # 结果目录
└── rainbow_model/             # Rainbow DQN模型实现
    ├── __init__.py
    ├── agent.py              # DQN代理
    ├── env.py                # 环境封装
    ├── memory.py             # 优先经验回放
    ├── model.py              # DQN网络结构
    ├── test.py               # 测试函数
    └── train.py              # 训练函数
```

## 安装

1. 克隆本仓库:
   ```bash
   git clone https://github.com/yourusername/rainbow_svjj_project.git
   cd rainbow_svjj_project
   ```

2. 创建并激活虚拟环境（推荐）:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/MacOS
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 数据

项目使用随机波动率跳跃扩散模型(SVCJ)生成的模拟期权定价数据。您可以使用以下方式生成数据:

```bash
python main.py --data_source simulated --samples 10000
```

生成的数据将保存在`data/simulated_data.csv`。

## 使用方法

### 训练所有模型并比较

```bash
python main.py --mode train --model_type all --output_dir results --verbose
```

### 只训练特定类型的模型

```bash
# 只训练传统机器学习模型
python main.py --mode train --model_type ml --output_dir results/ml

# 只训练深度学习模型
python main.py --mode train --model_type dl --output_dir results/dl

# 只训练强化学习模型
python main.py --mode train --model_type rl --output_dir results/rl

# 只训练Rainbow DQN模型
python main.py --mode train --model_type rainbow --output_dir results/rainbow
```

### 调整Rainbow DQN的训练步数

```bash
python main.py --mode train --model_type rainbow --rainbow_steps 50000
```

### 保存训练好的模型

```bash
python main.py --mode train --model_type all --save_models
```

## 模型说明

### 传统机器学习模型
- 线性回归 (OLS)
- Lasso回归
- Ridge回归
- ElasticNet
- SGD回归器
- 决策树
- 随机森林
- 梯度提升树
- AdaBoost
- K近邻
- 支持向量回归 (线性)
- 支持向量回归 (RBF核)

### 深度学习模型
- DeepMLP：深层多层感知机
- ResidualMLP：带残差连接的多层感知机
- SimpleCNN：简单的一维卷积神经网络
- AdvancedCNN：更复杂的卷积神经网络
- SimpleRNN：循环神经网络
- SimpleLSTM：长短期记忆网络
- SimpleGRU：门控循环单元网络
- SimpleTransformer：简单的Transformer模型
- BayesianNN：贝叶斯神经网络

### 强化学习模型
- DQN：深度Q网络
- PPO：近端策略优化
- Actor-Critic：自定义Actor-Critic实现

### Rainbow DQN
完整的Rainbow DQN实现，包含以下改进:
- 双Q学习
- 优先经验回放
- 分布式RL（C51）
- 噪声网络
- 多步学习
- 残差网络架构

## 结果和可视化

训练完成后，结果将保存在指定的输出目录中：
- `ml_results.csv`：传统机器学习模型的性能指标
- `dl_results.json`：深度学习模型的性能指标
- `rl_results.json`：强化学习模型的性能指标
- `rainbow_metrics.json`：Rainbow DQN的训练指标
- `model_comparison.png`：所有模型性能的可视化比较
- `all_models_metrics.csv`：综合性能指标
- `metrics_report.txt`：详细的性能报告

## 参考文献

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
- Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. 