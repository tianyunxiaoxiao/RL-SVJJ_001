# (3) 传统ML + DL模型的对比
# === ml_dl_models_compare.py ===
"""
对比非强化学习的模型（传统ML + DL）。
包括：
    - train_and_evaluate_ml_models：传统机器学习模型（线性回归、决策树等）
    - 深度学习模型：
        - SimpleCNN：一维卷积神经网络
        - AdvancedCNN：多层卷积神经网络
        - SimpleRNN：简单循环神经网络
        - SimpleLSTM：长短期记忆网络
        - SimpleGRU：门控循环单元网络
        - SimpleTransformer：简单Transformer模型
        - DeepMLP：深层多层感知机
        - ResidualMLP：残差连接的多层感知机
        - BayesianNN：贝叶斯神经网络
    - 评估工具：
    - run_multiple_times_dl：在同一个模型上做多次训练对比
        - train_dl_model：训练深度学习模型
        - evaluate_models_performance：比较不同模型性能
    - 迁移学习功能：
        - train_and_transfer_ml_models：传统机器学习模型的迁移学习
        - train_and_transfer_dl_models：深度学习模型的迁移学习
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.distributions as distributions

# scikit-learn相关
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.base import BaseEstimator, clone

# 项目模块
import config
from utils import (
    get_features_and_labels, 
    calculate_metrics, 
    evaluate_predictions, 
    visualize_model_comparisons,
    save_all_metrics
)
from transfer_utils import TransferDataProcessor

# 1) 传统机器学习对比
def train_and_evaluate_ml_models(X_train, y_train, X_test, y_test, n_runs=5, verbose=True):
    """
    训练和评估多种传统机器学习模型
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签，shape为(n_samples, 2)，包含call和put价格
        X_test: 测试集特征
        y_test: 测试集标签，shape为(n_samples, 2)
        n_runs: 每个模型运行次数
        verbose: 是否打印训练过程
        
    返回:
        summary: 所有模型的性能摘要统计
        results: 所有模型的详细结果
    """
    # 定义要评估的模型
    model_constructors = {
        'OLS': lambda: LinearRegression(),
        'Lasso': lambda: Lasso(alpha=0.01, max_iter=10000, tol=1e-4),
        'Ridge': lambda: Ridge(alpha=0.1),
        'ElasticNet': lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, tol=1e-4),
        'SGDRegressor': lambda: SGDRegressor(max_iter=1000, tol=1e-4),
        'DecisionTree': lambda: DecisionTreeRegressor(max_depth=10, min_samples_split=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        'AdaBoost': lambda: AdaBoostRegressor(n_estimators=50),
        'KNN': lambda: KNeighborsRegressor(n_neighbors=5),
        'SVR-Linear': lambda: SVR(kernel='linear', C=1.0, epsilon=0.1),
        'SVR-RBF': lambda: SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
            }

    # 初始化结果字典
    results = {}
    for m_name in model_constructors:
        results[m_name] = {
            'time': [],
            'mse_call': [],
            'mse_put': [],
            'mae_call': [],
            'mae_put': [],
            'r2_call': [],
            'r2_put': []
            }

    # 为每个模型进行n_runs次训练和评估
    if verbose:
        print("开始训练和评估传统机器学习模型...")

    for m_name, ctor in model_constructors.items():
        if verbose:
            print(f"正在处理模型: {m_name}")
        
        for run in range(n_runs):
            if verbose:
                print(f"  运行 {run+1}/{n_runs}", end="\r")
            
            start_t = time.time()

            # 分别训练看涨期权和看跌期权的模型
            y_call_train = y_train[:,0]
            y_put_train = y_train[:,1]
            y_call_test = y_test[:,0]
            y_put_test = y_test[:,1]

            # 创建并训练模型
            try:
                # 创建模型实例
                model_call = ctor()
                model_put = ctor()

                # 训练模型
                model_call.fit(X_train, y_call_train)
                model_put.fit(X_train, y_put_train)

                # 预测
                pred_call = model_call.predict(X_test)
                pred_put = model_put.predict(X_test)

                # 计算多种评估指标
                mse_call = mean_squared_error(y_call_test, pred_call)
                mse_put = mean_squared_error(y_put_test, pred_put)
                mae_call = mean_absolute_error(y_call_test, pred_call)
                mae_put = mean_absolute_error(y_put_test, pred_put)
                r2_call = r2_score(y_call_test, pred_call)
                r2_put = r2_score(y_put_test, pred_put)
                
                # 记录耗时和评估指标
                elapsed = time.time() - start_t
                results[m_name]["time"].append(elapsed)
                results[m_name]["mse_call"].append(mse_call)
                results[m_name]["mse_put"].append(mse_put)
                results[m_name]["mae_call"].append(mae_call)
                results[m_name]["mae_put"].append(mae_put)
                results[m_name]["r2_call"].append(r2_call)
                results[m_name]["r2_put"].append(r2_put)
                
            except Exception as e:
                if verbose:
                    print(f"模型 {m_name} 在第 {run+1} 次运行时发生错误: {str(e)}")
        
        if verbose:
            print(f"完成模型: {m_name}")

# 新增: 用于迁移学习的传统机器学习模型训练与评估函数
def train_and_transfer_ml_models(source_data: Dict[str, np.ndarray], 
                              target_data: Dict[str, np.ndarray], 
                              models_to_use: List[str] = None,
                              n_runs: int = 3, 
                              verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    训练传统机器学习模型并将其迁移到目标域
    
    参数:
        source_data: 源域数据字典，包含'X_train'、'y_train'等键
        target_data: 目标域数据字典，包含'X_train'、'y_train'等键
        models_to_use: 要使用的模型列表，如果为None，则使用所有模型
        n_runs: 每个模型运行次数
        verbose: 是否打印详细信息
        
    返回:
        results: 包含预训练、迁移和直接训练结果的字典
    """
    # 定义模型构造函数
    model_constructors = {
        'OLS': lambda: LinearRegression(),
        'Lasso': lambda: Lasso(alpha=0.01, max_iter=10000, tol=1e-4),
        'Ridge': lambda: Ridge(alpha=0.1),
        'ElasticNet': lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, tol=1e-4),
        'SGDRegressor': lambda: SGDRegressor(max_iter=1000, tol=1e-4),
        'DecisionTree': lambda: DecisionTreeRegressor(max_depth=10, min_samples_split=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        'AdaBoost': lambda: AdaBoostRegressor(n_estimators=50),
        'KNN': lambda: KNeighborsRegressor(n_neighbors=5),
        'SVR-Linear': lambda: SVR(kernel='linear', C=1.0, epsilon=0.1),
        'SVR-RBF': lambda: SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
            }
    
    # 如果指定了要使用的模型，则只使用这些模型
    if models_to_use:
        model_constructors = {name: ctor for name, ctor in model_constructors.items() if name in models_to_use}
    
    # 识别支持warm_start的模型和支持重校准的模型
    warm_start_models = config.ML_MODELS_PARAMS["transfer"]["warm_start_models"]
    recalibration_models = config.ML_MODELS_PARAMS["transfer"]["recalibration_models"]
    
    # 提取数据
    X_source_train = source_data["X_train"]
    y_source_train = source_data["y_train"]
    X_source_test = source_data["X_test"]
    y_source_test = source_data["y_test"]
    
    X_target_train = target_data["X_train"]
    y_target_train = target_data["y_train"]
    X_target_test = target_data["X_test"]
    y_target_test = target_data["y_test"]
    
    # 初始化结果字典
    results = {}
    for model_name in model_constructors:
        results[model_name] = {
            "pretrain": {},  # 预训练模型在源域上的性能
            "finetune": {},  # 迁移学习后在目标域上的性能
            "direct": {}     # 直接在目标域上训练的性能
            }
    
    # 对每个模型进行训练和迁移
    if verbose:
        print("开始传统机器学习模型的迁移学习...")
    
    for model_name, constructor in model_constructors.items():
        if verbose:
            print(f"\n处理模型: {model_name}")
        
        # 存储每次运行的结果
        pretrain_metrics = {
            "rmse": [], "mae": [], "r2": [], "time": []
            }
        finetune_metrics = {
            "rmse": [], "mae": [], "r2": [], "time": []
            }
        direct_metrics = {
            "rmse": [], "mae": [], "r2": [], "time": []
            }
        
        for run in range(n_runs):
            if verbose:
                print(f"  运行 {run+1}/{n_runs}")
            
            # 1. 在源域上预训练模型
            start_time = time.time()
            source_model = constructor()
            source_model.fit(X_source_train, y_source_train)
            
            # 在源域测试集上评估
            source_preds = source_model.predict(X_source_test)
            source_metrics = calculate_metrics(y_source_test, source_preds)
            pretrain_time = time.time() - start_time
            
            pretrain_metrics["rmse"].append(source_metrics["rmse"])
            pretrain_metrics["mae"].append(source_metrics["mae"])
            pretrain_metrics["r2"].append(source_metrics["r2"])
            pretrain_metrics["time"].append(pretrain_time)
            
            if verbose:
                print(f"    预训练完成，源域RMSE: {source_metrics['rmse']:.6f}, R2: {source_metrics['r2']:.6f}")
            
            # 2. 迁移学习到目标域
            start_time = time.time()
            
            if model_name in warm_start_models:
                # 对于支持warm_start的模型，使用已训练的模型继续训练
                if hasattr(source_model, 'warm_start'):
                    # 备份模型
                    transferred_model = clone(source_model)
                    # 设置warm_start=True
                    if hasattr(transferred_model, 'set_params'):
                        transferred_model.set_params(warm_start=True)
                    # 在目标域数据上继续训练
                    transferred_model.fit(X_target_train, y_target_train)
                else:
                    # 如果模型没有warm_start属性，但在配置中标记为支持，则尝试其他方法
                    if model_name == 'RandomForest' or model_name == 'GradientBoosting':
                        # 复制模型
                        transferred_model = clone(source_model)
                        # 保存原始估计器
                        original_estimators = source_model.estimators_.copy() if hasattr(source_model, 'estimators_') else []
                        # 在目标域训练额外的估计器
                        transferred_model.fit(X_target_train, y_target_train)
                        # 合并估计器
                        if hasattr(transferred_model, 'estimators_') and original_estimators:
                            # 使用前一半源域估计器和后一半目标域估计器
                            n_estimators = len(transferred_model.estimators_)
                            half_n = n_estimators // 2
                            transferred_model.estimators_[:half_n] = original_estimators[:half_n]
                    else:
                        # 默认情况下，简单地在目标域上重新训练
                        transferred_model = constructor()
                        transferred_model.fit(X_target_train, y_target_train)
            
            elif model_name in recalibration_models:
                # 对于线性模型，可以使用源模型的系数作为初始值，然后在目标域上调整
                if model_name == 'OLS' or model_name.startswith('Ridge') or model_name.startswith('Lasso') or model_name == 'ElasticNet':
                    # 创建新模型
                    transferred_model = constructor()
                    
                    # 如果有源模型的系数，将其作为初始权重
                    if hasattr(source_model, 'coef_'):
                        # 这里是简化处理，实际上我们需要更复杂的机制来调整系数
                        # 假设我们在目标域上使用源域系数的0.5倍作为初始化
                        transferred_model.fit(X_target_train, y_target_train)
                        
                        # 在部分线性模型中混合源和目标系数
                        if hasattr(transferred_model, 'coef_'):
                            blend_ratio = 0.5  # 源域和目标域权重混合比例
                            transferred_model.coef_ = (blend_ratio * source_model.coef_ + 
                                                      (1 - blend_ratio) * transferred_model.coef_)
                        
                        if hasattr(transferred_model, 'intercept_'):
                            transferred_model.intercept_ = (blend_ratio * source_model.intercept_ + 
                                                         (1 - blend_ratio) * transferred_model.intercept_)
                    else:
                        # 如果无法获取系数，则简单地在目标域上训练
                        transferred_model.fit(X_target_train, y_target_train)
                else:
                    # 对于其他模型，简单地在目标域上重新训练
                    transferred_model = constructor()
                    transferred_model.fit(X_target_train, y_target_train)
            
            else:
                # 对于不支持迁移的模型，在目标域上重新训练
                transferred_model = constructor()
                transferred_model.fit(X_target_train, y_target_train)
            
            # 在目标域测试集上评估迁移模型
            transfer_preds = transferred_model.predict(X_target_test)
            transfer_metrics = calculate_metrics(y_target_test, transfer_preds)
            transfer_time = time.time() - start_time
            
            finetune_metrics["rmse"].append(transfer_metrics["rmse"])
            finetune_metrics["mae"].append(transfer_metrics["mae"])
            finetune_metrics["r2"].append(transfer_metrics["r2"])
            finetune_metrics["time"].append(transfer_time)
            
            if verbose:
                print(f"    迁移学习完成，目标域RMSE: {transfer_metrics['rmse']:.6f}, R2: {transfer_metrics['r2']:.6f}")
            
            # 3. 直接在目标域上训练模型作为基线
            start_time = time.time()
            direct_model = constructor()
            direct_model.fit(X_target_train, y_target_train)
            
            # 在目标域测试集上评估
            direct_preds = direct_model.predict(X_target_test)
            direct_metrics = calculate_metrics(y_target_test, direct_preds)
            direct_time = time.time() - start_time
            
            direct_metrics["rmse"].append(direct_metrics["rmse"])
            direct_metrics["mae"].append(direct_metrics["mae"])
            direct_metrics["r2"].append(direct_metrics["r2"])
            direct_metrics["time"].append(direct_time)
            
            if verbose:
                print(f"    直接训练完成，目标域RMSE: {direct_metrics['rmse']:.6f}, R2: {direct_metrics['r2']:.6f}")
        
        # 计算每个方法的平均指标
        for method, method_metrics in [("pretrain", pretrain_metrics), 
                                     ("finetune", finetune_metrics), 
                                     ("direct", direct_metrics)]:
            for metric, values in method_metrics.items():
                if values:  # 确保有值
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    results[model_name][method][metric] = mean_val
                    results[model_name][method][f"{metric}_std"] = std_val
        
        # 比较迁移学习和直接训练的性能差异
        if "rmse" in results[model_name]["finetune"] and "rmse" in results[model_name]["direct"]:
            rmse_improvement = (results[model_name]["direct"]["rmse"] - results[model_name]["finetune"]["rmse"]) / results[model_name]["direct"]["rmse"] * 100
            results[model_name]["improvement"] = {
                "rmse_percent": rmse_improvement
            }
            
            if verbose:
                if rmse_improvement > 0:
                    print(f"  {model_name} 通过迁移学习在RMSE上提升了 {rmse_improvement:.2f}%")
                else:
                    print(f"  {model_name} 通过迁移学习在RMSE上降低了 {-rmse_improvement:.2f}%")
    
    return results

def save_ml_transfer_results(results: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """
    保存传统机器学习迁移学习结果
    
    参数:
        results: 迁移学习结果字典
        output_dir: 输出目录
        
    返回:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    rows = []
    for model_name, model_results in results.items():
        pretrain = model_results.get("pretrain", {})
        finetune = model_results.get("finetune", {})
        direct = model_results.get("direct", {})
        improvement = model_results.get("improvement", {})
        
        row = {
            "model": model_name,
            "pretrain_rmse": pretrain.get("rmse", float('nan')),
            "pretrain_mae": pretrain.get("mae", float('nan')),
            "pretrain_r2": pretrain.get("r2", float('nan')),
            "pretrain_time": pretrain.get("time", float('nan')),
            
            "finetune_rmse": finetune.get("rmse", float('nan')),
            "finetune_mae": finetune.get("mae", float('nan')),
            "finetune_r2": finetune.get("r2", float('nan')),
            "finetune_time": finetune.get("time", float('nan')),
            
            "direct_rmse": direct.get("rmse", float('nan')),
            "direct_mae": direct.get("mae", float('nan')),
            "direct_r2": direct.get("r2", float('nan')),
            "direct_time": direct.get("time", float('nan')),
            
            "rmse_improvement_pct": improvement.get("rmse_percent", float('nan'))
            }
        
        rows.append(row)
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(rows)
    
    # 按RMSE改进百分比排序
    results_df = results_df.sort_values(by="rmse_improvement_pct", ascending=False)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ml_transfer_results_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    results_df.to_csv(file_path, index=False)
    
    print(f"传统机器学习迁移学习结果已保存至: {file_path}")
    return file_path

def perform_hyperparameter_tuning(X_train, y_train, tuning_type='random', n_iter=10, cv=3, verbose=1):
    """
    对主要机器学习模型执行超参数调优
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        tuning_type: 调优类型，'grid'或'random'
        n_iter: RandomizedSearchCV的迭代次数
        cv: 交叉验证折数
        verbose: 详细程度
        
    返回:
        best_models: 包含调优后最佳模型的字典
    """
    # 超参数网格定义
    param_grids = {
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': np.logspace(-4, 1, 10),
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-3, 1e-4, 1e-5]
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': np.logspace(-4, 1, 10),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'params': {
                'alpha': np.logspace(-4, 1, 10),
                'l1_ratio': np.linspace(0.1, 0.9, 9),
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-3, 1e-4, 1e-5]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': np.logspace(-3, 3, 7),
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
                'epsilon': [0.01, 0.1, 0.2]
            }
            }
            }
    
    best_models = {}
    
    # 分别对看涨期权和看跌期权价格进行训练
    for target_idx, target_name in enumerate(['call', 'put']):
        target = y_train[:, target_idx]
        print(f"\n开始对{target_name}期权价格进行超参数调优...")
        
        for model_name, model_info in param_grids.items():
            print(f"调优模型: {model_name}")
            
            if tuning_type == 'grid':
                search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv,
                    n_jobs=-1,
                    verbose=verbose,
                    scoring='neg_mean_squared_error'
                )
            else:  # random
                search = RandomizedSearchCV(
                    model_info['model'],
                    model_info['params'],
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=-1,
                    verbose=verbose,
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
            
            search.fit(X_train, target)
            
            # 保存最佳模型和参数
            best_models[f"{model_name}_{target_name}"] = {
                'model': search.best_estimator_,
                'params': search.best_params_,
                'score': -search.best_score_
            }
            
            print(f"最佳参数: {search.best_params_}")
            print(f"最佳MSE: {-search.best_score_:.6f}")
    
    return best_models

# 2) 深度学习模型实现

class DeepMLP(nn.Module):
    """
    深层多层感知机网络
    """
    def __init__(self, in_dim, hidden_layers=[64, 128, 64], out_dim=2, dropout_rate=0.2):
        super(DeepMLP, self).__init__()
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 构建中间隐藏层
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    """
    残差块实现
    """
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out

class ResidualMLP(nn.Module):
    """
    带残差连接的多层感知机
    """
    def __init__(self, in_dim, hidden_dim=64, blocks=3, out_dim=2, dropout_rate=0.2):
        super(ResidualMLP, self).__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x

class SimpleCNN(nn.Module):
    """
    简单的一维卷积神经网络
    """
    def __init__(self, in_dim, out_dim=2, channels=[16, 32], kernel_size=3, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 计算卷积后的特征维度
        self.fc = nn.Linear(channels[1], out_dim)

    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)   # -> [batch, 1, in_dim]
        
        # 第一个卷积块
        x = self.conv1(x)    # -> [batch, channels[0], in_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)    # -> [batch, channels[0], in_dim/2]
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.conv2(x)    # -> [batch, channels[1], in_dim/2]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)    # -> [batch, channels[1], 1]
        x = self.dropout2(x)
        
        # 展平并通过全连接层
        x = x.view(x.size(0), -1) # -> [batch, channels[1]]
        x = self.fc(x)       # -> [batch, out_dim]
        return x

class AdvancedCNN(nn.Module):
    """
    高级的一维卷积神经网络，包含多个卷积层和残差连接
    """
    def __init__(self, in_dim, out_dim=2, channels=[16, 32, 64, 128], kernel_size=3, dropout_rate=0.2):
        super(AdvancedCNN, self).__init__()
        
        self.channels = channels
        self.conv_blocks = nn.ModuleList()
        
        # 输入投影
        self.input_proj = nn.Conv1d(1, channels[0], kernel_size=1)
        
        # 构建卷积块
        for i in range(len(channels)-1):
            self.conv_blocks.append(self._make_conv_block(
                channels[i], 
                channels[i+1], 
                kernel_size, 
                dropout_rate
            ))
        
        # 全局池化和输出层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], out_dim)
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size, dropout_rate):
        """创建包含残差连接的卷积块"""
        block = nn.Sequential(
            # 第一个卷积层
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第二个卷积层
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
        )
        
        # 如果输入和输出通道数不同，则添加1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        
        return block
    
    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)   # -> [batch, 1, in_dim]
        x = self.input_proj(x)  # -> [batch, channels[0], in_dim]
        
        # 通过卷积块
        for i, block in enumerate(self.conv_blocks):
            identity = x
            x = block(x)
            
            # 添加残差连接
            x += self.shortcut(identity) if hasattr(self, 'shortcut') else identity
            x = F.relu(x)
            
            # 在每个卷积块后应用池化（除了最后一个）
            if i < len(self.conv_blocks) - 1:
                x = F.max_pool1d(x, 2)
        
        # 全局池化
        x = self.global_pool(x)  # -> [batch, channels[-1], 1]
        x = x.view(x.size(0), -1)  # -> [batch, channels[-1]]
        
        # 全连接层
        x = self.fc(x)  # -> [batch, out_dim]
        return x

class SimpleRNN(nn.Module):
    """
    简单的循环神经网络
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, out_dim=2, dropout_rate=0.2, bidirectional=False):
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * self.num_directions, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)  # [batch, 1, in_dim]
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        
        # 通过RNN
        out, hn = self.rnn(x, h0)  # out: [batch, 1, hidden_dim * num_directions]
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 连接前向和后向的最后一个隐藏状态
            out = torch.cat((out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]), dim=1)
        else:
            out = out[:, -1, :]  # [batch, hidden_dim]
        
        # 应用dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)  # [batch, out_dim]
        
        return out

class SimpleLSTM(nn.Module):
    """
    简单的长短期记忆网络
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, out_dim=2, dropout_rate=0.2, bidirectional=False):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * self.num_directions, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)  # [batch, 1, in_dim]
        
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        
        # 通过LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: [batch, 1, hidden_dim * num_directions]
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 连接前向和后向的最后一个隐藏状态
            out = torch.cat((out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]), dim=1)
        else:
            out = out[:, -1, :]  # [batch, hidden_dim]
        
        # 应用dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)  # [batch, out_dim]
        
        return out

class SimpleGRU(nn.Module):
    """
    简单的门控循环单元网络
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, out_dim=2, dropout_rate=0.2, bidirectional=False):
        super(SimpleGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU层
        self.gru = nn.GRU(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * self.num_directions, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)  # [batch, 1, in_dim]
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim).to(x.device)
        
        # 通过GRU
        out, hn = self.gru(x, h0)  # out: [batch, 1, hidden_dim * num_directions]
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 连接前向和后向的最后一个隐藏状态
            out = torch.cat((out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]), dim=1)
        else:
            out = out[:, -1, :]  # [batch, hidden_dim]
        
        # 应用dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)  # [batch, out_dim]
        
        return out

class SimpleTransformer(nn.Module):
    """
    简单的Transformer模型
    """
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, out_dim=2, dropout_rate=0.1):
        super(SimpleTransformer, self).__init__()
        
        # 输入投影
        self.embed = nn.Linear(in_dim, d_model)
        
        # TransformerEncoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch, in_dim]
        x = x.unsqueeze(1)           # [batch, 1, in_dim]
        x = self.embed(x)            # [batch, 1, d_model]
        x = x.permute(1, 0, 2)       # [1, batch, d_model]
        
        # 创建padding mask，这里我们没有padding，所以不需要mask
        # mask = torch.zeros(x.size(1), x.size(0)).bool().to(x.device)
        
        # 通过Transformer Encoder
        x = self.transformer_encoder(x)  # [1, batch, d_model]
        x = x.permute(1, 0, 2)           # [batch, 1, d_model]
        
        # 使用序列的最后一个位置进行预测
        x = x[:, -1, :]                  # [batch, d_model]
        x = self.dropout(x)
        x = self.fc(x)                   # [batch, out_dim]
        
        return x

class BayesianLinear(nn.Module):
    """
    贝叶斯线性层，用于贝叶斯神经网络
    """
    def __init__(self, in_features, out_features, prior_sigma_1=0.1, prior_sigma_2=0.001, prior_pi=0.5):
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重和偏置的均值参数
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # 初始化参数
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.uniform_(self.weight_rho, -5, -4)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.uniform_(self.bias_rho, -5, -4)
        
        # 先验分布参数
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # 标记是否在训练模式
        self.training = True
    
    def forward(self, x):
        # 根据训练/测试模式生成权重
        if self.training:
            # 从标准正态分布中采样随机噪声
            weight_epsilon = torch.randn(self.out_features, self.in_features).to(x.device)
            bias_epsilon = torch.randn(self.out_features).to(x.device)
            
            # 通过重参数化技巧计算权重和偏置
            weight = self.weight_mu + torch.log(1 + torch.exp(self.weight_rho)) * weight_epsilon
            bias = self.bias_mu + torch.log(1 + torch.exp(self.bias_rho)) * bias_epsilon
            
            # 保存log likelihood和KL散度的计算所需的值
            self.weight_sample = weight
            self.bias_sample = bias
        else:
            # 测试时使用权重的均值
            weight = self.weight_mu
            bias = self.bias_mu
        
        # 执行线性变换
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """计算KL散度"""
        # 对于权重
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        weight_kl = self._kl_divergence(self.weight_mu, weight_sigma,
                                        self.prior_sigma_1, self.prior_sigma_2, self.prior_pi)
        
        # 对于偏置
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        bias_kl = self._kl_divergence(self.bias_mu, bias_sigma,
                                      self.prior_sigma_1, self.prior_sigma_2, self.prior_pi)
        
        return weight_kl + bias_kl
    
    def _kl_divergence(self, mu, sigma, prior_sigma_1, prior_sigma_2, prior_pi):
        """辅助函数：计算KL散度"""
        # 计算KL散度（权重的后验分布与混合高斯先验分布之间）
        kl_div = torch.log(prior_pi / prior_sigma_1 + (1 - prior_pi) / prior_sigma_2) - 0.5
        kl_div += 0.5 * (prior_sigma_1**2 + prior_sigma_2**2) * (sigma**2 + mu**2) / (prior_sigma_1**2 * prior_sigma_2**2)
        kl_div -= 0.5 * (torch.log(sigma**2) + 1)
        
        return kl_div.sum()

class BayesianNN(nn.Module):
    """
    贝叶斯神经网络
    """
    def __init__(self, in_dim, hidden_dims=[64, 32], out_dim=2, prior_sigma_1=0.1, prior_sigma_2=0.001, prior_pi=0.5):
        super(BayesianNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 构建贝叶斯线性层
        layer_dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(layer_dims) - 1):
            self.layers.append(BayesianLinear(
                layer_dims[i], layer_dims[i+1],
                prior_sigma_1=prior_sigma_1,
                prior_sigma_2=prior_sigma_2,
                prior_pi=prior_pi
            ))
        
        # 存储层数以便后续计算KL散度
        self.num_layers = len(self.layers)
    
    def forward(self, x):
        # 前向传播
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        
        # 最后一层不使用激活函数
        x = self.layers[-1](x)
        
        return x
    
    def kl_loss(self):
        """计算整个网络的KL散度"""
        kl = 0
        for layer in self.layers:
            kl += layer.kl_loss()
        return kl

def train_dl_model(model, X_train, y_train, X_val, y_val,
                  epochs=100, lr=1e-3, batch_size=64, weight_decay=1e-4, 
                  patience=10, lr_scheduler='plateau', device='cpu', verbose=True):
    """
    训练深度学习模型
    
    参数:
        model: PyTorch模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        lr: 初始学习率
        batch_size: 批量大小
        weight_decay: L2正则化系数
        patience: 早停的耐心值
        lr_scheduler: 学习率调度器类型 ('plateau', 'cosine', None)
        device: 训练设备 ('cpu', 'cuda')
        verbose: 是否打印训练进度
    
    返回:
        训练好的模型和训练历史
    """
    # 准备优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 准备学习率调度器
    if lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//3, verbose=verbose)
    elif lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    else:
        scheduler = None
    
    # 将模型和数据转移到指定设备
    model = model.to(device)
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 存储训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse_call': [],
        'val_mse_put': [],
        'learning_rates': []
            }
    
    # 早停设置
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # 训练一个epoch
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 如果是贝叶斯神经网络，添加KL散度
            if hasattr(model, 'kl_loss'):
                kl_weight = 1.0 / len(train_loader)  # 可根据需要调整KL权重
                loss += kl_weight * model.kl_loss()
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # 计算训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            # 分别计算call和put的MSE
            val_mse_call = F.mse_loss(val_outputs[:, 0], y_val_t[:, 0]).item()
            val_mse_put = F.mse_loss(val_outputs[:, 1], y_val_t[:, 1]).item()
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存训练历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse_call'].append(val_mse_call)
        history['val_mse_put'].append(val_mse_put)
        history['learning_rates'].append(current_lr)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"早停触发，在epoch {epoch+1}/{epochs}停止训练")
                break
        
        # 打印训练进度
        if verbose and (epoch+1) % (max(1, epochs//10)) == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Val MSE Call: {val_mse_call:.6f}, "
                  f"Val MSE Put: {val_mse_put:.6f}, "
                  f"LR: {current_lr:.6f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def run_multiple_times_dl(ModelClass, X_train, y_train, X_test, y_test,
                         runs=5, device='cpu', model_params=None,
                         train_params=None, verbose=True):
    """
    多次训练深度学习模型，评估性能稳定性
    
    参数:
        ModelClass: 模型类
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        runs: 运行次数
        device: 训练设备
        model_params: 模型参数字典
        train_params: 训练参数字典
        verbose: 是否打印详细信息
    
    返回:
        summary: 性能摘要
        results: 详细结果
    """
    # 默认模型参数
    if model_params is None:
        model_params = {}
    
    # 默认训练参数
    if train_params is None:
        train_params = {
            'epochs': 100,
            'lr': 1e-3,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'patience': 10,
            'lr_scheduler': 'plateau'
            }
    
    # 初始化结果字典
    results = {
        'time': [],
        'mse_call': [],
        'mse_put': [],
        'mae_call': [],
        'mae_put': [],
        'r2_call': [],
        'r2_put': [],
        'histories': []
            }
    
    if verbose:
        print(f"开始训练 {ModelClass.__name__} 模型，运行 {runs} 次...")
    
    for run in range(runs):
        if verbose:
            print(f"\n--- 运行 {run+1}/{runs} ---")
        
        start_t = time.time()

        # 创建模型实例
        model = ModelClass(in_dim=X_train.shape[1], out_dim=2, **model_params)
        
        # 训练模型
        trained_model, history = train_dl_model(
            model, X_train, y_train, X_test, y_test,
            device=device, verbose=verbose,
            **train_params
        )
        
        # 记录训练时间
        elapsed = time.time() - start_t
        results['time'].append(elapsed)
        results['histories'].append(history)

        # 评估模型
        trained_model.eval()
        X_test_t = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            pred = trained_model(X_test_t).cpu().numpy()
        
        # 计算多种评估指标
        mse_call = mean_squared_error(y_test[:,0], pred[:,0])
        mse_put = mean_squared_error(y_test[:,1], pred[:,1])
        mae_call = mean_absolute_error(y_test[:,0], pred[:,0])
        mae_put = mean_absolute_error(y_test[:,1], pred[:,1])
        r2_call = r2_score(y_test[:,0], pred[:,0])
        r2_put = r2_score(y_test[:,1], pred[:,1])
        
        # 保存评估结果
        results['mse_call'].append(mse_call)
        results['mse_put'].append(mse_put)
        results['mae_call'].append(mae_call)
        results['mae_put'].append(mae_put)
        results['r2_call'].append(r2_call)
        results['r2_put'].append(r2_put)
        
        if verbose:
            print(f"运行 {run+1} 结果:")
            print(f"  时间: {elapsed:.2f}秒")
            print(f"  MSE Call: {mse_call:.6f}")
            print(f"  MSE Put: {mse_put:.6f}")
            print(f"  MAE Call: {mae_call:.6f}")
            print(f"  MAE Put: {mae_put:.6f}")
            print(f"  R² Call: {r2_call:.6f}")
            print(f"  R² Put: {r2_put:.6f}")
    
    # 计算摘要统计
    t_arr = np.array(results['time'])
    c_mse_arr = np.array(results['mse_call'])
    p_mse_arr = np.array(results['mse_put'])
    c_mae_arr = np.array(results['mae_call'])
    p_mae_arr = np.array(results['mae_put'])
    c_r2_arr = np.array(results['r2_call'])
    p_r2_arr = np.array(results['r2_put'])
    
    summary = {
                'time_mean': t_arr.mean(), 
                'time_std': t_arr.std(),
                'mse_call_mean': c_mse_arr.mean(), 
                'mse_call_std': c_mse_arr.std(),
                'mse_put_mean': p_mse_arr.mean(), 
                'mse_put_std': p_mse_arr.std(),
                'mae_call_mean': c_mae_arr.mean(), 
                'mae_call_std': c_mae_arr.std(),
                'mae_put_mean': p_mae_arr.mean(), 
                'mae_put_std': p_mae_arr.std(),
                'r2_call_mean': c_r2_arr.mean(), 
                'r2_call_std': c_r2_arr.std(),
                'r2_put_mean': p_r2_arr.mean(), 
                'r2_put_std': p_r2_arr.std()
            }
    
    if verbose:
        print("\n模型性能摘要:")
        print(f"时间: {summary['time_mean']:.2f} ± {summary['time_std']:.2f}秒")
        print(f"MSE Call: {summary['mse_call_mean']:.6f} ± {summary['mse_call_std']:.6f}")
        print(f"MSE Put: {summary['mse_put_mean']:.6f} ± {summary['mse_put_std']:.6f}")
        print(f"R² Call: {summary['r2_call_mean']:.6f} ± {summary['r2_call_std']:.6f}")
        print(f"R² Put: {summary['r2_put_mean']:.6f} ± {summary['r2_put_std']:.6f}")
    
    return summary, results

def evaluate_models_performance(X_train, y_train, X_test, y_test, device='cpu', runs=3, verbose=True):
    """
    评估所有模型的性能
    
    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        device: 训练设备
        runs: 每个模型运行次数
        verbose: 是否打印详细信息
        
    返回:
        all_models_summary: 所有模型的性能摘要
    """
    # 定义要评估的模型
    models_to_evaluate = {
        'DeepMLP': {
            'class': DeepMLP,
            'params': {'hidden_layers': [64, 128, 64], 'dropout_rate': 0.2}
        },
        'ResidualMLP': {
            'class': ResidualMLP,
            'params': {'hidden_dim': 64, 'blocks': 3, 'dropout_rate': 0.2}
        },
        'SimpleCNN': {
            'class': SimpleCNN,
            'params': {'channels': [16, 32], 'kernel_size': 3, 'dropout_rate': 0.2}
        },
        'AdvancedCNN': {
            'class': AdvancedCNN,
            'params': {'channels': [16, 32, 64, 128], 'kernel_size': 3, 'dropout_rate': 0.2}
        },
        'SimpleRNN': {
            'class': SimpleRNN,
            'params': {'hidden_dim': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'bidirectional': True}
        },
        'SimpleLSTM': {
            'class': SimpleLSTM,
            'params': {'hidden_dim': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'bidirectional': True}
        },
        'SimpleGRU': {
            'class': SimpleGRU,
            'params': {'hidden_dim': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'bidirectional': True}
        },
        'SimpleTransformer': {
            'class': SimpleTransformer,
            'params': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout_rate': 0.1}
        },
        'BayesianNN': {
            'class': BayesianNN,
            'params': {'hidden_dims': [64, 32], 'prior_sigma_1': 0.1, 'prior_sigma_2': 0.001, 'prior_pi': 0.5}
            }
            }
    
    # 定义训练参数
    train_params = {
        'epochs': 100,
        'lr': 1e-3,
        'batch_size': 64,
        'weight_decay': 1e-4,
        'patience': 10,
        'lr_scheduler': 'plateau'
            }
    
    # 存储所有模型的结果
    all_models_summary = {}
    
    for model_name, model_info in models_to_evaluate.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"评估模型: {model_name}")
            print(f"{'='*50}")
        
        # 运行多次训练和评估
        summary, results = run_multiple_times_dl(
            model_info['class'],
            X_train, y_train, X_test, y_test,
            runs=runs,
            device=device,
            model_params=model_info['params'],
            train_params=train_params,
            verbose=verbose
        )
        
        # 保存结果
        all_models_summary[model_name] = summary
    
    # 打印比较结果
    if verbose:
        print("\n\n模型性能比较 (按看涨期权MSE排序):")
        sorted_models = sorted(all_models_summary.items(), key=lambda x: x[1]['mse_call_mean'])
        
        print(f"\n{'模型名称':<20} {'MSE Call':<15} {'MSE Put':<15} {'R² Call':<15} {'R² Put':<15} {'时间(秒)':<15}")
        print('-' * 85)
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} "
                  f"{metrics['mse_call_mean']:.6f} ± {metrics['mse_call_std']:.6f}   "
                  f"{metrics['mse_put_mean']:.6f} ± {metrics['mse_put_std']:.6f}   "
                  f"{metrics['r2_call_mean']:.4f} ± {metrics['r2_call_std']:.4f}   "
                  f"{metrics['r2_put_mean']:.4f} ± {metrics['r2_put_std']:.4f}   "
                  f"{metrics['time_mean']:.1f} ± {metrics['time_std']:.1f}")
    
    return all_models_summary

def visualize_learning_curves(histories, model_name=None, save_path=None):
    """
    可视化学习曲线
    
    参数:
        histories: 训练历史记录列表
        model_name: 模型名称
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制训练和验证损失
    plt.subplot(2, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history['train_loss'], label=f'Train {i+1}')
        plt.plot(history['val_loss'], label=f'Val {i+1}', linestyle='--')
    plt.title(f'{model_name} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 绘制看涨期权和看跌期权的MSE
    plt.subplot(2, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history['val_mse_call'], label=f'Call {i+1}')
        plt.plot(history['val_mse_put'], label=f'Put {i+1}', linestyle='--')
    plt.title(f'{model_name} - 验证MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率变化
    plt.subplot(2, 2, 3)
    for i, history in enumerate(histories):
        plt.plot(history['learning_rates'], label=f'Run {i+1}')
    plt.title(f'{model_name} - 学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 保存图像
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 示例数据生成
    from svjj_model import SVCJParams, generate_random_option_data
    import numpy as np
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
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
    num_samples = 10000
    data = generate_random_option_data(params, num_samples=num_samples)
    
    # 准备特征和标签
    features = ['S', 'V', 'K', 'tau', 'moneyness']
    X = data[features].values
    y = data[['call_price', 'put_price']].values
    
    # 分割训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 评估传统机器学习模型
    print("\n评估传统机器学习模型:")
    ml_summary, ml_results = train_and_evaluate_ml_models(
        X_train, y_train, X_test, y_test, n_runs=3, verbose=True
    )
    
    # 评估深度学习模型
    print("\n评估深度学习模型:")
    # 为了示例，只评估几个模型
    dl_models = {
        'DeepMLP': DeepMLP,
        'SimpleLSTM': SimpleLSTM,
        'SimpleTransformer': SimpleTransformer
            }
    
    dl_results = {}
    for name, model_class in dl_models.items():
        print(f"\n评估 {name}:")
        summary, results = run_multiple_times_dl(
            model_class, 
            X_train, y_train, X_test, y_test,
            runs=2, 
            device=device,
            verbose=True
        )
        dl_results[name] = (summary, results)
        
        # 可视化学习曲线
        visualize_learning_curves(
            results['histories'], 
            model_name=name,
            save_path=f"{name}_learning_curves.png"
        )

# 新增: 深度学习模型的迁移学习函数
def train_and_transfer_dl_model(
    ModelClass: nn.Module, 
    source_data: Dict[str, torch.Tensor], 
    target_data: Dict[str, torch.Tensor], 
    model_params: Dict[str, Any] = None, 
    transfer_params: Dict[str, Any] = None,
    device: str = 'cpu', 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    训练深度学习模型并将其迁移到目标域
    
    参数:
        ModelClass: 模型类
        source_data: 源域数据字典，包含'X_train'、'y_train'等键
        target_data: 目标域数据字典，包含'X_train'、'y_train'等键
        model_params: 模型初始化参数
        transfer_params: 迁移学习参数
        device: 训练设备
        verbose: 是否打印详细信息
        
    返回:
        results: 包含预训练、迁移和直接训练结果的字典
    """
    if model_params is None:
        model_params = {}
    
    if transfer_params is None:
        transfer_params = config.DL_MODELS_PARAMS["common"]["transfer"]
    
    # 提取数据
    X_source_train = torch.tensor(source_data["X_train"], dtype=torch.float32).to(device)
    y_source_train = torch.tensor(source_data["y_train"], dtype=torch.float32).to(device)
    X_source_test = torch.tensor(source_data["X_test"], dtype=torch.float32).to(device)
    y_source_test = torch.tensor(source_data["y_test"], dtype=torch.float32).to(device)
    
    X_target_train = torch.tensor(target_data["X_train"], dtype=torch.float32).to(device)
    y_target_train = torch.tensor(target_data["y_train"], dtype=torch.float32).to(device)
    X_target_test = torch.tensor(target_data["X_test"], dtype=torch.float32).to(device)
    y_target_test = torch.tensor(target_data["y_test"], dtype=torch.float32).to(device)
    
    # 初始化结果字典
    results = {
        "pretrain": {},  # 预训练模型在源域上的性能
        "finetune": {},  # 迁移学习后在目标域上的性能
        "direct": {},    # 直接在目标域上训练的性能
        "model_info": {} # 模型信息
            }
    
    if verbose:
        print(f"开始 {ModelClass.__name__} 的迁移学习...")
    
    # 1. 在源域上预训练模型
    start_time = time.time()
    
    # 创建源域模型
    in_dim = X_source_train.shape[1]
    out_dim = y_source_train.shape[1] if len(y_source_train.shape) > 1 else 1
    
    # 初始化模型参数
    init_params = {"in_dim": in_dim, "out_dim": out_dim}
    init_params.update(model_params)
    
    source_model = ModelClass(**init_params).to(device)
    
    # 训练源域模型
    source_train_params = config.DL_MODELS_PARAMS["common"].copy()
    source_train_params.update({
        "epochs": config.TRANSFER_LEARNING_CONFIG["epochs_pretrain"],
        "device": device,
        "verbose": verbose
    })
    
    # 训练源域模型
    source_train_history = train_dl_model(
        source_model, 
        X_source_train, y_source_train,
        X_source_test, y_source_test,
        **source_train_params
    )
    
    # 在源域测试集上评估
    source_model.eval()
    with torch.no_grad():
        source_preds = source_model(X_source_test).cpu().numpy()
    
    source_metrics = calculate_metrics(y_source_test.cpu().numpy(), source_preds)
    pretrain_time = time.time() - start_time
    
    results["pretrain"] = {
        "rmse": source_metrics["rmse"],
        "mae": source_metrics["mae"],
        "r2": source_metrics["r2"],
        "time": pretrain_time,
        "history": source_train_history
            }
    
    if verbose:
        print(f"预训练完成，源域RMSE: {source_metrics['rmse']:.6f}, R2: {source_metrics['r2']:.6f}")
    
    # 2. 迁移学习到目标域
    start_time = time.time()
    
    # 复制预训练模型
    transfer_model = ModelClass(**init_params).to(device)
    transfer_model.load_state_dict(source_model.state_dict())
    
    # 冻结部分层
    freeze_ratio = transfer_params["freeze_ratio"]
    if freeze_ratio > 0:
        freeze_layers(transfer_model, freeze_ratio)
    
    # 设置迁移学习参数
    transfer_train_params = config.DL_MODELS_PARAMS["common"].copy()
    transfer_train_params.update({
        "epochs": config.TRANSFER_LEARNING_CONFIG["epochs_finetune"],
        "lr": transfer_params["finetune_lr"],
        "weight_decay": transfer_params["finetune_wd"],
        "device": device,
        "verbose": verbose
    })
    
    # 训练迁移模型
    transfer_train_history = train_dl_model(
        transfer_model, 
        X_target_train, y_target_train,
        X_target_test, y_target_test,
        **transfer_train_params
    )
    
    # 在目标域测试集上评估
    transfer_model.eval()
    with torch.no_grad():
        transfer_preds = transfer_model(X_target_test).cpu().numpy()
    
    transfer_metrics = calculate_metrics(y_target_test.cpu().numpy(), transfer_preds)
    transfer_time = time.time() - start_time
    
    results["finetune"] = {
        "rmse": transfer_metrics["rmse"],
        "mae": transfer_metrics["mae"],
        "r2": transfer_metrics["r2"],
        "time": transfer_time,
        "history": transfer_train_history
            }
    
    if verbose:
        print(f"迁移学习完成，目标域RMSE: {transfer_metrics['rmse']:.6f}, R2: {transfer_metrics['r2']:.6f}")
    
    # 3. 直接在目标域上训练模型作为基线
    start_time = time.time()
    
    # 创建目标域模型
    in_dim = X_target_train.shape[1]
    out_dim = y_target_train.shape[1] if len(y_target_train.shape) > 1 else 1
    
    # 初始化模型参数
    init_params = {"in_dim": in_dim, "out_dim": out_dim}
    init_params.update(model_params)
    
    direct_model = ModelClass(**init_params).to(device)
    
    # 训练目标域模型
    direct_train_params = config.DL_MODELS_PARAMS["common"].copy()
    direct_train_params.update({
        "epochs": config.TRANSFER_LEARNING_CONFIG["epochs_finetune"],
        "device": device,
        "verbose": verbose
    })
    
    # 训练直接模型
    direct_train_history = train_dl_model(
        direct_model, 
        X_target_train, y_target_train,
        X_target_test, y_target_test,
        **direct_train_params
    )
    
    # 在目标域测试集上评估
    direct_model.eval()
    with torch.no_grad():
        direct_preds = direct_model(X_target_test).cpu().numpy()
    
    direct_metrics = calculate_metrics(y_target_test.cpu().numpy(), direct_preds)
    direct_time = time.time() - start_time
    
    results["direct"] = {
        "rmse": direct_metrics["rmse"],
        "mae": direct_metrics["mae"],
        "r2": direct_metrics["r2"],
        "time": direct_time,
        "history": direct_train_history
            }
    
    if verbose:
        print(f"直接训练完成，目标域RMSE: {direct_metrics['rmse']:.6f}, R2: {direct_metrics['r2']:.6f}")
    
    # 计算迁移学习相对于直接训练的改进
    rmse_improvement = (results["direct"]["rmse"] - results["finetune"]["rmse"]) / results["direct"]["rmse"] * 100
    results["improvement"] = {
        "rmse_percent": rmse_improvement
            }
    
    if verbose:
        if rmse_improvement > 0:
            print(f"{ModelClass.__name__} 通过迁移学习在RMSE上提升了 {rmse_improvement:.2f}%")
        else:
            print(f"{ModelClass.__name__} 通过迁移学习在RMSE上降低了 {-rmse_improvement:.2f}%")
    
    # 保存模型信息
    results["model_info"] = {
        "class": ModelClass.__name__,
        "params": init_params
            }
    
    return results


def freeze_layers(model: nn.Module, freeze_ratio: float = 0.5) -> None:
    """
    按照比例冻结模型前几层
    
    参数:
        model: PyTorch模型
        freeze_ratio: 要冻结的层比例，0.5表示冻结前一半层
    """
    # 获取模型的所有参数
    all_params = list(model.named_parameters())
    num_layers = len(all_params)
    
    # 计算要冻结的层数
    num_freeze = int(num_layers * freeze_ratio)
    
    # 冻结前num_freeze层
    for i, (name, param) in enumerate(all_params):
        if i < num_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # 打印冻结信息
    frozen_params = sum(p.numel() for n, p in all_params[:num_freeze])
    total_params = sum(p.numel() for n, p in all_params)
    print(f"冻结了{num_freeze}/{num_layers}层，共{frozen_params}/{total_params}参数 ({frozen_params/total_params*100:.1f}%)")


def run_multiple_dl_transfer(
    ModelClass: nn.Module,
    source_data: Dict[str, np.ndarray],
    target_data: Dict[str, np.ndarray],
    runs: int = 3,
    model_params: Dict[str, Any] = None,
    transfer_params: Dict[str, Any] = None,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    多次运行深度学习模型的迁移学习，并返回平均结果
    
    参数:
        ModelClass: 模型类
        source_data: 源域数据字典
        target_data: 目标域数据字典
        runs: 运行次数
        model_params: 模型初始化参数
        transfer_params: 迁移学习参数
        device: 训练设备
        verbose: 是否打印详细信息
        
    返回:
        平均结果字典
    """
    # 存储每次运行的结果
    all_results = []
    
    for run in range(runs):
        if verbose:
            print(f"\n=== 运行 {run+1}/{runs} ===")
        
        # 运行一次迁移学习
        result = train_and_transfer_dl_model(
            ModelClass,
            source_data,
            target_data,
            model_params,
            transfer_params,
            device,
            verbose
        )
        
        all_results.append(result)
    
    # 计算平均结果
    avg_result = {
        "pretrain": {},
        "finetune": {},
        "direct": {},
        "improvement": {},
        "model_info": all_results[0]["model_info"]  # 模型信息相同，直接使用第一次运行的
            }
    
    # 计算每个指标的平均值
    for phase in ["pretrain", "finetune", "direct"]:
        for metric in ["rmse", "mae", "r2", "time"]:
            values = [r[phase][metric] for r in all_results]
            avg_result[phase][metric] = np.mean(values)
            avg_result[phase][f"{metric}_std"] = np.std(values)
    
    # 计算改进的平均值
    improvement_values = [r["improvement"]["rmse_percent"] for r in all_results]
    avg_result["improvement"]["rmse_percent"] = np.mean(improvement_values)
    avg_result["improvement"]["rmse_percent_std"] = np.std(improvement_values)
    
    # 保存所有运行的结果
    avg_result["all_runs"] = all_results
    
    return avg_result


def evaluate_dl_transfer_performance(
    source_data: Dict[str, np.ndarray],
    target_data: Dict[str, np.ndarray],
    models_to_use: List[str] = None,
    runs: int = 3,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    评估多种深度学习模型的迁移学习性能
    
    参数:
        source_data: 源域数据字典
        target_data: 目标域数据字典
        models_to_use: 要使用的模型列表，如果为None，则使用所有模型
        runs: 每个模型运行次数
        device: 训练设备
        verbose: 是否打印详细信息
        
    返回:
        包含每个模型迁移学习结果的字典
    """
    # 定义要评估的模型类和参数
    model_classes = {
        "DeepMLP": (DeepMLP, config.DL_MODELS_PARAMS["DeepMLP"]),
        "ResidualMLP": (ResidualMLP, config.DL_MODELS_PARAMS["ResidualMLP"]),
        "SimpleCNN": (SimpleCNN, config.DL_MODELS_PARAMS["SimpleCNN"]),
        "AdvancedCNN": (AdvancedCNN, config.DL_MODELS_PARAMS["AdvancedCNN"]),
        "SimpleRNN": (SimpleRNN, config.DL_MODELS_PARAMS["SimpleRNN"]),
        "SimpleLSTM": (SimpleLSTM, config.DL_MODELS_PARAMS["SimpleLSTM"]),
        "SimpleGRU": (SimpleGRU, config.DL_MODELS_PARAMS["SimpleGRU"]),
        "SimpleTransformer": (SimpleTransformer, config.DL_MODELS_PARAMS["SimpleTransformer"]),
        "BayesianNN": (BayesianNN, config.DL_MODELS_PARAMS["BayesianNN"])
            }
    
    # 如果指定了要使用的模型，则只使用这些模型
    if models_to_use:
        model_classes = {name: model_classes[name] for name in models_to_use if name in model_classes}
    
    # 结果字典
    results = {}
    
    for model_name, (ModelClass, model_params) in model_classes.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"评估模型: {model_name}")
            print(f"{'='*50}")
        
        # 运行多次迁移学习
        model_result = run_multiple_dl_transfer(
            ModelClass,
            source_data,
            target_data,
            runs,
            model_params,
            None,  # 使用默认迁移参数
            device,
            verbose
        )
        
        # 存储结果
        results[model_name] = model_result
        
        # 打印摘要
        if verbose:
            rmse_improvement = model_result["improvement"]["rmse_percent"]
            if rmse_improvement > 0:
                print(f"\n{model_name} 通过迁移学习平均在RMSE上提升了 {rmse_improvement:.2f}%")
            else:
                print(f"\n{model_name} 通过迁移学习平均在RMSE上降低了 {-rmse_improvement:.2f}%")
    
    return results


def save_dl_transfer_results(results: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """
    保存深度学习迁移学习结果
    
    参数:
        results: 迁移学习结果字典
        output_dir: 输出目录
        
    返回:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    rows = []
    for model_name, model_results in results.items():
        pretrain = model_results.get("pretrain", {})
        finetune = model_results.get("finetune", {})
        direct = model_results.get("direct", {})
        improvement = model_results.get("improvement", {})
        
        row = {
            "model": model_name,
            "pretrain_rmse": pretrain.get("rmse", float('nan')),
            "pretrain_rmse_std": pretrain.get("rmse_std", float('nan')),
            "pretrain_mae": pretrain.get("mae", float('nan')),
            "pretrain_mae_std": pretrain.get("mae_std", float('nan')),
            "pretrain_r2": pretrain.get("r2", float('nan')),
            "pretrain_r2_std": pretrain.get("r2_std", float('nan')),
            "pretrain_time": pretrain.get("time", float('nan')),
            
            "finetune_rmse": finetune.get("rmse", float('nan')),
            "finetune_rmse_std": finetune.get("rmse_std", float('nan')),
            "finetune_mae": finetune.get("mae", float('nan')),
            "finetune_mae_std": finetune.get("mae_std", float('nan')),
            "finetune_r2": finetune.get("r2", float('nan')),
            "finetune_r2_std": finetune.get("r2_std", float('nan')),
            "finetune_time": finetune.get("time", float('nan')),
            
            "direct_rmse": direct.get("rmse", float('nan')),
            "direct_rmse_std": direct.get("rmse_std", float('nan')),
            "direct_mae": direct.get("mae", float('nan')),
            "direct_mae_std": direct.get("mae_std", float('nan')),
            "direct_r2": direct.get("r2", float('nan')),
            "direct_r2_std": direct.get("r2_std", float('nan')),
            "direct_time": direct.get("time", float('nan')),
            
            "rmse_improvement_pct": improvement.get("rmse_percent", float('nan')),
            "rmse_improvement_pct_std": improvement.get("rmse_percent_std", float('nan'))
            }
        
        rows.append(row)
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(rows)
    
    # 按RMSE改进百分比排序
    results_df = results_df.sort_values(by="rmse_improvement_pct", ascending=False)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"dl_transfer_results_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    results_df.to_csv(file_path, index=False)
    
    print(f"深度学习迁移学习结果已保存至: {file_path}")
    return file_path
