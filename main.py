#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rainbow SVJJ 项目主入口

此脚本提供了项目的主要入口点，包括:
1. 从config.py读取配置
2. 生成模拟数据或加载实际数据
3. 预处理数据
4. 训练和评估各种模型
5. 可视化结果
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import json
import pickle

# 导入项目模块
import config
from svjj_model import SVCJParams, generate_random_option_data
from ml_dl_models_compare import train_and_evaluate_ml_models, evaluate_models_performance
from rl_compare import run_multiple_times_rl
from rainbow_model import train_rainbow
from utils import (
    get_features_and_labels, 
    print_all_comparisons,
    visualize_model_comparisons,
    create_metrics_dataframe,
    evaluate_predictions,
    save_all_metrics
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Rainbow SVJJ 项目")
    
    # 基本参数
    parser.add_argument("--mode", choices=["train", "eval", "compare"], default="train",
                       help="运行模式: train=训练模型, eval=评估模型, compare=比较模型")
    parser.add_argument("--model_type", choices=["ml", "dl", "rl", "rainbow", "all"], default="all",
                       help="模型类型: ml=传统机器学习, dl=深度学习, rl=强化学习, rainbow=Rainbow DQN, all=全部")
    parser.add_argument("--data_source", choices=["simulated", "real", "both"], default="simulated",
                       help="数据来源: simulated=模拟数据, real=真实数据, both=两者")
    
    # 数据生成参数
    parser.add_argument("--samples", type=int, default=1000000,
                       help="生成的样本数量")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="测试集比例")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    parser.add_argument("--save_models", action="store_true",
                       help="是否保存训练好的模型")
    parser.add_argument("--verbose", action="store_true",
                       help="是否显示详细输出")
    
    # Rainbow DQN特定参数
    parser.add_argument("--rainbow_steps", type=int, default=None,
                       help="Rainbow DQN训练步数，若未指定则使用config中的设置")
    
    return parser.parse_args()


def generate_data(args):
    """生成或加载数据"""
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
    svjj_data = generate_random_option_data(params, num_samples=args.samples)
    
    # 保存原始数据
    os.makedirs(args.output_dir, exist_ok=True)
    svjj_data.to_csv(os.path.join(args.output_dir, "simulated_data.csv"), index=False)
    
    return svjj_data


def preprocess_data(data, test_size=0.2):
    """预处理数据"""
    print("预处理数据...")
    
    # 提取特征和标签
    feature_cols = config.DATA_CONFIG["features"]
    label_cols = config.DATA_CONFIG["labels"]
    
    X, y = get_features_and_labels(data, feature_cols, label_cols)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 保存scaler供后续使用
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, X_test, y_train, y_test


def train_ml_models(X_train, y_train, X_test, y_test, args):
    """训练传统机器学习模型"""
    print("\n=== 训练传统机器学习模型 ===")
    
    # 获取运行次数
    n_runs = config.ML_MODELS_PARAMS["n_runs"]
    verbose = args.verbose
    
    # 训练和评估模型
    start_time = time.time()
    ml_summary, ml_results = train_and_evaluate_ml_models(
        X_train, y_train, X_test, y_test, n_runs=n_runs, verbose=verbose
    )
    training_time = time.time() - start_time
    
    # 保存结果
    ml_results_df = pd.DataFrame()
    for model_name, metrics in ml_summary.items():
        model_df = pd.DataFrame([metrics])
        model_df["model"] = model_name
        ml_results_df = pd.concat([ml_results_df, model_df], ignore_index=True)
    
    ml_results_df.to_csv(os.path.join(args.output_dir, "ml_results.csv"), index=False)
    
    print(f"传统机器学习模型训练完成，耗时: {training_time:.2f}秒")
    return ml_summary, ml_results


def train_dl_models(X_train, y_train, X_test, y_test, args):
    """训练深度学习模型"""
    print("\n=== 训练深度学习模型 ===")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取运行参数
    runs = config.DL_MODELS_PARAMS["common"]["runs"]
    verbose = args.verbose
    
    # 训练和评估模型
    start_time = time.time()
    dl_results = evaluate_models_performance(
        X_train, y_train, X_test, y_test, device=device, runs=runs, verbose=verbose
    )
    training_time = time.time() - start_time
    
    # 提取摘要信息
    dl_summary = {}
    for model_name, result in dl_results.items():
        if "summary" in result:
            dl_summary[model_name] = result["summary"]
    
    # 保存结果
    with open(os.path.join(args.output_dir, "dl_results.json"), "w") as f:
        json.dump(dl_results, f, default=lambda o: float(o) if isinstance(o, np.float32) else o)
    
    print(f"深度学习模型训练完成，耗时: {training_time:.2f}秒")
    return dl_summary, dl_results


def train_rl_models(X_train, y_train, X_test, y_test, args):
    """训练强化学习模型"""
    print("\n=== 训练强化学习模型 ===")
    
    # 使用配置文件中的参数
    model_types = [m for m in ["dqn", "ppo", "actor_critic"] 
                   if config.RL_COMPARE_CONFIG[m]["enabled"]]
    runs = config.RL_COMPARE_CONFIG["training"]["runs"]
    timesteps = config.RL_COMPARE_CONFIG["training"]["timesteps"]
    max_train_time = config.RL_COMPARE_CONFIG["training"]["max_train_time"]
    
    # 训练和评估模型
    start_time = time.time()
    rl_results = run_multiple_times_rl(
        None, X_train, y_train[:, 0], y_train[:, 1],
        model_types=model_types,
        runs=runs,
        timesteps=timesteps,
        max_train_time=max_train_time
    )
    training_time = time.time() - start_time
    
    # 提取摘要信息
    rl_summary = {}
    for model_name, result in rl_results.items():
        if "summary" in result:
            rl_summary[model_name] = result["summary"]
    
    # 保存结果
    with open(os.path.join(args.output_dir, "rl_results.json"), "w") as f:
        json.dump(rl_results, f, default=lambda o: float(o) if isinstance(o, np.float32) else o)
    
    print(f"强化学习模型训练完成，耗时: {training_time:.2f}秒")
    return rl_summary, rl_results


def train_rainbow_dqn(X_train, y_train, X_test, y_test, args):
    """训练Rainbow DQN模型"""
    print("\n=== 训练Rainbow DQN模型 ===")
    
    # 创建Rainbow配置
    rainbow_config = config.RAINBOW_CONFIG.copy()
    
    # 如果命令行指定了训练步数，则覆盖配置
    if args.rainbow_steps is not None:
        rainbow_config["T_max"] = args.rainbow_steps
    
    # 转换配置为Object类型(以便与Agent兼容)
    class RainbowArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    rainbow_args = RainbowArgs(**rainbow_config)
    
    # 创建结果目录
    os.makedirs(os.path.join(args.output_dir, "rainbow"), exist_ok=True)
    rainbow_args.id = os.path.join(args.output_dir, "rainbow")
    
    # 训练模型
    start_time = time.time()
    agent, metrics = train_rainbow(rainbow_args, X_train, y_train, verbose=args.verbose)
    training_time = time.time() - start_time
    
    # 评估模型
    agent.eval()
    
    # 保存结果
    with open(os.path.join(args.output_dir, "rainbow_metrics.json"), "w") as f:
        json.dump(metrics, f, default=lambda o: float(o) if isinstance(o, np.float32) else o)
    
    # 保存模型
    if args.save_models:
        agent.save(os.path.join(args.output_dir, "rainbow"))
    
    print(f"Rainbow DQN模型训练完成，耗时: {training_time:.2f}秒")
    print(f"最佳平均奖励: {metrics['best_avg_reward']}")
    
    return agent, metrics


def compare_models(ml_summary, dl_summary, rl_summary, rainbow_metrics, args):
    """比较所有模型的性能"""
    print("\n=== 比较所有模型性能 ===")
    
    # 打印比较结果
    print_all_comparisons(ml_summary, dl_summary, rl_summary, args.output_dir)
    
    # 创建比较图表
    visualize_model_comparisons(
        ml_summary, dl_summary, rl_summary,
        save_path=os.path.join(args.output_dir, "model_comparison.png")
    )
    
    # 创建综合指标数据框
    metrics_df = create_metrics_dataframe(ml_summary, dl_summary, rl_summary)
    metrics_df.to_csv(os.path.join(args.output_dir, "all_models_metrics.csv"), index=False)
    
    # 保存所有指标
    save_all_metrics(args.output_dir)
    
    print("模型比较完成，结果已保存")


def main(args):
    """主函数"""
    print("\n===== Rainbow SVJJ 项目 =====")
    print(f"运行模式: {args.mode}")
    print(f"模型类型: {args.model_type}")
    print(f"数据来源: {args.data_source}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成或加载数据
    if args.data_source in ["simulated", "both"]:
        data = generate_data(args)
    else:
        # 如果有真实数据，从这里加载
        data_path = config.DATA_PATH
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            print(f"找不到真实数据文件: {data_path}，改为生成模拟数据")
            data = generate_data(args)
    
    # 预处理数据
    X_train, X_test, y_train, y_test = preprocess_data(data, args.test_size)
    
    # 训练模式
    if args.mode == "train":
        ml_summary, ml_results = None, None
        dl_summary, dl_results = None, None
        rl_summary, rl_results = None, None
        rainbow_agent, rainbow_metrics = None, None
        
        # 训练选择的模型类型
        if args.model_type in ["ml", "all"]:
            ml_summary, ml_results = train_ml_models(X_train, y_train, X_test, y_test, args)
        
        if args.model_type in ["dl", "all"]:
            dl_summary, dl_results = train_dl_models(X_train, y_train, X_test, y_test, args)
        
        if args.model_type in ["rl", "all"]:
            rl_summary, rl_results = train_rl_models(X_train, y_train, X_test, y_test, args)
        
        if args.model_type in ["rainbow", "all"]:
            rainbow_agent, rainbow_metrics = train_rainbow_dqn(X_train, y_train, X_test, y_test, args)
        
        # 比较所有已训练的模型
        if args.model_type == "all":
            compare_models(ml_summary, dl_summary, rl_summary, rainbow_metrics, args)
    
    # 评估模式
    elif args.mode == "eval":
        # 从保存的模型文件加载模型进行评估
        print("评估模式尚未实现")
    
    # 比较模式
    elif args.mode == "compare":
        # 从保存的结果文件加载结果进行比较
        print("比较模式尚未实现")
    
    print("\n===== 运行完成 =====")


if __name__ == "__main__":
    args = parse_args()
    main(args) 