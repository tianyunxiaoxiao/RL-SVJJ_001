#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目工具模块
包含用于数据处理、评估和可视化的通用函数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error
)

import config


def get_features_and_labels(data: pd.DataFrame, feature_cols: List[str], label_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从DataFrame中提取特征和标签
    
    参数:
        data: 包含特征和标签的DataFrame
        feature_cols: 特征列名列表
        label_cols: 标签列名列表
        
    返回:
        特征数组和标签数组
    """
    X = data[feature_cols].values
    y = data[label_cols].values
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     metrics: List[str] = None) -> Dict[str, float]:
    """
    计算回归指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        metrics: 要计算的指标列表
        
    返回:
        包含各指标值的字典
    """
    if metrics is None:
        metrics = config.METRICS_CONFIG["standard_metrics"]
    
    results = {}
    
    # 标准指标
    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_true, y_pred)
    if "r2" in metrics:
        results["r2"] = r2_score(y_true, y_pred)
    
    # 高级指标
    if "median_ae" in metrics:
        results["median_ae"] = median_absolute_error(y_true, y_pred)
    if "max_error" in metrics:
        results["max_error"] = max_error(y_true, y_pred)
    if "explained_variance" in metrics:
        results["explained_variance"] = explained_variance_score(y_true, y_pred)
    
    # 相对指标
    if "mape" in metrics:
        # 避免除以零
        mask = y_true != 0
        results["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    if "rmae" in metrics:
        # 相对MAE
        mask = y_true != 0
        results["rmae"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    if "rmse_pct" in metrics:
        # RMSE百分比
        mask = y_true != 0
        results["rmse_pct"] = np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100
    
    return results


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        is_option_data: bool = True,
                        moneyness: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    全面评估预测结果，包括标准指标和期权特定指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        is_option_data: 是否为期权数据
        moneyness: 期权虚实值数组
        
    返回:
        包含各指标值的字典
    """
    # 计算标准指标
    metrics = calculate_metrics(y_true, y_pred, 
                              metrics=config.METRICS_CONFIG["standard_metrics"] + 
                                      config.METRICS_CONFIG["advanced_metrics"] + 
                                      config.METRICS_CONFIG["relative_metrics"])
    
    # 如果是期权数据，且提供了虚实值，则计算额外的期权特定指标
    if is_option_data and moneyness is not None:
        # 计算时间价值 (假设是call期权)
        # 实值期权 (moneyness > 1)
        itm_mask = moneyness > 1.0
        # 虚值期权 (moneyness < 1)
        otm_mask = moneyness < 1.0
        
        # 计算实值期权指标
        if np.any(itm_mask):
            itm_metrics = calculate_metrics(y_true[itm_mask], y_pred[itm_mask], 
                                         metrics=["rmse", "mae"])
            metrics["in_money_rmse"] = itm_metrics["rmse"]
            metrics["in_money_mae"] = itm_metrics["mae"]
        
        # 计算虚值期权指标
        if np.any(otm_mask):
            otm_metrics = calculate_metrics(y_true[otm_mask], y_pred[otm_mask], 
                                         metrics=["rmse", "mae"])
            metrics["out_money_rmse"] = otm_metrics["rmse"]
            metrics["out_money_mae"] = otm_metrics["mae"]
    
    return metrics


def create_metrics_dataframe(model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    创建包含各模型指标的DataFrame，便于比较和可视化
    
    参数:
        model_results: 包含各模型评估结果的字典
        
    返回:
        整理后的DataFrame
    """
    # 创建空的DataFrame
    metrics_df = pd.DataFrame()
    
    # 遍历每个模型的结果
    for model_name, results in model_results.items():
        # 提取摘要指标
        if "summary" in results:
            row = results["summary"].copy()
        else:
            row = results.copy()
        
        # 添加模型名称
        row["model"] = model_name
        
        # 添加到DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    
    # 重新排列列，使模型名称在前面
    cols = metrics_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("model")))
    metrics_df = metrics_df[cols]
    
    return metrics_df


def save_all_metrics(metrics_df: pd.DataFrame, output_dir: str = None, prefix: str = "") -> str:
    """
    保存指标DataFrame到CSV文件
    
    参数:
        metrics_df: 包含指标的DataFrame
        output_dir: 输出目录
        prefix: 文件名前缀
        
    返回:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文件名
    filename = f"{prefix}_metrics.csv" if prefix else "metrics.csv"
    file_path = os.path.join(output_dir, filename)
    
    # 保存为CSV
    metrics_df.to_csv(file_path, index=False)
    print(f"指标已保存至: {file_path}")
    
    return file_path


def print_all_comparisons(model_results: Dict[str, Dict[str, float]], 
                         metrics: List[str] = None) -> None:
    """
    打印各模型在关键指标上的比较
    
    参数:
        model_results: 包含各模型评估结果的字典
        metrics: 要打印的指标列表
    """
    if metrics is None:
        metrics = ["rmse", "mae", "r2", "time_mean"]
    
    metrics_df = create_metrics_dataframe(model_results)
    
    # 打印模型比较结果
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n=== 模型性能比较 ===")
    if all(metric in metrics_df.columns for metric in metrics):
        print(metrics_df[["model"] + metrics])
    else:
        # 如果某些指标不存在，只打印存在的指标
        existing_metrics = [m for m in metrics if m in metrics_df.columns]
        print(metrics_df[["model"] + existing_metrics])


def visualize_model_comparisons(model_results: Dict[str, Dict[str, float]], 
                              metrics: List[str] = None, 
                              output_dir: str = None,
                              prefix: str = "") -> None:
    """
    可视化不同模型的性能对比
    
    参数:
        model_results: 包含各模型评估结果的字典
        metrics: 要可视化的指标列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    
    if metrics is None:
        metrics = ["rmse", "mae", "r2", "time_mean"]
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataFrame
    metrics_df = create_metrics_dataframe(model_results)
    
    # 设置可视化样式
    plt.style.use(config.VISUALIZATION_CONFIG["style"])
    
    # 设置字体和图表大小
    plt.rcParams['font.family'] = config.VISUALIZATION_CONFIG["font_family"]
    plt.rcParams['font.size'] = config.VISUALIZATION_CONFIG["font_size"]
    
    # 为选定的指标创建条形图
    for metric in metrics:
        if metric not in metrics_df.columns:
            print(f"指标 '{metric}' 在数据中不存在，跳过可视化")
            continue
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config.VISUALIZATION_CONFIG["figsize"])
        
        # 按指标值排序 (对于r2，越高越好；对于其他指标，越低越好)
        if metric == "r2":
            sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        else:
            sorted_df = metrics_df.sort_values(by=metric)
        
        # 限制显示的模型数量，避免图表过于拥挤
        max_models = 15  # 最多显示的模型数量
        if len(sorted_df) > max_models:
            sorted_df = sorted_df.head(max_models)
            print(f"注意: 限制显示前{max_models}个模型以保持图表清晰")
        
        # 创建条形图
        sns.barplot(x="model", y=metric, data=sorted_df, ax=ax)
        
        # 添加标题和标签
        metric_name = metric.replace("_", " ").upper()
        ax.set_title(f"模型{metric_name}对比", fontsize=config.VISUALIZATION_CONFIG["title_font_size"])
        ax.set_xlabel("模型", fontsize=config.VISUALIZATION_CONFIG["font_size"])
        ax.set_ylabel(metric_name, fontsize=config.VISUALIZATION_CONFIG["font_size"])
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        
        # 在每个条形上方显示数值
        for i, v in enumerate(sorted_df[metric]):
            ax.text(i, v * 1.01, f"{v:.4f}", ha='center', 
                  fontsize=config.VISUALIZATION_CONFIG["legend_font_size"])
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filename = f"{prefix}_{metric}_comparison.{config.VISUALIZATION_CONFIG['save_format']}" if prefix else f"{metric}_comparison.{config.VISUALIZATION_CONFIG['save_format']}"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG["dpi"])
        print(f"图表已保存至: {save_path}")
        plt.close()


def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, 
                         output_dir: str = None,
                         prefix: str = "") -> None:
    """
    可视化预测结果与真实值的对比
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置可视化样式
    plt.style.use(config.VISUALIZATION_CONFIG["style"])
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(config.VISUALIZATION_CONFIG["figsize"][0], 
                                         config.VISUALIZATION_CONFIG["figsize"][1] // 2))
    
    # 散点图：预测值vs真实值
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线(理想预测)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title(f'{model_name} - 预测值vs真实值')
    
    # 误差直方图
    errors = y_pred - y_true
    axes[1].hist(errors, bins=30, alpha=0.7, color='skyblue')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('预测误差')
    axes[1].set_ylabel('频率')
    axes[1].set_title(f'{model_name} - 误差分布')
    
    plt.tight_layout()
    
    # 保存图表
    filename = f"{prefix}_{model_name}_predictions.{config.VISUALIZATION_CONFIG['save_format']}" if prefix else f"{model_name}_predictions.{config.VISUALIZATION_CONFIG['save_format']}"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG["dpi"])
    print(f"预测可视化已保存至: {save_path}")
    plt.close()


def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                        model_name: str, 
                        output_dir: str = None,
                        prefix: str = "") -> None:
    """
    绘制学习曲线(训练和验证损失)
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        model_name: 模型名称
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置可视化样式
    plt.style.use(config.VISUALIZATION_CONFIG["style"])
    
    # 创建图表
    plt.figure(figsize=config.VISUALIZATION_CONFIG["figsize"])
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'{model_name} - 学习曲线')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    filename = f"{prefix}_{model_name}_learning_curve.{config.VISUALIZATION_CONFIG['save_format']}" if prefix else f"{model_name}_learning_curve.{config.VISUALIZATION_CONFIG['save_format']}"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG["dpi"])
    print(f"学习曲线已保存至: {save_path}")
    plt.close()
