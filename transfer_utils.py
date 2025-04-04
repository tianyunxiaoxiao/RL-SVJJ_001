#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于处理数据预处理和迁移学习的工具类
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, List, Any
import os

class TransferDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_svjj_data(self, file_path: str) -> pd.DataFrame:
        """加载SVJJ理论模型数据"""
        return pd.read_csv(file_path)
    
    def load_real_data(self, file_path: str) -> pd.DataFrame:
        """加载真实数据"""
        return pd.read_csv(file_path)
    
    def prepare_svjj_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备SVJJ数据用于训练"""
        # 选择特征列（除了价格列）
        feature_cols = ['tau', 'V', 'S', 'moneyness', 'discount_factor', 'forward_price', 'H']
        # 选择目标列
        target_cols = ['call_price', 'put_price']
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def prepare_real_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备真实数据用于迁移学习"""
        # 数据预处理
        df['days_gap'] = df['days_gap'].str.extract('(\d+)').astype(float)  # 提取数字并转换为浮点数
        
        # 将日期字符串转换为时间戳
        df['maturity'] = pd.to_datetime(df['maturity']).astype(np.int64) // 10**9  # 转换为Unix时间戳
        
        # 选择与SVJJ数据相对应的特征列
        feature_cols = [
            'days',  # 对应tau
            'impl_volatility',  # 对应V
            'impl_strike',  # 对应S
            'moneyness',  # 对应moneyness
            'rate',  # 可以用来计算discount_factor
            'BS',  # 可以作为forward_price的替代
            'historical_vol'  # 可以作为H的替代
        ]
        
        # 选择目标列（使用impl_premium作为价格）
        target_cols = ['impl_premium']
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # 使用相同的scaler进行标准化
        X = self.scaler.transform(X)
        
        # 将y转换为2D数组以匹配预训练模型的输出维度
        y = np.column_stack([y, y])  # 复制一列以匹配预训练模型的输出维度
        
        return X, y
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray,
                          batch_size: int = 64,
                          train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器"""
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 分割训练集和验证集
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def save_model_and_scaler(self, model: Any, save_dir: str, model_name: str):
        """保存模型和标准化器"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_model.pth"))
        else:
            import joblib
            joblib.dump(model, os.path.join(save_dir, f"{model_name}_model.joblib"))
        
        # 保存标准化器
        import joblib
        joblib.dump(self.scaler, os.path.join(save_dir, f"{model_name}_scaler.joblib"))
    
    def load_model_and_scaler(self, save_dir: str, model_name: str) -> Tuple[Any, StandardScaler]:
        """加载模型和标准化器"""
        # 加载标准化器
        import joblib
        scaler = joblib.load(os.path.join(save_dir, f"{model_name}_scaler.joblib"))
        
        # 加载模型
        model_path = os.path.join(save_dir, f"{model_name}_model.pth")
        if os.path.exists(model_path):
            # 如果是PyTorch模型
            model = torch.load(model_path)
        else:
            # 如果是其他模型
            model = joblib.load(os.path.join(save_dir, f"{model_name}_model.joblib"))
        
        return model, scaler 