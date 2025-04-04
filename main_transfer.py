#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两阶段训练过程的主脚本：
1. 在SVJJ理论模型数据上预训练
2. 在真实数据上微调
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from transfer_utils import TransferDataProcessor
from ml_dl_models_compare import (
    DeepMLP, ResidualMLP, SimpleCNN, AdvancedCNN,
    SimpleRNN, SimpleLSTM, SimpleGRU, SimpleTransformer,
    BayesianNN
)

def parse_args():
    parser = argparse.ArgumentParser(description='两阶段训练过程')
    parser.add_argument('--mode', type=str, default='all',
                      choices=['ml', 'dl', 'all'],
                      help='训练模式：机器学习、深度学习或全部')
    parser.add_argument('--model_type', type=str, default='all',
                      choices=['all', 'deepmlp', 'residualmlp', 'cnn', 'advancedcnn',
                              'rnn', 'lstm', 'gru', 'transformer', 'bayesian'],
                      help='要训练的模型类型')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                      help='预训练轮数')
    parser.add_argument('--finetune_epochs', type=int, default=50,
                      help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='训练设备')
    parser.add_argument('--output_dir', type=str, default='results/transfer',
                      help='输出目录')
    return parser.parse_args()

def train_model(model, train_loader, val_loader, epochs, device, learning_rate=1e-3):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred_batch = model(X_batch)
            # 只使用第一列作为预测结果
            y_true.extend(y_batch[:, 0].numpy())
            y_pred.extend(y_pred_batch[:, 0].cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

def plot_learning_curves(train_losses, val_losses, title, save_path):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化数据处理器
    processor = TransferDataProcessor()
    
    # 加载数据
    print("加载SVJJ理论模型数据...")
    svjj_data = processor.load_svjj_data('data/SVJJ001.csv')
    X_svjj, y_svjj = processor.prepare_svjj_data(svjj_data)
    
    print("加载真实数据...")
    real_data = processor.load_real_data('/Users/a1/Desktop/ML999.csv')
    X_real, y_real = processor.prepare_real_data(real_data)
    
    # 分割真实数据为训练集和测试集
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    svjj_train_loader, svjj_val_loader = processor.create_data_loaders(X_svjj, y_svjj, args.batch_size)
    real_train_loader, real_val_loader = processor.create_data_loaders(X_real_train, y_real_train, args.batch_size)
    real_test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_real_test), torch.FloatTensor(y_real_test)),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 定义要训练的模型
    models = {
        'deepmlp': DeepMLP(X_svjj.shape[1], hidden_layers=[128, 64, 32], out_dim=y_svjj.shape[1]),
        'residualmlp': ResidualMLP(X_svjj.shape[1], hidden_dim=64, blocks=3, out_dim=y_svjj.shape[1]),
        'cnn': SimpleCNN(X_svjj.shape[1], out_dim=y_svjj.shape[1], channels=[32, 64]),
        'advancedcnn': AdvancedCNN(X_svjj.shape[1], out_dim=y_svjj.shape[1], channels=[32, 64, 128]),
        'rnn': SimpleRNN(X_svjj.shape[1], hidden_dim=64, num_layers=2, out_dim=y_svjj.shape[1]),
        'lstm': SimpleLSTM(X_svjj.shape[1], hidden_dim=64, num_layers=2, out_dim=y_svjj.shape[1]),
        'gru': SimpleGRU(X_svjj.shape[1], hidden_dim=64, num_layers=2, out_dim=y_svjj.shape[1]),
        'transformer': SimpleTransformer(X_svjj.shape[1], d_model=64, nhead=4, num_layers=2, out_dim=y_svjj.shape[1]),
        'bayesian': BayesianNN(X_svjj.shape[1], hidden_dims=[64, 32], out_dim=y_svjj.shape[1])
    }
    
    # 训练和评估结果
    results = {}
    
    # 选择要训练的模型
    if args.model_type == 'all':
        model_types = models.keys()
    else:
        model_types = [args.model_type]
    
    for model_type in model_types:
        print(f"\n训练 {model_type} 模型...")
        
        # 创建模型实例
        model = models[model_type].to(args.device)
        
        # 预训练阶段
        print("预训练阶段...")
        model, pretrain_train_losses, pretrain_val_losses = train_model(
            model, svjj_train_loader, svjj_val_loader,
            args.pretrain_epochs, args.device, args.learning_rate
        )
        
        # 保存预训练模型
        processor.save_model_and_scaler(model, args.output_dir, f"{model_type}_pretrained")
        
        # 绘制预训练学习曲线
        plot_learning_curves(
            pretrain_train_losses, pretrain_val_losses,
            f'{model_type} 预训练学习曲线',
            os.path.join(args.output_dir, f'{model_type}_pretrain_curves.png')
        )
        
        # 微调阶段
        print("微调阶段...")
        model, finetune_train_losses, finetune_val_losses = train_model(
            model, real_train_loader, real_val_loader,
            args.finetune_epochs, args.device, args.learning_rate * 0.1
        )
        
        # 保存微调后的模型
        processor.save_model_and_scaler(model, args.output_dir, f"{model_type}_finetuned")
        
        # 绘制微调学习曲线
        plot_learning_curves(
            finetune_train_losses, finetune_val_losses,
            f'{model_type} 微调学习曲线',
            os.path.join(args.output_dir, f'{model_type}_finetune_curves.png')
        )
        
        # 评估模型
        print("评估模型...")
        metrics = evaluate_model(model, real_test_loader, args.device)
        
        # 保存结果
        results[model_type] = {
            'pretrain_metrics': {
                'final_train_loss': pretrain_train_losses[-1],
                'final_val_loss': pretrain_val_losses[-1]
            },
            'finetune_metrics': {
                'final_train_loss': finetune_train_losses[-1],
                'final_val_loss': finetune_val_losses[-1]
            },
            'test_metrics': metrics
        }
        
        print(f"测试集评估结果:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R2: {metrics['r2']:.4f}")
    
    # 保存所有结果
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(args.output_dir, 'results.csv'))
    
    # 绘制模型比较图
    plt.figure(figsize=(12, 6))
    metrics = ['mse', 'mae', 'r2']
    x = np.arange(len(model_types))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[model_type]['test_metrics'][metric] for model_type in model_types]
        plt.bar(x + i*width, values, width, label=metric.upper())
    
    plt.xlabel('Models')
    plt.ylabel('Metric Values')
    plt.title('Model Performance Comparison on Test Set')
    plt.xticks(x + width, model_types, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    main(args) 