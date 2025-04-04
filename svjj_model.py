# (5) SVJJ相关模型 & 数据生成
# === svjj_model.py ===
"""
SVJJ模型相关：SVCJParams定义 & 数据模拟函数
实现了完整版本的随机波动率跳跃扩散模型(SVCJ)，包括特征函数和期权定价
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
from scipy.stats import norm
from scipy.integrate import quad
import pandas as pd
from dataclasses import dataclass, replace
from typing import List, Tuple
import warnings
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
import dataclasses
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class SVCJParams:
    """随机波动率跳跃扩散模型(SVCJ)的参数定义"""
    
    alp_v: float     # Mean-reverting speed of volatility
    m_v: float       # Long-term mean of volatility
    sig_v: float     # Volatility of volatility
    rho: float       # Correlation between underlying and volatility
    lam: float       # Jump arrival rate (P measure)
    lam_q: float     # Jump arrival rate (Q measure)
    v0: float        # Initial volatility
    rho_j: float     # Correlation between jumps
    mu_v: float      # Vol Jump's mean under P
    mu_vq: float     # Vol Jump's mean under Q
    gam_s: float     # Market price risks of stock
    gam_v: float     # Market price risks of vol
    mu_s: float      # Jump mean under P
    sig_s: float     # Jump vol under P
    mu_sq: float     # Jump mean under Q
    sig_sq: float    # Jump vol under Q
    r: float         # Risk-free rate
    q: float         # Dividend yield

def cij_all(tau, z1, z2, bij, uij, r, a, rho, sig_v, rho_j, mu_v, mu_s, sig_s, lam_y, lam_v, lam_c):
    """
    计算SVCJ特征函数的积分项
    参数:
        tau: 期权期限
        z1, z2: 复变量
        bij, uij: 辅助参数
        r: 无风险利率
        a: 波动率均值回归速度与长期均值的乘积
        rho: 标的与波动率的相关性
        sig_v: 波动率的波动率
        rho_j: 跳跃相关性
        mu_v: 波动率跳跃均值(P测度)
        mu_s: 价格跳跃均值(P测度)
        sig_s: 价格跳跃波动率(P测度)
        lam_y, lam_v, lam_c: 跳跃到达率
    """
    lam_bar = lam_y + lam_v + lam_c
    xj = bij - rho * sig_v * z1
    dj = np.sqrt(xj**2 - (sig_v**2) * (2 * uij * z1 + z1**2))
    
    if abs(xj-dj) < abs(xj+dj):
        xjmdj = (sig_v**2) * (2 * uij * z1 + z1**2) / (xj + dj)
        gj = xjmdj / (xj + dj)
        qj = (xjmdj - sig_v**2 * z2) / (xj + dj - sig_v**2 * z2)
        
        Cleft = np.exp(mu_s * z1 + 0.5 * sig_s**2 * z1**2)
        fy = Cleft * tau
        
        def Guv(t): 
            return (sig_v**2 * (1 - qj * np.exp(-dj * t)) - 
                   mu_v * (xj + dj) * (gj - qj * np.exp(-dj * t)))
        lnGuv = np.log(Guv(tau) / Guv(0))
        fv = (sig_v**2 * tau / (sig_v**2 - mu_v * xjmdj) - 
             2 * sig_v**2 * mu_v / ((sig_v**2 - mu_v * xj)**2 - mu_v**2 * dj**2) * lnGuv)
        
        m = 1 - rho_j * z1 * mu_v
        def Gu(t):
            return (m * sig_v**2 * (1 - qj * np.exp(-dj * t)) - 
                   mu_v * (xj + dj) * (gj - qj * np.exp(-dj * t)))
        lnGu = np.log(Gu(tau) / Gu(0))
    else:
        gjinv = (xj + dj) / (xj - dj)
        qjinv = (xj + dj - sig_v**2 * z2) / (xj - dj - sig_v**2 * z2)
        
        Cleft = np.exp(mu_s * z1 + 0.5 * sig_s**2 * z1**2)
        fy = Cleft * tau
        
        def Guv(t):
            return (sig_v**2 * (qjinv - np.exp(-dj * t)) - 
                   mu_v * (xj - dj) * (qjinv - gjinv * np.exp(-dj * t)))
        lnGuv = np.log(Guv(tau) / Guv(0))
        fv = (sig_v**2 * tau / (sig_v**2 - mu_v * (xj - dj)) - 
             2 * sig_v**2 * mu_v / ((sig_v**2 - mu_v * xj)**2 - mu_v**2 * dj**2) * lnGuv)
        
        m = 1 - rho_j * z1 * mu_v
        def Gu(t):
            return (m * sig_v**2 * (qjinv - np.exp(-dj * t)) - 
                   mu_v * (xj - dj) * (qjinv - gjinv * np.exp(-dj * t)))
        lnGu = np.log(Gu(tau) / Gu(0))
    
    Cright = (sig_v**2 * tau / (m * sig_v**2 - mu_v * (xj - dj)) - 
             2 * sig_v**2 * mu_v / ((m * sig_v**2 - mu_v * xj)**2 - mu_v**2 * dj**2) * lnGu)
    fc = Cleft * Cright
    
    return (1 / lam_bar) * (lam_y * fy + lam_v * fv + lam_c * fc)

def bij(tau, z1, z2, bij_val, uij, r, a, rho, sig_v):
    """
    计算Riccati方程的bij项
    参数:
        tau: 期权期限
        z1, z2: 复变量
        bij_val: bij的初始值
        uij: 辅助参数
        r: 无风险利率
        a: 波动率均值回归速度与长期均值的乘积
        rho: 标的与波动率的相关性
        sig_v: 波动率的波动率
    """
    xj = bij_val - rho * sig_v * z1
    dj = np.sqrt(xj**2 - (sig_v**2) * (2 * uij * z1 + z1**2))
    
    if abs(xj-dj) < abs(xj+dj):
        gj = (xj - dj) / (xj + dj)
        qj = (xj - dj - sig_v**2 * z2) / (xj + dj - sig_v**2 * z2)
        B = ((xj + dj) / (sig_v**2)) * ((gj - qj * np.exp(-dj * tau)) / 
                                       (1 - qj * np.exp(-dj * tau)))
    else:
        gjinv = (xj + dj) / (xj - dj)
        qjinv = (xj + dj - sig_v**2 * z2) / (xj - dj - sig_v**2 * z2)
        B = ((xj - dj) / (sig_v**2)) * ((qjinv - gjinv * np.exp(-dj * tau)) / 
                                       (qjinv - np.exp(-dj * tau)))
    return B

def aij(tau, z1, z2, bij_val, uij, r, a, rho, sig_v, q, rho_j, mu_v, mu_s, sig_s, 
        lam, lam_q, mu_sq, sig_sq, mu_vq):
    """
    计算Riccati方程的aij项
    参数:
        tau: 期权期限
        z1, z2: 复变量
        bij_val: bij的值
        uij: 辅助参数
        r: 无风险利率
        a: 波动率均值回归速度与长期均值的乘积
        rho: 标的与波动率的相关性
        sig_v: 波动率的波动率
        q: 股息率
        rho_j: 跳跃相关性
        mu_v: 波动率跳跃均值(P测度)
        mu_s: 价格跳跃均值(P测度)
        sig_s: 价格跳跃波动率(P测度)
        lam: P测度下的跳跃到达率
        lam_q: Q测度下的跳跃到达率
        mu_sq: Q测度下的价格跳跃均值
        sig_sq: Q测度下的价格跳跃波动率
        mu_vq: Q测度下的波动率跳跃均值
    """
    barmu_s = np.exp(mu_sq + 0.5 * sig_sq**2) / (1 - rho_j * mu_vq) - 1
    Cij0 = cij_all(tau, z1, z2, bij_val, uij, r, a, rho, sig_v, rho_j, mu_v, mu_s, sig_s, 
                   0, 0, lam)
    
    xj = bij_val - rho * sig_v * z1
    dj = np.sqrt(xj**2 - (sig_v**2) * (2 * uij * z1 + z1**2))
    
    if abs(xj-dj) < abs(xj+dj):
        qj = (xj - dj - sig_v**2 * z2) / (xj + dj - sig_v**2 * z2)
        def xx(t): 
            return 1 - qj * np.exp(-dj * t)
        A = ((r - q - lam_q * barmu_s) * z1 * tau + 
             a / (sig_v**2) * ((xj - dj) * tau - 2 * np.log(xx(tau) / xx(0))) + 
             lam * (Cij0 - tau))
    else:
        qjinv = (xj + dj - sig_v**2 * z2) / (xj - dj - sig_v**2 * z2)
        def xx(t): 
            return qjinv - np.exp(-dj * t)
        A = ((r - q - lam_q * barmu_s) * z1 * tau + 
             a / (sig_v**2) * ((xj - dj) * tau - 2 * np.log(xx(tau) / xx(0))) + 
             lam * (Cij0 - tau))
    return A

def cf_svcji(z, St, T, t, H, params: SVCJParams):
    """
    计算SVCJ模型的特征函数
    参数:
        z: 傅里叶变量
        St: 当前标的价格
        T: 期权到期时间
        t: 当前时间
        H: 中间时间点
        params: SVCJ模型参数
    """
    ln_s = np.log(St)
    ui1 = -0.5
    ui2 = -0.5 + params.gam_s
    bi1 = params.alp_v + params.gam_v
    bi2 = params.alp_v
    a = params.alp_v * params.m_v
    
    Ai1 = aij(T-H, z, 0, bi1, ui1, params.r, a, params.rho, params.sig_v, 
              params.q, params.rho_j, params.mu_vq, params.mu_sq, params.sig_sq, 
              params.lam_q, params.lam_q, params.mu_sq, params.sig_sq, params.mu_vq)
    
    Bi1 = bij(T-H, z, 0, bi1, ui1, params.r, a, params.rho, params.sig_v)
    
    Ai2 = aij(H-t, z, Bi1, bi2, ui2, params.r, a, params.rho, params.sig_v, 
              params.q, params.rho_j, params.mu_v, params.mu_s, params.sig_s,
              params.lam, params.lam_q, params.mu_sq, params.sig_sq, params.mu_vq)
    
    Bi2 = bij(H-t, z, Bi1, bi2, ui2, params.r, a, params.rho, params.sig_v)
    
    return np.exp(-params.r * (T-H) + Ai1 + Ai2 + Bi2 * params.v0 + z * ln_s)

def svcj_option_price(St, K, t, T, H, params: SVCJParams):
    """
    计算SVCJ模型下的期权价格
    参数:
        St: 当前标的价格
        K: 期权行权价
        t: 当前时间
        T: 期权到期时间
        H: 中间时间点
        params: SVCJ模型参数
    
    返回:
        ExPC: 看涨期权价格
        ExPP: 看跌期权价格
        SH: 远期价格
        DIS: 贴现因子
        P1: 一阶概率
        P2: 二阶概率
    """
    def integrand1(x):
        return np.real(np.exp(-1j * x * np.log(K)) * 
                      (cf_svcji(x * 1j + 1, St, T, t, H, params) / 
                       cf_svcji(1 + 1e-11, St, T, t, H, params)) / (1j * x))
    
    def integrand2(x):
        return np.real(np.exp(-1j * x * np.log(K)) * 
                      (cf_svcji(x * 1j, St, T, t, H, params) / 
                       cf_svcji(0 + 1e-11, St, T, t, H, params)) / (1j * x))
    
    P1 = 0.5 + (1/np.pi) * quad(integrand1, 0, np.inf)[0]
    P2 = 0.5 + (1/np.pi) * quad(integrand2, 0, np.inf)[0]
    
    SH = cf_svcji(1 + 1e-11, St, T, t, H, params)
    DIS = cf_svcji(0 + 1e-11, St, T, t, H, params)
    
    ExPC = SH * P1 - K * DIS * P2
    ExPP = ExPC - SH + K * DIS
    ExPS = SH * np.exp(params.q * (T-H))
    ExPB = DIS
    
    return ExPC, ExPP, SH, DIS, P1, P2

def generate_jumps(params: SVCJParams, n_jumps: int):
    """
    生成相关跳跃项
    参数:
        params: SVCJ模型参数
        n_jumps: 跳跃次数
    
    返回:
        J_s: 价格跳跃
        J_v: 波动率跳跃
    """
    if n_jumps <= 0:
        return 0.0, 0.0

    # 生成波动率跳跃(指数分布)
    J_v = np.random.exponential(params.mu_v, n_jumps)
    
    # 生成条件价格跳跃
    J_s = np.random.normal(
        params.mu_s + params.rho_j * J_v,  # 条件均值
        params.sig_s,                      # 标准差
        n_jumps
    )
    
    return np.sum(J_s), np.sum(J_v)

def simulate_weekly_atm_option_data(params: SVCJParams, S0=100, weeks=52):
    """
    模拟生成平价期权的周度数据
    参数:
        params: SVCJ模型参数
        S0: 初始价格
        weeks: 模拟周数
    
    返回:
        pandas.DataFrame: 包含模拟数据的数据框
    """
    dt = 1/252  # 每日时间步长(一年252个交易日)
    days_per_week = 5  # 每周5个交易日
    days = weeks * days_per_week  # 总交易日数
    
    # 初始化价格和波动率序列
    S = np.zeros(days + 1)
    v = np.zeros(days + 1)
    S[0] = S0
    v[0] = params.v0  # 使用初始波动率
    
    # 生成相关布朗运动
    dW1 = np.random.normal(0, np.sqrt(dt), days)
    dW2 = params.rho * dW1 + np.sqrt(1 - params.rho**2) * np.random.normal(0, np.sqrt(dt), days)
    
    # 计算价格跳跃大小的无条件均值
    mu_bar = np.exp(params.mu_s + 0.5 * params.sig_s**2) / (1 - params.rho_j * params.mu_v) - 1
    
    # 模拟路径
    for t in range(days):
        # 根据泊松到达生成跳跃
        n_jumps = np.random.poisson(params.lam * dt)
        
        if n_jumps > 0:
            # 根据指定分布生成相关跳跃
            J_s, J_v = generate_jumps(params, n_jumps)
        else:
            J_s, J_v = 0.0, 0.0
        
        # 更新波动率过程
        v[t+1] = np.maximum(
            v[t] + params.alp_v * (params.m_v - v[t]) * dt +
            params.sig_v * np.sqrt(np.maximum(v[t], 0.0)) * dW1[t] + J_v,
            0.0  # 确保波动率非负
        )
        
        # 计算漂移项
        drift = (params.r - params.q + params.gam_s * v[t] - 0.5 * v[t] - 
                params.lam * mu_bar)  
        
        # 更新价格过程
        S[t+1] = S[t] * np.exp(
            drift * dt +
            np.sqrt(np.maximum(v[t], 0.0)) * dW2[t] + J_s
        )
    
    # 生成期权数据(周度)
    data = []
    tau = 1/12  # 固定一个月到期
    
    for week in range(weeks):
        t_idx = week * days_per_week
        St = S[t_idx]
        vt = v[t_idx]
        
        # 平价期权
        K = St
        
        # 计算期权价格
        call, put, SH, DIS, P1, P2 = svcj_option_price(
            St, K, 0, tau, tau/2, params
        )
        
        # 存储结果
        data.append({
            'week': week,
            'S': St,
            'V': vt,
            'K': K,
            'tau': tau,
            'H': tau/2,
            'moneyness': 1.0,
            'call_price': call,
            'put_price': put,
            'P1': P1,
            'P2': P2,
            'forward_price': SH,
            'discount_factor': DIS,
            'log_return': np.log(St/S[t_idx-1]) if t_idx > 0 else 0
        })
    
    return pd.DataFrame(data)

def generate_random_option_data(params: SVCJParams, num_samples=10000, S0=100, vol_range=(0.1, 0.5),
                               moneyness_range=(0.8, 1.2), tau_range=(1/52, 1/4)):
    """
    生成随机期权数据样本，用于训练深度学习/强化学习模型
    
    参数:
        params: SVCJ模型参数
        num_samples: 生成样本数量
        S0: 初始价格
        vol_range: 波动率范围(最小值,最大值)
        moneyness_range: 期权价值比范围(K/S)
        tau_range: 到期时间范围(最小值,最大值)
        
    返回:
        pandas.DataFrame: 包含随机生成期权数据的数据框
    """
    data = []
    
    # 生成随机样本
    for i in range(num_samples):
        # 随机股票价格(围绕S0)
        St = S0 * np.exp(np.random.normal(0, 0.2))
        
        # 随机波动率
        vt = np.random.uniform(vol_range[0], vol_range[1]) ** 2  # 平方得到方差
        
        # 随机价值比
        moneyness = np.random.uniform(moneyness_range[0], moneyness_range[1])
        K = St * moneyness
        
        # 随机到期时间(年)
        tau = np.random.uniform(tau_range[0], tau_range[1])
        
        # 设置params的v0
        params_updated = replace(params, v0=vt)
        
        # 计算期权价格
        call, put, SH, DIS, P1, P2 = svcj_option_price(
            St, K, 0, tau, tau/2, params_updated
        )
        
        # 存储结果
        data.append({
            'sample_id': i,
            'S': St,
            'V': vt,
            'K': K,
            'tau': tau,
            'H': tau/2,
            'moneyness': moneyness,
            'call_price': call,
            'put_price': put,
            'P1': P1,
            'P2': P2,
            'forward_price': SH,
            'discount_factor': DIS
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 设置SVCJ模型参数
    params = SVCJParams(
        r=0.0201,                   # 无风险利率
        q=0.0174,                   # 股息率
        alp_v=0.026 * 252,          # 波动率均值回归速度
        m_v=0.54 * 252 / 10000,     # 波动率长期均值
        sig_v=0.08 * 252 / 100,     # 波动率的波动率
        rho=-0.48,                  # 标的与波动率相关性
        lam=0.006 * 252,            # 跳跃到达率(P测度)
        lam_q=0.006 * 252,          # 跳跃到达率(Q测度)
        v0=0.54 * 252 / 10000,      # 初始波动率
        rho_j=np.finfo(float).eps,  # 跳跃相关性(接近零)
        mu_v=1.48 * 252 / 10000,    # 波动率跳跃均值(P测度)
        mu_vq=8.78 * 252 / 10000,   # 波动率跳跃均值(Q测度)
        gam_s=0.04,                 # 股票市场价格风险
        gam_v=-0.031 * 252,         # 波动率市场价格风险
        mu_s=-2.63 / 100,           # 价格跳跃均值(P测度)
        sig_s=2.89 / 100,           # 价格跳跃波动率(P测度)
        mu_sq=-2.63 / 100,          # 价格跳跃均值(Q测度)
        sig_sq=2.89 / 100           # 价格跳跃波动率(Q测度)
    )

    # 随机生成样本点
    num_samples = 1000000  # 生成样本数量，可根据需要调整
    option_data = generate_random_option_data(params, num_samples=num_samples)

    # 保存生成的数据
    # option_data.to_csv('random_option_data.csv', index=False)

    print("\n生成的随机期权数据摘要:")
    print(f"观测数量: {len(option_data)}")
    print("\n统计摘要:")
    print(option_data.describe())
    print("\n数据样例:")
    print(option_data.head())

    # 可视化一些关键关系
    plt.figure(figsize=(15, 10))
    
    # 1. 波动率与期权价格关系
    plt.subplot(2, 2, 1)
    plt.scatter(option_data['V'], option_data['call_price'], alpha=0.5, s=5)
    plt.title('波动率与看涨期权价格关系')
    plt.xlabel('波动率(V)')
    plt.ylabel('看涨期权价格')
    
    # 2. 价值比与期权价格关系
    plt.subplot(2, 2, 2)
    plt.scatter(option_data['moneyness'], option_data['call_price'], alpha=0.5, s=5)
    plt.title('价值比与看涨期权价格关系')
    plt.xlabel('价值比(K/S)')
    plt.ylabel('看涨期权价格')
    
    # 3. 到期时间与期权价格关系
    plt.subplot(2, 2, 3)
    plt.scatter(option_data['tau'], option_data['call_price'], alpha=0.5, s=5)
    plt.title('到期时间与看涨期权价格关系')
    plt.xlabel('到期时间(年)')
    plt.ylabel('看涨期权价格')
    
    # 4. 波动率与看跌期权价格关系
    plt.subplot(2, 2, 4)
    plt.scatter(option_data['V'], option_data['put_price'], alpha=0.5, s=5)
    plt.title('波动率与看跌期权价格关系')
    plt.xlabel('波动率(V)')
    plt.ylabel('看跌期权价格')
    
    plt.tight_layout()
    # plt.savefig('option_data_visualization.png')
    # plt.show()
