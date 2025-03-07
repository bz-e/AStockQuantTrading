#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票比较功能示例
演示如何比较多只A股的性能表现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baostock as bs
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')  # 忽略matplotlib中文警告

# 创建images目录（如果不存在）
if not os.path.exists('images'):
    os.makedirs('images')

# 计算技术指标
def calculate_indicators(df):
    """计算各种技术指标"""
    # 确保数据按时间排序
    df = df.sort_values('date')
    
    # 计算移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # 计算RSI (相对强弱指标)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def get_stock_data(stock_code, start_date, end_date):
    """使用Baostock获取股票数据"""
    # 登录baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Baostock登录失败: {lg.error_msg}')
        return None
    
    # 获取股票日K线数据
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,volume,amount,adjustflag,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 复权类型: 1:后复权, 2:前复权, 3:不复权
    )
    
    if rs.error_code != '0':
        print(f'获取股票数据失败: {rs.error_msg}')
        bs.logout()
        return None
    
    # 处理数据
    data_list = []
    while (rs.next()):
        data_list.append(rs.get_row_data())
    
    # 退出baostock
    bs.logout()
    
    if len(data_list) == 0:
        print(f'无法获取股票数据，请检查网络连接或股票代码是否正确')
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算各种技术指标
    df = calculate_indicators(df)
    
    print(f'成功获取{stock_code}数据，共 {len(df)} 条记录')
    return df

def get_stock_info(stock_code):
    """获取股票基本信息"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Baostock登录失败: {lg.error_msg}')
        return None
    
    rs = bs.query_stock_basic(code=stock_code)
    if rs.error_code != '0':
        print(f'获取股票信息失败: {rs.error_msg}')
        bs.logout()
        return None
    
    # 处理数据
    data_list = []
    while (rs.next()):
        data_list.append(rs.get_row_data())
    
    bs.logout()
    
    if len(data_list) == 0:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    return df

def compare_stocks(stock_codes, start_date, end_date):
    """比较多只股票的表现"""
    stock_data = {}
    stock_names = {}
    
    for code in stock_codes:
        # 获取股票信息
        info_df = get_stock_info(code)
        if info_df is not None and not info_df.empty:
            stock_names[code] = info_df.iloc[0]['code_name']
        else:
            stock_names[code] = code
        
        # 获取股票数据
        df = get_stock_data(code, start_date, end_date)
        if df is not None and not df.empty:
            stock_data[code] = df
    
    if not stock_data:
        print("未能获取任何股票数据")
        return
    
    # 绘制股价对比图
    plt.figure(figsize=(12, 10))
    
    # 价格走势对比
    plt.subplot(2, 1, 1)
    for code, df in stock_data.items():
        # 首日价格标准化为100，便于比较相对表现
        first_price = df['close'].iloc[0]
        normalized_price = df['close'] / first_price * 100
        plt.plot(df['date'], normalized_price, label=f"{stock_names.get(code, code)}")
    
    plt.title('股票价格走势对比 (基准化为100)', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('价格 (基准化)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # RSI对比
    plt.subplot(2, 1, 2)
    for code, df in stock_data.items():
        plt.plot(df['date'], df['RSI'], label=f"{stock_names.get(code, code)} RSI")
    
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    plt.title('RSI指标对比', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('RSI值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/stock_comparison.png')
    plt.show()
    
    # 绩效统计对比
    performance = {}
    for code, df in stock_data.items():
        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        returns = (last_price / first_price - 1) * 100
        volatility = df['pctChg'].std()
        max_drawdown = 0
        peak = df['close'].iloc[0]
        
        for price in df['close']:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        performance[code] = {
            '股票名称': stock_names.get(code, code),
            '收益率(%)': round(returns, 2),
            '波动率(%)': round(volatility, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '夏普比率': round(returns / volatility if volatility > 0 else 0, 2)
        }
    
    perf_df = pd.DataFrame.from_dict(performance, orient='index')
    perf_df = perf_df.sort_values('收益率(%)', ascending=False)
    
    print("\n======== 绩效对比 ========")
    print(perf_df)
    
    # 性能分析与建议
    best_stock = perf_df.iloc[0].name
    best_name = perf_df.iloc[0]['股票名称']
    worst_stock = perf_df.iloc[-1].name
    worst_name = perf_df.iloc[-1]['股票名称']
    
    print("\n======== 分析建议 ========")
    print(f"1. 在分析周期内，{best_name}({best_stock})表现最佳，收益率为{perf_df.iloc[0]['收益率(%)']}%")
    print(f"2. {worst_name}({worst_stock})表现最差，收益率为{perf_df.iloc[-1]['收益率(%)']}%")
    
    # 简单的相关性分析
    print("\n======== 相关性分析 ========")
    price_data = {}
    for code, df in stock_data.items():
        price_data[stock_names.get(code, code)] = df['close'].values
    
    price_df = pd.DataFrame(price_data)
    corr_matrix = price_df.corr()
    print(corr_matrix)
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('股票价格相关性矩阵', fontsize=14)
    
    # 添加相关系数标签
    tick_marks = np.arange(len(corr_matrix.columns))
    plt.xticks(tick_marks, corr_matrix.columns, rotation=45)
    plt.yticks(tick_marks, corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                    ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('images/correlation_matrix.png')
    plt.show()

if __name__ == "__main__":
    # 设置股票代码列表（茅台、平安银行、中国石油）
    stock_list = ["sh.600519", "sz.000001", "sh.601857"]
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 一年数据
    
    print(f"对比分析股票: {', '.join(stock_list)}")
    print(f"分析日期范围: {start_date} 至 {end_date}")
    
    # 进行对比分析
    compare_stocks(stock_list, start_date, end_date)
