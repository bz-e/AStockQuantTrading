#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易示例脚本 - 离线版本
使用内置的样本数据，无需联网获取实时数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# 创建模拟的股票数据
def create_sample_data(symbol="模拟股票", days=250):
    """创建模拟的股票数据用于演示"""
    np.random.seed(42)  # 设定随机种子，确保结果可复现
    
    end_date = datetime.datetime.now()
    date_range = pd.date_range(end=end_date, periods=days)
    
    # 创建初始价格和每日波动
    initial_price = 100.0
    daily_returns = np.random.normal(0.0005, 0.018, days)  # 均值略大于0，标准差约为1.8%
    
    # 使用累积回报计算价格序列
    # 这会创建一个具有轻微上升趋势和波动性的序列
    price_series = initial_price * (1 + daily_returns).cumprod()
    
    # 创建OHLC数据
    high = price_series * (1 + np.random.uniform(0, 0.015, days))
    low = price_series * (1 - np.random.uniform(0, 0.015, days))
    open_price = low + np.random.uniform(0, 1, days) * (high - low)
    close = price_series
    volume = np.random.randint(1000000, 10000000, days)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=date_range)
    
    return df

def calculate_ma(df, ma_list=[5, 10, 20, 30, 60]):
    """计算移动平均线"""
    for ma in ma_list:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
    return df

def calculate_rsi(df, periods=14):
    """计算RSI指标"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    return df

def plot_with_indicators(df, symbol, title=None):
    """绘制股价走势、移动平均线和RSI指标"""
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制股价和均线
    ax1.plot(df.index, df['Close'], label='收盘价', linewidth=2)
    
    # 绘制移动平均线
    for col in df.columns:
        if col.startswith('MA'):
            ax1.plot(df.index, df[col], label=col, alpha=0.7)
    
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title(f"{symbol} 股价走势与技术指标", fontsize=14)
    
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # 绘制RSI指标
    ax2.plot(df.index, df['RSI'], color='purple', label='RSI(14)')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
    ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True)
    ax2.set_ylim(0, 100)
    
    # 美化图表
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{symbol}_analysis.png')
    plt.show()

def main():
    # 设置参数
    symbol = "模拟A股"
    
    print(f"创建 {symbol} 的模拟数据用于演示")
    
    # 创建模拟数据
    df = create_sample_data(symbol)
    
    print(f"成功创建模拟数据，共 {len(df)} 条记录")
    
    # 计算指标
    df = calculate_ma(df)
    df = calculate_rsi(df)
    
    # 打印数据概览
    print("\n数据概览:")
    print(df.head())
    
    # 绘制图表
    plot_with_indicators(df, symbol)
    
    # 简单的技术分析
    current_price = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]
    
    print("\n简单技术分析:")
    print(f"当前价格: {current_price:.2f}")
    print(f"20日均线: {ma20:.2f}")
    print(f"60日均线: {ma60:.2f}")
    print(f"当前RSI(14): {current_rsi:.2f}")
    
    # 趋势分析
    if current_price > ma20 > ma60:
        trend = "多头排列，可能处于上升趋势"
    elif current_price < ma20 < ma60:
        trend = "空头排列，可能处于下降趋势"
    else:
        trend = "趋势不明确"
    
    # RSI分析
    if current_rsi > 70:
        rsi_signal = "超买区域，可能面临回调风险"
    elif current_rsi < 30:
        rsi_signal = "超卖区域，可能存在反弹机会"
    else:
        rsi_signal = "中性区域"
    
    print(f"均线分析: {trend}")
    print(f"RSI分析: {rsi_signal}")
    
    # 简单的策略建议
    if current_price > ma20 > ma60 and 50 < current_rsi < 70:
        print("策略建议: 可考虑逢低买入，趋势向上")
    elif current_price < ma20 < ma60 and 30 < current_rsi < 50:
        print("策略建议: 可考虑逢高卖出，趋势向下")
    elif current_rsi > 70:
        print("策略建议: 谨慎追高，注意回调风险")
    elif current_rsi < 30:
        print("策略建议: 可能存在超跌反弹机会")
    else:
        print("策略建议: 观望为主，等待更明确的信号")

if __name__ == "__main__":
    main()
