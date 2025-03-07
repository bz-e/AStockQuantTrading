#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易示例脚本 - 兼容性版本
使用pandas-datareader获取数据，不依赖akshare和TA-Lib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime
import os

def get_index_data(symbol, start_date, end_date):
    """获取指数数据 - 使用Yahoo Finance数据源"""
    try:
        # 使用Yahoo获取指数数据
        if symbol == "上证指数":
            yahoo_symbol = "000001.SS"
        elif symbol == "深证成指":
            yahoo_symbol = "399001.SZ"
        elif symbol == "沪深300":
            yahoo_symbol = "000300.SS"
        else:
            yahoo_symbol = symbol
            
        df = pdr.data.get_data_yahoo(yahoo_symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"获取数据出错: {e}")
        return None

def calculate_ma(df, ma_list=[5, 10, 20, 30, 60]):
    """计算移动平均线"""
    for ma in ma_list:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
    return df

def plot_with_ma(df, symbol, title=None):
    """绘制走势图与移动平均线"""
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['Close'], label='收盘价')
    
    # 绘制移动平均线
    for col in df.columns:
        if col.startswith('MA'):
            plt.plot(df.index, df[col], label=col)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"{symbol} 走势与均线")
    
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{symbol}_analysis.png')
    plt.show()

def main():
    # 设置参数
    symbol = "上证指数"
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    
    print(f"获取 {symbol} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据")
    
    # 获取数据
    df = get_index_data(symbol, start_date, end_date)
    
    if df is not None and not df.empty:
        print(f"成功获取数据，共 {len(df)} 条记录")
        
        # 计算移动平均线
        df = calculate_ma(df)
        
        # 打印数据概览
        print("\n数据概览:")
        print(df.head())
        
        # 绘制图表
        plot_with_ma(df, symbol)
        
        # 简单的技术分析
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        ma60 = df['MA60'].iloc[-1]
        
        print("\n简单技术分析:")
        print(f"当前价格: {current_price:.2f}")
        print(f"20日均线: {ma20:.2f}")
        print(f"60日均线: {ma60:.2f}")
        
        if current_price > ma20 > ma60:
            print("技术面分析: 多头排列，可能处于上升趋势")
        elif current_price < ma20 < ma60:
            print("技术面分析: 空头排列，可能处于下降趋势")
        else:
            print("技术面分析: 趋势不明确")
    else:
        print("无法获取数据，请检查网络连接或代码是否正确")

if __name__ == "__main__":
    main()
