#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易示例脚本 - 简化版，不依赖TA-Lib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime
import os

def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    try:
        # 使用AKShare获取股票数据
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
        return stock_df
    except Exception as e:
        print(f"获取股票数据出错: {e}")
        return None

def calculate_ma(df, ma_list=[5, 10, 20, 30, 60]):
    """计算移动平均线"""
    for ma in ma_list:
        df[f'MA{ma}'] = df['收盘'].rolling(ma).mean()
    return df

def plot_stock_with_ma(df, stock_code, title=None):
    """绘制股票走势图与移动平均线"""
    plt.figure(figsize=(15, 8))
    plt.plot(df['日期'], df['收盘'], label='收盘价')
    
    # 绘制移动平均线
    for col in df.columns:
        if col.startswith('MA'):
            plt.plot(df['日期'], df[col], label=col)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"{stock_code} 股价走势与均线")
    
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{stock_code}_analysis.png')
    plt.show()

def main():
    # 设置参数
    stock_code = "600519"  # 贵州茅台
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
    
    print(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据")
    
    # 获取数据
    stock_df = get_stock_data(stock_code, start_date, end_date)
    
    if stock_df is not None:
        print(f"成功获取数据，共 {len(stock_df)} 条记录")
        
        # 计算移动平均线
        stock_df = calculate_ma(stock_df)
        
        # 打印数据概览
        print("\n数据概览:")
        print(stock_df.head())
        
        # 绘制图表
        plot_stock_with_ma(stock_df, stock_code)
        
        # 简单的技术分析
        current_price = stock_df['收盘'].iloc[-1]
        ma20 = stock_df['MA20'].iloc[-1]
        ma60 = stock_df['MA60'].iloc[-1]
        
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
        print("无法获取股票数据，请检查网络连接或股票代码是否正确")

if __name__ == "__main__":
    main()
