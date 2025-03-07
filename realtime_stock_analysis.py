#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中航光电(002179.SZ)实时股票分析工具
基于增强版机器学习预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
import warnings
from datetime import datetime, timedelta
import os
import argparse

# 忽略警告
warnings.filterwarnings('ignore')

# 命令行参数设置
parser = argparse.ArgumentParser(description='A股实时股票预测分析工具')
parser.add_argument('--token', help='Tushare API 令牌')
parser.add_argument('--stock', default='002179.SZ', help='股票代码 (默认: 002179.SZ)')
parser.add_argument('--model', default='model_rf.pkl', help='模型文件路径 (默认: model_rf.pkl)')
parser.add_argument('--scaler', default='scaler.pkl', help='归一化器文件路径 (默认: scaler.pkl)')
parser.add_argument('--features', default='features.pkl', help='特征列表文件路径 (默认: features.pkl)')
parser.add_argument('--days', type=int, default=100, help='历史数据天数 (默认: 100)')
parser.add_argument('--manual', action='store_true', help='使用手动输入的实时数据 (而非API获取)')
args = parser.parse_args()

# 设置Tushare token
if args.token:
    os.environ['TUSHARE_TOKEN'] = args.token
elif os.environ.get('TUSHARE_TOKEN') is None:
    os.environ['TUSHARE_TOKEN'] = '83bedfeb742a6492aab917a5af5e2a33964aec6b56dda3513d761734'

# 股票代码
stock_code = args.stock

# 获取实时股票数据
def get_realtime_data(stock_code):
    """
    获取实时股票数据
    
    参数:
    stock_code: 股票代码
    
    返回:
    pandas.DataFrame: 包含实时数据的DataFrame
    """
    print(f"获取 {stock_code} 的实时行情数据...")
    
    try:
        # 使用tushare
        import tushare as ts
        token = os.environ.get('TUSHARE_TOKEN')
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 尝试获取实时行情
        try:
            # 使用pro接口
            today = datetime.now().strftime('%Y%m%d')
            df = pro.daily(ts_code=stock_code, start_date=today, end_date=today)
            
            if df is not None and not df.empty:
                print("成功获取今日数据")
                return df
            else:
                # 如果pro接口没有当天数据，尝试使用quote_ctx
                print("今日数据尚未更新，尝试使用实时行情...")
                df_rt = ts.get_realtime_quotes(stock_code.split('.')[0])
                
                if df_rt is not None and not df_rt.empty:
                    # 转换实时行情格式为日线数据格式
                    data = {
                        'ts_code': stock_code,
                        'trade_date': datetime.now().strftime('%Y%m%d'),
                        'open': float(df_rt['open'].iloc[0]),
                        'high': float(df_rt['high'].iloc[0]),
                        'low': float(df_rt['low'].iloc[0]),
                        'close': float(df_rt['price'].iloc[0]),
                        'pre_close': float(df_rt['pre_close'].iloc[0]),
                        'vol': float(df_rt['volume'].iloc[0]) / 100,  # 单位转换
                        'amount': float(df_rt['amount'].iloc[0]) / 1000  # 单位转换
                    }
                    
                    # 计算涨跌幅
                    data['pct_chg'] = (data['close'] - data['pre_close']) / data['pre_close'] * 100
                    
                    return pd.DataFrame([data])
                
        except Exception as e:
            print(f"获取实时行情出错: {str(e)}")
        
        print("无法获取实时数据，将使用手动输入模式...")
        return get_manual_realtime_data(stock_code)
    
    except Exception as e:
        print(f"获取实时数据失败: {str(e)}")
        print("将使用手动输入模式...")
        return get_manual_realtime_data(stock_code)

# 手动输入实时数据
def get_manual_realtime_data(stock_code):
    """
    手动输入实时数据
    
    参数:
    stock_code: 股票代码
    
    返回:
    pandas.DataFrame: 包含实时数据的DataFrame
    """
    print("\n请输入实时数据:")
    try:
        price = float(input("当前价格 (元): "))
        open_price = float(input("开盘价 (元): "))
        high_price = float(input("最高价 (元): "))
        low_price = float(input("最低价 (元): "))
        volume = float(input("成交量 (手): "))
        pre_close = float(input("昨收价 (元): "))
        amount = float(input("成交额 (元, 可选): ") or volume * price * 100)
        
        latest = pd.DataFrame({
            'ts_code': [stock_code],
            'trade_date': [datetime.now().strftime('%Y%m%d')],
            'open': [open_price],
            'high': [high_price],
            'low': [low_price],
            'close': [price],
            'vol': [volume],
            'amount': [amount],
            'pct_chg': [price / pre_close * 100 - 100]
        })
        
        print(f"手动输入的实时数据: 价格 {price} 元, 成交量 {volume} 手")
        return latest
    except ValueError:
        print("输入无效，请确保所有输入都是数字。")
        return None

# 主函数
def main():
    """主函数"""
    global stock_code
    print(f"\n===== A股实时分析: {stock_code} =====")
    
    try:
        # 获取股票信息（名称等）
        import tushare as ts
        token = os.environ.get('TUSHARE_TOKEN')
        ts.set_token(token)
        pro = ts.pro_api()
        
        stock_info = pro.stock_basic(ts_code=stock_code, fields='name')
        stock_name = stock_info['name'].iloc[0] if not stock_info.empty else "未知"
        print(f"股票名称: {stock_name}")
        
        # 载入模型和缩放器
        model, scaler, features_list = load_models()
        
        # 获取历史数据
        df = get_stock_historical_data(stock_code, args.days, token)
        if df is None or df.empty:
            print("获取历史数据失败，程序退出")
            return
        
        # 获取实时数据
        if args.manual:
            real_data = get_manual_realtime_data(stock_code)
        else:
            real_data = get_realtime_data(stock_code)
        
        if real_data is None or real_data.empty:
            print("获取实时数据失败，程序退出")
            return
        
        # 合并历史和实时数据并预处理
        combined_data = preprocess_data(df, real_data)
        
        # 特征工程
        features_df = calculate_features(combined_data)
        
        # 预测
        signal, proba = predict_signal(features_df, model, scaler, features_list)
        
        # 解读结果
        interpret_results(combined_data, features_df, signal, proba)
        
        # 可视化
        visualize_results(combined_data, features_df, signal)
    
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
