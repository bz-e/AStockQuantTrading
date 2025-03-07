#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易 - 成交量分析工具
包含5种基于成交量的分析策略：
1. VWMA (成交量加权均线)
2. 成交量突破策略
3. OBV (能量潮指标)
4. RSI与成交量结合策略
5. 成交量异常检测

使用方法：
python volume_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import datetime
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# Tushare token配置
TUSHARE_TOKEN = "83bedfeb742a6492aab917a5af5e2a33964aec6b56dda3513d761734"

# 创建images目录（如果不存在）
if not os.path.exists('images'):
    os.makedirs('images')

def init_tushare():
    """初始化Tushare"""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()

def get_stock_data(pro, stock_code, start_date, end_date):
    """获取股票数据"""
    # 转换为tushare格式日期
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    
    try:
        # 获取日线数据
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        
        if df.empty:
            print(f"未获取到{stock_code}的数据，请检查日期范围或股票代码")
            return None
        
        # 按日期排序，从旧到新
        df = df.sort_values('trade_date')
        
        # 重命名列，便于后续处理
        df = df.rename(columns={
            'trade_date': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'pre_close': 'pre_close',
            'vol': '成交量',  # 注意tushare的vol单位是手（100股）
            'amount': '成交额'
        })
        
        # 将成交量转换为股
        df['成交量'] = df['成交量'] * 100
        
        print(f"成功获取数据，共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None

# 1. 成交量加权均线策略 (VWMA)
def calculate_vwma(df, period=20):
    """计算成交量加权均线"""
    df['VWMA'] = (df['收盘'] * df['成交量']).rolling(window=period).sum() / df['成交量'].rolling(window=period).sum()
    return df

def vwma_signals(df):
    """生成VWMA信号"""
    df['VWMA_Signal'] = 0
    # 上穿VWMA，买入信号
    df.loc[df['收盘'] > df['VWMA'], 'VWMA_Signal'] = 1
    # 下穿VWMA，卖出信号
    df.loc[df['收盘'] < df['VWMA'], 'VWMA_Signal'] = -1
    return df

# 2. 成交量突破策略
def calculate_volume_breakout(df, period=20):
    """计算成交量突破策略"""
    # 计算期间最高价、最低价
    df['最高_' + str(period)] = df['最高'].rolling(window=period).max()
    df['最低_' + str(period)] = df['最低'].rolling(window=period).min()
    # 计算成交量均值
    df['成交量均值'] = df['成交量'].rolling(window=period).mean()
    return df

def volume_breakout_signals(df):
    """生成成交量突破信号"""
    df['Volume_Breakout_Signal'] = 0
    # 价格突破高点且成交量大于均值，买入信号
    df.loc[(df['收盘'] > df['最高_20'].shift(1)) & (df['成交量'] > df['成交量均值']), 'Volume_Breakout_Signal'] = 1
    # 价格跌破低点且成交量大于均值，卖出信号
    df.loc[(df['收盘'] < df['最低_20'].shift(1)) & (df['成交量'] > df['成交量均值']), 'Volume_Breakout_Signal'] = -1
    return df

# 3. OBV指标
def calculate_obv(df):
    """计算OBV指标"""
    df['价格变动'] = df['收盘'].diff()
    # 初始化OBV列
    df['OBV'] = 0
    
    # 首日OBV设为0
    if len(df) > 0:
        df.iloc[0, df.columns.get_indexer(['OBV'])[0]] = 0
        
    # 计算OBV
    for i in range(1, len(df)):
        if df['价格变动'].iloc[i] > 0:  # 价格上涨
            df.iloc[i, df.columns.get_indexer(['OBV'])[0]] = df['OBV'].iloc[i-1] + df['成交量'].iloc[i]
        elif df['价格变动'].iloc[i] < 0:  # 价格下跌
            df.iloc[i, df.columns.get_indexer(['OBV'])[0]] = df['OBV'].iloc[i-1] - df['成交量'].iloc[i]
        else:  # 价格不变
            df.iloc[i, df.columns.get_indexer(['OBV'])[0]] = df['OBV'].iloc[i-1]
    
    return df

def obv_signals(df):
    """生成OBV信号"""
    df['OBV_Signal'] = 0
    # OBV上涨且价格上涨，买入信号
    df.loc[(df['收盘'] > df['收盘'].shift(1)) & (df['OBV'] > df['OBV'].shift(1)), 'OBV_Signal'] = 1
    # OBV下跌且价格下跌，卖出信号
    df.loc[(df['收盘'] < df['收盘'].shift(1)) & (df['OBV'] < df['OBV'].shift(1)), 'OBV_Signal'] = -1
    return df

# 4. RSI与成交量结合策略
def calculate_rsi(df, period=14):
    """计算RSI指标"""
    delta = df['收盘'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def rsi_volume_signals(df):
    """生成RSI与成交量结合的信号"""
    df['RSI_Volume_Signal'] = 0
    # RSI低于30且成交量大于均值，买入信号
    df.loc[(df['RSI'] < 30) & (df['成交量'] > df['成交量均值']), 'RSI_Volume_Signal'] = 1
    # RSI高于70且成交量大于均值，卖出信号
    df.loc[(df['RSI'] > 70) & (df['成交量'] > df['成交量均值']), 'RSI_Volume_Signal'] = -1
    return df

# 5. 成交量异常检测策略
def volume_anomaly_signals(df, threshold=2):
    """生成成交量异常信号"""
    # 计算价格变化百分比
    df['价格变化率'] = df['收盘'].pct_change()
    # 成交量是否超过均值的threshold倍
    df['成交量异常'] = df['成交量'] > threshold * df['成交量均值']
    
    df['Volume_Anomaly_Signal'] = 0
    # 成交量异常且价格上涨，买入信号
    df.loc[(df['成交量异常']) & (df['价格变化率'] > 0), 'Volume_Anomaly_Signal'] = 1
    # 成交量异常且价格下跌，卖出信号
    df.loc[(df['成交量异常']) & (df['价格变化率'] < 0), 'Volume_Anomaly_Signal'] = -1
    return df

def plot_volume_analysis(df, stock_code):
    """绘制成交量分析图表"""
    plt.figure(figsize=(14, 14))
    
    # 价格和VWMA子图
    plt.subplot(5, 1, 1)
    plt.plot(df['日期'], df['收盘'], label='收盘价')
    plt.plot(df['日期'], df['VWMA'], label='VWMA')
    plt.title(f'{stock_code} 价格与VWMA')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 成交量和成交量均值子图
    plt.subplot(5, 1, 2)
    plt.bar(df['日期'], df['成交量'], label='成交量')
    plt.plot(df['日期'], df['成交量均值'], color='red', label='成交量均值')
    plt.title('成交量分析')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # OBV子图
    plt.subplot(5, 1, 3)
    plt.plot(df['日期'], df['OBV'], label='OBV')
    plt.title('OBV能量潮')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RSI子图
    plt.subplot(5, 1, 4)
    plt.plot(df['日期'], df['RSI'], label='RSI')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.3)
    plt.title('RSI指标')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 信号汇总子图
    plt.subplot(5, 1, 5)
    
    # 汇总所有信号
    df['Signal_Sum'] = (df['VWMA_Signal'] + df['Volume_Breakout_Signal'] + 
                        df['OBV_Signal'] + df['RSI_Volume_Signal'] + 
                        df['Volume_Anomaly_Signal'])
    
    # 绘制汇总信号
    plt.bar(df['日期'], df['Signal_Sum'], label='信号汇总')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('信号汇总 (正值:买入，负值:卖出)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/{stock_code}_volume_analysis.png')
    plt.close()

def generate_volume_analysis_signals(df):
    """生成所有成交量分析信号，并返回综合建议"""
    # 确保数据按日期排序
    df = df.sort_values('日期')
    
    # 应用所有成交量分析策略
    df = calculate_vwma(df)
    df = calculate_volume_breakout(df)
    df = calculate_obv(df)
    df = calculate_rsi(df)
    
    # 生成各策略信号
    df = vwma_signals(df)
    df = volume_breakout_signals(df)
    df = obv_signals(df)
    df = rsi_volume_signals(df)
    df = volume_anomaly_signals(df)
    
    # 汇总所有信号
    df['Signal_Sum'] = (df['VWMA_Signal'] + df['Volume_Breakout_Signal'] + 
                       df['OBV_Signal'] + df['RSI_Volume_Signal'] + 
                       df['Volume_Anomaly_Signal'])
    
    # 最新的综合信号
    latest_signal = df['Signal_Sum'].iloc[-1] if not df.empty else 0
    
    return df, latest_signal

def print_analysis_summary(df, stock_code):
    """打印分析摘要"""
    if df.empty:
        print("没有足够的数据进行分析")
        return
    
    latest_data = df.iloc[-1]
    
    print("\n======== 成交量分析摘要 ========")
    print(f"股票代码: {stock_code}")
    print(f"日期: {latest_data['日期']}")
    print(f"收盘价: {latest_data['收盘']:.2f}")
    print(f"VWMA(20): {latest_data['VWMA']:.2f}")
    print(f"RSI(14): {latest_data['RSI']:.2f}")
    print(f"成交量: {latest_data['成交量']:,.0f}")
    print(f"成交量均值: {latest_data['成交量均值']:,.0f}")
    print(f"成交量/均值: {latest_data['成交量']/latest_data['成交量均值']:.2f}倍")
    
    print("\n======== 信号分析 ========")
    print(f"VWMA信号: {'买入' if latest_data['VWMA_Signal'] > 0 else ('卖出' if latest_data['VWMA_Signal'] < 0 else '中性')}")
    print(f"成交量突破信号: {'买入' if latest_data['Volume_Breakout_Signal'] > 0 else ('卖出' if latest_data['Volume_Breakout_Signal'] < 0 else '中性')}")
    print(f"OBV信号: {'买入' if latest_data['OBV_Signal'] > 0 else ('卖出' if latest_data['OBV_Signal'] < 0 else '中性')}")
    print(f"RSI+成交量信号: {'买入' if latest_data['RSI_Volume_Signal'] > 0 else ('卖出' if latest_data['RSI_Volume_Signal'] < 0 else '中性')}")
    print(f"成交量异常信号: {'买入' if latest_data['Volume_Anomaly_Signal'] > 0 else ('卖出' if latest_data['Volume_Anomaly_Signal'] < 0 else '中性')}")
    
    print("\n======== 综合建议 ========")
    signal_sum = latest_data['Signal_Sum']
    if signal_sum >= 3:
        recommendation = "强烈买入 - 多重成交量指标确认买入信号"
    elif signal_sum >= 1:
        recommendation = "谨慎买入 - 部分成交量指标显示买入信号"
    elif signal_sum <= -3:
        recommendation = "强烈卖出 - 多重成交量指标确认卖出信号"
    elif signal_sum <= -1:
        recommendation = "谨慎卖出 - 部分成交量指标显示卖出信号"
    else:
        recommendation = "观望 - 没有明确的买入或卖出信号"
    
    print(f"信号汇总: {signal_sum}")
    print(f"建议: {recommendation}")
    
    print("\n======== 各指标最新数据 ========")
    # 显示最近5天数据，只选取关键列
    key_columns = ['日期', '收盘', '成交量', 'VWMA', 'RSI', 'OBV', 'Signal_Sum']
    print(df[key_columns].tail(5))

def main():
    """主函数"""
    # 初始化tushare
    pro = init_tushare()
    if not pro:
        print("初始化Tushare失败")
        return
    
    # 设置股票代码和日期范围
    stock_code = "002179.SZ"  # 中航光电
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')  # 半年数据
    
    print(f"获取 {stock_code} 从 {start_date} 到 {today} 的数据")
    
    # 获取股票数据
    df = get_stock_data(pro, stock_code, start_date, today)
    if df is None or df.empty:
        print("未能获取有效数据，退出分析")
        return
    
    # 打印数据概览
    print("\n数据概览:")
    print(df.head(5))
    
    # 生成所有成交量分析信号
    df, latest_signal = generate_volume_analysis_signals(df)
    
    # 绘制图表
    plot_volume_analysis(df, stock_code)
    
    # 打印分析摘要
    print_analysis_summary(df, stock_code)
    
    print(f"\n分析完成！图表已保存至 images/{stock_code}_volume_analysis.png")

if __name__ == "__main__":
    main()
