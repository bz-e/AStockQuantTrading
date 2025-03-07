#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易示例脚本 - Baostock版本
不需要token，可直接使用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import baostock as bs
import datetime
import os

def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    try:
        # 登录Baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f'Baostock登录失败: {lg.error_msg}')
            return None
        
        # 获取股票日线数据
        rs = bs.query_history_k_data_plus(
            code=stock_code,
            fields="date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",  # 日线
            adjustflag="2"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f'查询股票数据失败: {rs.error_msg}')
            bs.logout()
            return None
        
        # 将数据转换为DataFrame
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 登出系统
        bs.logout()
        
        if df.empty:
            print("获取到的数据为空")
            return None
        
        # 数据类型转换
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # 统一列名称
        df = df.rename(columns={
            'date': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
            'amount': '成交额'
        })
        
        return df
    except Exception as e:
        print(f"获取股票数据出错: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

def calculate_ma(df, ma_list=[5, 10, 20, 30, 60]):
    """计算移动平均线"""
    for ma in ma_list:
        df[f'MA{ma}'] = df['收盘'].rolling(ma).mean()
    return df

def calculate_rsi(df, periods=14):
    """计算RSI指标"""
    delta = df['收盘'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    return df

def plot_with_indicators(df, stock_code, title=None):
    """绘制股价走势、移动平均线和RSI指标"""
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制股价和均线
    ax1.plot(df['日期'], df['收盘'], label='收盘价', linewidth=2)
    
    # 绘制移动平均线
    for col in df.columns:
        if col.startswith('MA'):
            ax1.plot(df['日期'], df[col], label=col, alpha=0.7)
    
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title(f"{stock_code} 股价走势与技术指标", fontsize=14)
    
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # 绘制RSI指标
    ax2.plot(df['日期'], df['RSI'], color='purple', label='RSI(14)')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(df['日期'], df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
    ax2.fill_between(df['日期'], df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
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
    plt.savefig(f'images/{stock_code}_analysis.png')
    plt.show()

def get_stock_info(stock_code):
    """获取股票基本信息"""
    try:
        # 登录Baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f'Baostock登录失败: {lg.error_msg}')
            return None
        
        # 查询股票基本信息
        rs = bs.query_stock_basic(code=stock_code)
        if rs.error_code != '0':
            print(f'查询股票基本信息失败: {rs.error_msg}')
            bs.logout()
            return None
        
        # 处理返回的结果
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
        
        # 登出系统
        bs.logout()
        
        if data_list:
            return pd.DataFrame(data_list, columns=rs.fields)
        else:
            print("未找到股票基本信息")
            return None
    except Exception as e:
        print(f"获取股票基本信息出错: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

def main():
    # 设置参数
    stock_code = "sh.600519"  # 贵州茅台，Baostock的代码格式为sh.开头或sz.开头
    today = datetime.datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据")
    
    # 获取股票基本信息
    stock_info = get_stock_info(stock_code)
    if stock_info is not None:
        print("\n股票基本信息:")
        print(stock_info)
    
    # 获取股票数据
    stock_df = get_stock_data(stock_code, start_date, end_date)
    
    if stock_df is not None and not stock_df.empty:
        print(f"成功获取数据，共 {len(stock_df)} 条记录")
        
        # 计算技术指标
        stock_df = calculate_ma(stock_df)
        stock_df = calculate_rsi(stock_df)
        
        # 打印数据概览
        print("\n数据概览:")
        print(stock_df.head())
        
        # 绘制图表
        plot_with_indicators(stock_df, stock_code)
        
        # 简单的技术分析
        current_price = stock_df['收盘'].iloc[-1]
        ma20 = stock_df['MA20'].iloc[-1]
        ma60 = stock_df['MA60'].iloc[-1]
        current_rsi = stock_df['RSI'].iloc[-1]
        
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
    else:
        print("无法获取股票数据，请检查网络连接或股票代码是否正确")
        print("\nBaostock股票代码格式说明:")
        print("上海证券交易所股票：sh.股票代码，如sh.600519")
        print("深圳证券交易所股票：sz.股票代码，如sz.000001")

if __name__ == "__main__":
    main()
