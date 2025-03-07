#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities Module - Responsible for auxiliary functions and trading signal generation

This module implements various auxiliary functions, including trading signal generation,
performance evaluation, backtesting simulation, etc.
"""

import numpy as np
import pandas as pd
import os
import logging
import json
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_prediction')


def generate_trading_signals(df, predictions, threshold=0.5, consecutive_days=1):
    """
    Generate trading signals based on prediction results
    
    Parameters:
        df (pandas.DataFrame): Original stock data
        predictions (numpy.ndarray): Prediction results array
        threshold (float): Signal trigger threshold
        consecutive_days (int): Required consecutive prediction days
    
    Returns:
        pandas.DataFrame: DataFrame with added trading signals
    """
    try:
        if df is None or len(df) == 0:
            logger.error("Empty data provided")
            return None
            
        if predictions is None or len(predictions) == 0:
            logger.error("Empty predictions")
            return None
            
        if len(df) != len(predictions):
            logger.error(f"Data length mismatch: df={len(df)}, predictions={len(predictions)}")
            return None
            
        # Copy the DataFrame to avoid modifying the original data
        result_df = df.copy()
        
        # Add prediction results
        result_df['prediction'] = predictions
        
        # Initialize signals
        result_df['signal'] = 0  # 0: No operation, 1: Buy, -1: Sell
        
        # Consecutive up predictions as buy signals
        for i in range(consecutive_days - 1, len(result_df)):
            # Check if the past consecutive_days days have consecutive up predictions
            if all(result_df['prediction'].iloc[i-consecutive_days+1:i+1] >= threshold):
                result_df.loc[result_df.index[i], 'signal'] = 1
                
            # Check if the past consecutive_days days have consecutive down predictions
            elif all(result_df['prediction'].iloc[i-consecutive_days+1:i+1] < threshold):
                result_df.loc[result_df.index[i], 'signal'] = -1
                
        logger.info(f"Generated trading signals: Buy {sum(result_df['signal'] == 1)} times, Sell {sum(result_df['signal'] == -1)} times")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        return None


def simulate_trading(df, initial_capital=100000.0, commission_rate=0.0003, slippage=0.001):
    """
    Simulate trading and calculate returns
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing prices and signals
        initial_capital (float): Initial capital
        commission_rate (float): Commission rate
        slippage (float): Slippage rate
    
    Returns:
        tuple: (Total return, Return rate, Trading records)
    """
    try:
        if df is None or len(df) == 0:
            logger.error("Empty data provided")
            return 0, 0, pd.DataFrame()
            
        if 'signal' not in df.columns:
            logger.error("No signal column in data")
            return 0, 0, pd.DataFrame()
            
        if 'close' not in df.columns:
            logger.error("No close price column in data")
            return 0, 0, pd.DataFrame()
            
        # Copy the DataFrame
        backtest_df = df.copy()
        
        # Initialize account status
        capital = initial_capital
        position = 0
        trades = []
        
        # Simulate trading
        for i in range(len(backtest_df)):
            date = backtest_df.index[i] if isinstance(backtest_df.index[i], datetime) else backtest_df['date'].iloc[i]
            price = backtest_df['close'].iloc[i]
            signal = backtest_df['signal'].iloc[i]
            
            # Buy signal
            if signal == 1 and position == 0:
                # Calculate the number of shares to buy (considering commission and slippage)
                buy_price = price * (1 + slippage)
                max_shares = int(capital / buy_price / (1 + commission_rate) / 100) * 100  # Buy in multiples of 100 shares
                
                if max_shares > 0:
                    cost = max_shares * buy_price
                    commission = cost * commission_rate
                    total_cost = cost + commission
                    
                    if total_cost <= capital:
                        position = max_shares
                        capital -= total_cost
                        
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': buy_price,
                            'shares': max_shares,
                            'commission': commission,
                            'value': cost,
                            'capital': capital
                        })
                        
                        logger.info(f"Buy: {date}, Price: {buy_price:.2f}, Shares: {max_shares}, Commission: {commission:.2f}, Remaining capital: {capital:.2f}")
                        
            # Sell signal
            elif signal == -1 and position > 0:
                # Calculate the sell revenue (considering commission and slippage)
                sell_price = price * (1 - slippage)
                revenue = position * sell_price
                commission = revenue * commission_rate
                net_revenue = revenue - commission
                
                capital += net_revenue
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': position,
                    'commission': commission,
                    'value': revenue,
                    'capital': capital
                })
                
                logger.info(f"Sell: {date}, Price: {sell_price:.2f}, Shares: {position}, Commission: {commission:.2f}, Remaining capital: {capital:.2f}")
                
                position = 0
                
        # Force close the position on the last day
        if position > 0:
            price = backtest_df['close'].iloc[-1] * (1 - slippage)
            revenue = position * price
            commission = revenue * commission_rate
            net_revenue = revenue - commission
            
            capital += net_revenue
            
            date = backtest_df.index[-1] if isinstance(backtest_df.index[-1], datetime) else backtest_df['date'].iloc[-1]
            
            trades.append({
                'date': date,
                'action': 'FINAL_SELL',
                'price': price,
                'shares': position,
                'commission': commission,
                'value': revenue,
                'capital': capital
            })
            
            logger.info(f"Final close: {date}, Price: {price:.2f}, Shares: {position}, Commission: {commission:.2f}, Remaining capital: {capital:.2f}")
            
        # Calculate total return and return rate
        profit = capital - initial_capital
        profit_rate = profit / initial_capital * 100
        
        # Create trading records DataFrame
        trades_df = pd.DataFrame(trades)
        
        logger.info(f"Trading simulation completed: Total return {profit:.2f}, Return rate {profit_rate:.2f}%, Number of trades {len(trades)}")
        
        return profit, profit_rate, trades_df
        
    except Exception as e:
        logger.error(f"Error simulating trading: {str(e)}")
        return 0, 0, pd.DataFrame()


def calculate_metrics(predictions, y_true):
    """
    计算预测性能评估指标，用于评估模型预测效果
    
    该函数计算多种分类评估指标，包括准确率、精确率、召回率、F1分数等，
    适用于二元分类问题（预测股价涨跌）。所有指标都基于混淆矩阵计算得出。
    
    参数:
        predictions (numpy.ndarray): 模型预测结果数组，形状为(n_samples,)。
                                  其中0代表预测下跌，1代表预测上涨。数组应
                                  只包含0和1值，表示二元分类结果而非概率值。
        
        y_true (numpy.ndarray): 真实标签数组，形状为(n_samples,)。
                              其中0代表实际下跌，1代表实际上涨。该数组
                              应与predictions数组长度一致。
    
    返回:
        dict: 包含以下评估指标的字典:
            - accuracy (float): 准确率，正确预测的样本比例，范围0-100(%)
            - precision (float): 精确率，预测为上涨且实际上涨的比例，范围0-1
            - recall (float): 召回率，实际上涨被正确预测的比例，范围0-1
            - f1 (float): F1分数，精确率和召回率的调和平均值，范围0-1
            - correct_up (int): 正确预测上涨的样本数量
            - correct_down (int): 正确预测下跌的样本数量
            - confusion_matrix (list): 混淆矩阵，2×2列表格式
                                      [[真实下跌预测下跌, 真实下跌预测上涨],
                                       [真实上涨预测下跌, 真实上涨预测上涨]]
    
    评估指标解释:
        - 准确率(Accuracy): 所有预测中正确的比例，但在类别不平衡时可能有误导性
        - 精确率(Precision): 预测为上涨的样本中，实际上涨的比例，反映模型预测上涨信号的可靠性
        - 召回率(Recall): 实际上涨的样本中，被成功预测的比例，反映模型捕获上涨机会的能力
        - F1分数: 精确率和召回率的调和平均，平衡考虑两个指标，适合评估整体性能
    
    示例:
        >>> # 假设我们有预测结果和真实标签
        >>> predictions = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        >>> metrics = calculate_metrics(predictions, y_true)
        >>> print(f"准确率: {metrics['accuracy']:.2f}%")
        >>> print(f"精确率: {metrics['precision']:.4f}")
        >>> print(f"召回率: {metrics['recall']:.4f}")
        >>> print(f"F1分数: {metrics['f1']:.4f}")
        >>> print(f"混淆矩阵:\n{metrics['confusion_matrix']}")
    
    注意:
        1. 该函数专为二元分类设计，预测值和真实值必须为0或1，不支持多类别或回归问题
        2. 当某一类别样本数为零时，相关的指标(如精确率或召回率)可能变为零，需谨慎解读
        3. 在极端类别不平衡的情况下，准确率可能会产生误导，应更关注精确率、召回率和F1分数
        4. 混淆矩阵对角线上的值(TP, TN)越高，非对角线值(FP, FN)越低，模型性能越好
        5. 在股票预测场景中，不同指标的重要性应基于交易策略权衡，例如:
           - 追求稳健策略: 关注精确率，降低错误买入信号
           - 追求高覆盖策略: 关注召回率，确保不错过上涨机会
    """
    try:
        if predictions is None or y_true is None:
            logger.error("Empty predictions or true labels")
            return {}
            
        if len(predictions) != len(y_true):
            logger.error(f"Length mismatch: predictions={len(predictions)}, y_true={len(y_true)}")
            return {}
            
        if len(predictions) == 0:
            logger.error("Empty data")
            return {}
            
        # Calculate accuracy
        accuracy = sum(predictions == y_true) / len(y_true) * 100
        
        # Calculate precision, recall, and F1 score
        tp = sum((predictions == 1) & (y_true == 1))
        fp = sum((predictions == 1) & (y_true == 0))
        fn = sum((predictions == 0) & (y_true == 1))
        tn = sum((predictions == 0) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate correct up and down predictions
        correct_up = tp
        correct_down = tn
        
        # Calculate confusion matrix
        cm = [[tn, fp], [fn, tp]]
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct_up': correct_up,
            'correct_down': correct_down,
            'confusion_matrix': cm
        }
        
        logger.info(f"Calculated performance metrics: Accuracy={accuracy:.2f}%, F1 score={f1:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {}


def save_results(results, filepath, append=False):
    """
    Save prediction results to file
    
    Parameters:
        results (dict): Results dictionary
        filepath (str): File path to save
        append (bool): Whether to append to existing file
    
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Create directory if it does not exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Format results for storage
        formatted_results = {k: v for k, v in results.items() if isinstance(v, (str, int, float, bool, list, dict))}
        
        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
                
        formatted_results = convert_numpy(formatted_results)
        
        # Add timestamp
        formatted_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            if append:
                f.write('\n')
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False


def create_stop_loss_profit_strategy(predictions, close_prices, stop_loss_pct=0.05, take_profit_pct=0.1):
    """
    创建止损止盈交易策略，基于预测结果和价格数据生成交易信号
    
    该函数根据模型预测结果，结合止损和止盈条件，生成完整的交易信号序列。
    策略会在预测上涨时产生买入信号，并在触发止损、止盈或反转信号时产生卖出信号，
    适用于自动化交易系统或回测分析。
    
    参数:
        predictions (numpy.ndarray): 模型预测结果数组，形状为(n_samples,)。
                                   应为二元分类结果，其中0表示预测下跌，1表示
                                   预测上涨。数组顺序应与close_prices一致，
                                   确保时间对齐。
        
        close_prices (numpy.ndarray): 收盘价数组，形状为(n_samples,)。
                                    必须与predictions具有相同长度，且顺序一致，
                                    表示每个交易日的收盘价。
        
        stop_loss_pct (float): 止损百分比，默认为0.05(5%)。当持仓后价格下跌
                              超过此比例时触发止损卖出。较小的值意味着更严格的
                              风险控制，但可能导致更频繁的交易。范围通常为0.01-0.10。
        
        take_profit_pct (float): 止盈百分比，默认为0.1(10%)。当持仓后价格上涨
                                超过此比例时触发止盈卖出。较大的值意味着更宽松的
                                利润目标，可能让盈利持续更长时间。范围通常为0.05-0.30。
    
    返回:
        pandas.DataFrame: 包含以下列的DataFrame:
            - close: 收盘价
            - prediction: 原始预测结果(0或1)
            - signal: 交易信号，其中:
                      0=持仓不变(无操作)
                      1=买入信号
                      -1=卖出信号
            - exit_reason: 卖出原因(仅当signal=-1时存在)，可能的值包括:
                          'stop_loss'=止损触发
                          'take_profit'=止盈触发
                          'reverse_signal'=预测信号反转
    
    策略逻辑:
        1. 初始状态: 未持仓，等待买入信号
        2. 买入条件: 预测结果为上涨(1)且当前未持仓
        3. 卖出条件(以下任一条件满足时):
           a. 止损: 价格下跌超过买入价的stop_loss_pct
           b. 止盈: 价格上涨超过买入价的take_profit_pct
           c. 反转信号: 预测由上涨变为下跌
        4. 买入后会记录入场价格，并计算止损和止盈目标价位
        5. 每个交易日检查是否满足上述条件，并相应更新signal值
    
    示例:
        >>> # 假设已有预测结果和收盘价数据
        >>> predictions = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        >>> close_prices = np.array([100, 99, 101, 105, 103, 101, 98, 100, 106, 104])
        >>> 
        >>> # 创建更严格的止损策略(止损3%，止盈15%)
        >>> df = create_stop_loss_profit_strategy(
        ...     predictions=predictions,
        ...     close_prices=close_prices,
        ...     stop_loss_pct=0.03,
        ...     take_profit_pct=0.15
        ... )
        >>> 
        >>> # 查看生成的交易信号
        >>> print(df[['close', 'prediction', 'signal', 'exit_reason']])
        >>> 
        >>> # 统计各类信号
        >>> print(f"买入信号数量: {sum(df['signal'] == 1)}")
        >>> print(f"止损卖出次数: {sum((df['signal'] == -1) & (df['exit_reason'] == 'stop_loss'))}")
        >>> print(f"止盈卖出次数: {sum((df['signal'] == -1) & (df['exit_reason'] == 'take_profit'))}")
        >>> print(f"信号反转卖出次数: {sum((df['signal'] == -1) & (df['exit_reason'] == 'reverse_signal'))}")
    
    注意:
        1. 该策略基于历史数据和预测结果，实际交易中可能面临滑点、流动性等额外因素影响
        2. 止损止盈参数应根据标的波动性、持仓时间和风险偏好调整，不存在通用的最佳参数
        3. 函数假设价格和预测按时间顺序排列，并且是每日频率数据；不同频率数据可能需要调整逻辑
        4. 返回的DataFrame不会移除预测值为NaN的行，如有必要需在调用前预处理数据
        5. 该策略不考虑交易量和资金管理，实际应用时应结合资金管理策略使用
        6. 对于多个标的的组合策略，建议分别计算信号再综合考虑仓位分配
    """
    try:
        if predictions is None or len(predictions) == 0:
            logger.error("Empty predictions")
            return None
            
        if close_prices is None or len(close_prices) == 0:
            logger.error("Empty close prices")
            return None
            
        if len(predictions) != len(close_prices):
            logger.error(f"Length mismatch: predictions={len(predictions)}, close_prices={len(close_prices)}")
            return None
            
        # Create result DataFrame
        df = pd.DataFrame({
            'close': close_prices,
            'prediction': predictions,
            'signal': 0  # Initialize signal
        })
        
        # Initialize variables
        in_position = False
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        # Daily analysis
        for i in range(len(df)):
            # If not in position and prediction is up
            if not in_position and df['prediction'].iloc[i] == 1:
                # Buy signal
                df.loc[df.index[i], 'signal'] = 1
                in_position = True
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
            # If in position
            elif in_position:
                current_price = df['close'].iloc[i]
                
                # Check stop-loss condition
                if current_price <= stop_loss:
                    df.loc[df.index[i], 'signal'] = -1  # Trigger stop-loss
                    df.loc[df.index[i], 'exit_reason'] = 'stop_loss'
                    in_position = False
                    
                # Check take-profit condition
                elif current_price >= take_profit:
                    df.loc[df.index[i], 'signal'] = -1  # Trigger take-profit
                    df.loc[df.index[i], 'exit_reason'] = 'take_profit'
                    in_position = False
                    
                # Check reverse signal
                elif df['prediction'].iloc[i] == 0:
                    df.loc[df.index[i], 'signal'] = -1  # Prediction is down, sell
                    df.loc[df.index[i], 'exit_reason'] = 'reverse_signal'
                    in_position = False
                    
        # Calculate stop-loss and take-profit trigger counts
        stop_loss_count = sum((df['signal'] == -1) & (df['exit_reason'] == 'stop_loss'))
        take_profit_count = sum((df['signal'] == -1) & (df['exit_reason'] == 'take_profit'))
        reverse_count = sum((df['signal'] == -1) & (df['exit_reason'] == 'reverse_signal'))
        
        logger.info(f"Stop-loss and take-profit strategy: Stop-loss count={stop_loss_count}, Take-profit count={take_profit_count}, Reverse signal count={reverse_count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating stop-loss and take-profit strategy: {str(e)}")
        return None


def apply_stop_loss_take_profit(signals, close_prices, stop_loss=0.05, take_profit=0.1):
    """
    Apply stop-loss and take-profit strategy to existing trading signals
    
    Parameters:
        signals (numpy.ndarray): Array of trading signals where 1=buy, -1=sell, 0=hold
        close_prices (numpy.ndarray): Array of close prices corresponding to signals
        stop_loss (float): Stop loss percentage (default: 0.05 or 5%)
        take_profit (float): Take profit percentage (default: 0.1 or 10%)
        
    Returns:
        numpy.ndarray: Modified signals array with stop-loss and take-profit applied
    """
    try:
        if len(signals) != len(close_prices):
            raise ValueError(f"Length mismatch: signals={len(signals)}, close_prices={len(close_prices)}")
        
        # Make a copy of the signals to avoid modifying the original array
        modified_signals = signals.copy()
        
        # Keep track of the position status and entry price
        in_position = False
        entry_price = 0
        
        for i in range(len(signals)):
            # Current price
            current_price = close_prices[i]
            
            # If not in position and we have a buy signal
            if not in_position and signals[i] == 1:
                in_position = True
                entry_price = current_price
                # Keep the original buy signal
                
            # If in position, check for stop loss or take profit
            elif in_position:
                # Calculate price change since entry
                price_change = (current_price - entry_price) / entry_price
                
                # Stop loss triggered
                if price_change <= -stop_loss:
                    modified_signals[i] = -1  # Sell signal
                    in_position = False
                
                # Take profit triggered
                elif price_change >= take_profit:
                    modified_signals[i] = -1  # Sell signal
                    in_position = False
                
                # Original sell signal
                elif signals[i] == -1:
                    # Keep the original sell signal
                    in_position = False
        
        return modified_signals
        
    except Exception as e:
        logging.error(f"Error applying stop-loss and take-profit: {str(e)}")
        return signals  # Return original signals on error


def format_time(seconds):
    """
    Format time in human-readable format
    
    Parameters:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def generate_advanced_analysis(predictions, metrics, model_info, stock_code, args, trading_signals=None, returns_eval=None):
    """
    生成高级分析解读，提供详细的模型预测解释和交易建议
    
    参数:
        predictions (list/array): 预测结果
        metrics (dict): 模型评估指标
        model_info (dict/object): 模型相关信息
        stock_code (str): 股票代码
        args (object): 程序参数
        trading_signals (list/array, optional): 交易信号
        returns_eval (dict, optional): 收益评估结果
        
    返回:
        str: 详细的分析报告文本
    """
    # 保证预测结果是列表类型
    if predictions is not None and hasattr(predictions, 'tolist'):
        recent_preds = predictions[-args.predict_days:].tolist()
    elif predictions is not None:
        recent_preds = predictions[-args.predict_days:]
    else:
        recent_preds = []
    
    # 分析预测趋势
    if len(recent_preds) > 0:
        up_days = sum(1 for p in recent_preds if p > 0)
        down_days = sum(1 for p in recent_preds if p < 0)
        flat_days = len(recent_preds) - up_days - down_days
        
        # 计算信号强度
        signal_strength = sum(abs(p) for p in recent_preds) / len(recent_preds)
        
        # 确定总体趋势
        if up_days > down_days:
            trend = "上涨"
        elif down_days > up_days:
            trend = "下跌"
        else:
            trend = "震荡"
            
        # 强度评级
        if signal_strength > 0.015:
            strength = "强"
        elif signal_strength > 0.005:
            strength = "中"
        else:
            strength = "弱"
    else:
        return "预测数据不足，无法生成详细分析。"
    
    # 构建高级分析报告
    analysis = []
    analysis.append(f"## 🔍 {stock_code}详细分析解读\n")
    
    # 1. 模型预测强度分析
    analysis.append("### 1️⃣ 模型预测强度分析\n")
    accuracy = metrics.get('accuracy', 0) * 100
    analysis.append(f"- **预测准确率**: 模型整体准确率达到{accuracy:.2f}%，")
    
    if accuracy > 70:
        analysis.append("这在股票预测中属于非常高的水平，大大提高了策略可信度")
    elif accuracy > 60:
        analysis.append("这在股票预测中属于较好的水平（考虑到市场随机性）")
    elif accuracy > 50:
        analysis.append("这在股票预测中属于可接受的水平")
    else:
        analysis.append("这在股票预测中准确率偏低，建议谨慎参考")
    
    # 分析连续性信号
    if up_days == len(recent_preds):
        analysis.append(f"- **连续上涨信号**: 未来{args.predict_days}天全部预测为上涨走势，这种一致性信号非常罕见")
    elif down_days == len(recent_preds):
        analysis.append(f"- **连续下跌信号**: 未来{args.predict_days}天全部预测为下跌走势，这种一致性信号非常罕见")
    else:
        major_trend = "上涨" if up_days > down_days else "下跌"
        major_days = max(up_days, down_days)
        analysis.append(f"- **趋势信号**: 未来{args.predict_days}天中有{major_days}天预测为{major_trend}走势")
    
    # 信号强度评级
    analysis.append(f"- **信号强度评级**: 信号强度为\"{strength}\"级别，")
    if strength == "强":
        analysis.append("这意味着预测的平均涨跌幅相对较高")
    elif strength == "中":
        analysis.append("这表示预测的平均涨跌幅处于中等水平")
    else:
        analysis.append("这表示预测的平均涨跌幅较小")
    
    # 集成模型信息
    if hasattr(args, 'ensemble') and args.ensemble:
        analysis.append("- **集成模型决策**: 本预测结果来自多模型集成方法，")
        if hasattr(model_info, 'estimator_names_') and len(model_info.estimator_names_) > 0:
            model_names = ', '.join(model_info.estimator_names_)
            analysis.append(f"- 包含了{model_names}等算法的'投票'结果，这种集成方法通常比单一模型更可靠")
        else:
            analysis.append("汇集了多种算法的预测结果，通常比单一模型更可靠")
            
    # 2. 技术指标推断
    analysis.append("\n### 2️⃣ 技术指标佐证\n")
    analysis.append("基于模型训练过程和预测结果，推断可能的技术指标状态：\n")
    
    if trend == "上涨" and strength == "强":
        analysis.append("""
- **相对强弱指标(RSI)**: 可能处于上升趋势但未达到过度买入区域
- **MACD指标**: 可能出现金叉或正在形成多头格局
- **成交量变化**: 可能伴随着上涨信号的放量迹象
- **布林带位置**: 价格可能突破中轨向上轨运行
""")
    elif trend == "下跌" and strength == "强":
        analysis.append("""
- **相对强弱指标(RSI)**: 可能处于下降趋势但未达到过度卖出区域
- **MACD指标**: 可能出现死叉或正在形成空头格局
- **成交量变化**: 可能伴随着下跌信号的放量迹象
- **布林带位置**: 价格可能突破中轨向下轨运行
""")
    elif trend == "震荡":
        analysis.append("""
- **相对强弱指标(RSI)**: 可能在中性区域波动
- **MACD指标**: 可能在零轴附近波动，无明显趋势
- **成交量变化**: 可能成交量较为平稳
- **布林带位置**: 价格可能在中轨附近波动
""")
    else:
        # 其他组合情况
        analysis.append("""
- **相对强弱指标(RSI)**: 需进一步监测，目前无明确指向
- **MACD指标**: 需进一步确认趋势形成
- **成交量变化**: 建议关注后续成交量变化确认信号
""")
    
    # 3. 实际应用建议
    analysis.append("\n### 3️⃣ 实际应用建议\n")
    
    if trend == "上涨":
        if strength == "强":
            analysis.append("""
- **分批买入策略**: 考虑将资金分为3-4份，在未来2-3个交易日内分批买入
- **止损位设置**: 严格设置在当前价格的95%位置
- **目标价位**: 参考模型预测的上涨趋势，可以设定5-10%的盈利目标
- **持仓时间**: 建议持有5-7个交易日，与预测周期相对应
- **风险敞口控制**: 单次交易资金不超过总资金的15-20%
""")
        elif strength == "中":
            analysis.append("""
- **轻仓试探策略**: 考虑用较小仓位进行试探性买入
- **止损位设置**: 建议设置在当前价格的97%位置
- **目标价位**: 设定3-7%的盈利目标
- **持仓时间**: 建议持有3-5个交易日，随时准备止盈
- **风险敞口控制**: 单次交易资金不超过总资金的10%
""")
        else:
            analysis.append("""
- **观望为主策略**: 趋势虽为上涨但信号较弱，建议以观望为主
- **条件单设置**: 可设置突破条件单，确认突破后再介入
- **止损位设置**: 若介入，建议设置较紧的止损在当前价格的98%位置
- **风险敞口控制**: 若介入，建议使用不超过总资金5%的小仓位
""")
    elif trend == "下跌":
        if strength == "强":
            analysis.append("""
- **回避策略**: 不建议现在买入，市场看跌信号明确
- **持币观望**: 等待更好的买入时机或考虑其他投资标的
- **对冲策略**: 如有相关工具，可考虑适当做空对冲
- **减仓计划**: 若已持有，建议制定阶段性减仓计划
""")
        elif strength == "中":
            analysis.append("""
- **观望策略**: 不建议现在买入，市场下行趋势明显
- **止损设置**: 如已持有，建议设置止损位保护仓位
- **减仓规划**: 可考虑在反弹时减轻仓位
""")
        else:
            analysis.append("""
- **谨慎策略**: 保持谨慎，短期内市场可能有小幅回调
- **分批减仓**: 如已持有，可以制定分批减仓计划
- **观察支撑**: 关注近期支撑位表现，寻找企稳信号
""")
    else:  # 震荡
        analysis.append("""
- **区间交易策略**: 市场处于震荡期，可考虑区间交易策略
- **支撑阻力位重视**: 特别关注近期形成的支撑位和阻力位
- **轻仓短线**: 可尝试轻仓短线操作，严控风险
- **避免重仓**: 避免重仓操作，控制单笔交易风险
""")
    
    # 4. 风险因素剖析
    analysis.append("\n### 4️⃣ 风险因素剖析\n")
    
    analysis.append(f"- **模型置信度**: {accuracy:.2f}%的准确率意味着仍有约{100-accuracy:.2f}%的错误概率")
    analysis.append("- **外部事件风险**: 突发政策变化、公司公告或行业黑天鹅事件可能导致预测失效")
    analysis.append("- **市场流动性**: 若市场整体流动性收紧，可能影响预期涨跌幅实现")
    
    # 5. 决策建议总结
    analysis.append("\n### 5️⃣ 决策建议总结\n")
    
    if trend == "上涨":
        analysis.append("✅ **买入理由**:")
        if up_days == len(recent_preds):
            analysis.append(f"- 模型预测未来{args.predict_days}天连续上涨")
        else:
            analysis.append(f"- 模型预测未来{args.predict_days}天中{up_days}天上涨")
        analysis.append(f"- 信号强度为\"{strength}\"级别")
        analysis.append("- 模型验证显示较好的预测准确性")
    elif trend == "下跌":
        analysis.append("❌ **卖出/回避理由**:")
        if down_days == len(recent_preds):
            analysis.append(f"- 模型预测未来{args.predict_days}天连续下跌")
        else:
            analysis.append(f"- 模型预测未来{args.predict_days}天中{down_days}天下跌")
        analysis.append(f"- 信号强度为\"{strength}\"级别")
    else:
        analysis.append("⚖️ **观望理由**:")
        analysis.append("- 模型预测市场处于震荡状态")
        analysis.append(f"- 上涨天数:{up_days}，下跌天数:{down_days}，持平:{flat_days}")
        analysis.append(f"- 信号强度为\"{strength}\"级别")
    
    analysis.append("\n⚠️ **风险控制**:")
    if trend == "上涨":
        analysis.append(f"- 设置紧凑止损（{95+int(strength=='中')*2+int(strength=='弱')*3}%的当前价格）")
        analysis.append("- 分批建仓降低时点风险")
    elif trend == "下跌":
        analysis.append("- 避免逆势操作")
        analysis.append("- 如已持有，及时设置止损位")
    analysis.append("- 密切关注量能变化和盘中异动")
    
    # 合并所有分析段落
    return "\n".join(analysis)
