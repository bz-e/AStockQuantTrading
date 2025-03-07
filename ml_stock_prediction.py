#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易 - 机器学习预测模型
使用机器学习算法预测股票走势和交易信号
适用于A股市场特性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import datetime
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

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
        
        print(f"成功获取数据，共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None

def prepare_features(df):
    """准备特征数据"""
    # 复制数据，避免修改原始数据
    data = df.copy()
    
    # 计算基本技术指标
    # 移动平均线
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    
    # 价格变动百分比
    data['price_change_pct'] = data['close'].pct_change()
    
    # 波动率 (5日)
    data['volatility'] = data['close'].rolling(window=5).std()
    
    # 成交量指标
    data['volume_ma5'] = data['vol'].rolling(window=5).mean()
    data['volume_change'] = data['vol'].pct_change()
    
    # 价格与成交量关系
    data['price_volume_ratio'] = data['close'] / data['vol']
    
    # 涨跌幅度
    data['high_low_diff'] = (data['high'] - data['low']) / data['low'] * 100
    
    # 计算VWMA (成交量加权移动平均线)
    data['vwma'] = (data['close'] * data['vol']).rolling(window=10).sum() / data['vol'].rolling(window=10).sum()
    
    # 计算RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算OBV
    data['price_diff'] = data['close'].diff()
    data['obv'] = np.nan
    
    # 首日OBV设为0
    if len(data) > 0:
        data.loc[data.index[0], 'obv'] = 0
    
    for i in range(1, len(data)):
        if data['price_diff'].iloc[i] > 0:
            data.iloc[i, data.columns.get_indexer(['obv'])[0]] = data['obv'].iloc[i-1] + data['vol'].iloc[i]
        elif data['price_diff'].iloc[i] < 0:
            data.iloc[i, data.columns.get_indexer(['obv'])[0]] = data['obv'].iloc[i-1] - data['vol'].iloc[i]
        else:
            data.iloc[i, data.columns.get_indexer(['obv'])[0]] = data['obv'].iloc[i-1]
    
    # 涨跌停特征 (A股特性)
    # 涨停特征 (当日涨幅接近10%)
    data['is_limit_up'] = (data['pct_chg'] > 9.5).astype(int)
    # 跌停特征 (当日跌幅接近10%)
    data['is_limit_down'] = (data['pct_chg'] < -9.5).astype(int)
    
    # 移除NaN值
    data = data.dropna()
    
    return data

def create_labels(df, n_days=5):
    """
    创建标签数据
    n_days: 未来n天的价格变动
    """
    # 计算未来n天的收盘价变动比例
    df['future_return'] = df['close'].shift(-n_days) / df['close'] - 1
    
    # 设置标签: 1(买入)，-1(卖出)，0(持有)
    df['signal'] = 0
    # 如果未来n天上涨超过2%，标记为买入信号
    df.loc[df['future_return'] > 0.02, 'signal'] = 1
    # 如果未来n天下跌超过2%，标记为卖出信号
    df.loc[df['future_return'] < -0.02, 'signal'] = -1
    
    # 删除future_return列，只保留signal标签
    df = df.drop('future_return', axis=1)
    
    return df

def train_ml_model(df):
    """训练机器学习模型"""
    # 准备训练数据
    # 选择特征 (X) 和标签 (y)
    feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount', 
                      'ma5', 'ma10', 'ma20', 'price_change_pct', 'volatility',
                      'volume_ma5', 'volume_change', 'price_volume_ratio',
                      'high_low_diff', 'vwma', 'rsi', 'obv',
                      'is_limit_up', 'is_limit_down']
    
    X = df[feature_columns]
    y = df['signal']
    
    # 归一化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 训练随机森林分类器模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_columns

def predict_signals(model, scaler, features, df_new):
    """使用训练好的模型预测新数据的信号"""
    # 确保新数据包含所有必要的特征
    X_new = df_new[features]
    
    # 归一化特征
    X_new_scaled = scaler.transform(X_new)
    
    # 预测信号
    predicted_signals = model.predict(X_new_scaled)
    
    # 添加预测结果到数据框
    df_new['predicted_signal'] = predicted_signals
    
    return df_new

def plot_predictions(df, stock_code):
    """绘制预测结果与股价走势"""
    plt.figure(figsize=(14, 10))
    
    # 绘制股价走势
    plt.subplot(2, 1, 1)
    plt.plot(df['trade_date'], df['close'], label='收盘价')
    
    # 标记买入信号
    buy_signals = df[df['predicted_signal'] == 1]
    plt.scatter(buy_signals['trade_date'], buy_signals['close'], 
               color='green', label='买入信号', marker='^', s=100)
    
    # 标记卖出信号
    sell_signals = df[df['predicted_signal'] == -1]
    plt.scatter(sell_signals['trade_date'], sell_signals['close'], 
               color='red', label='卖出信号', marker='v', s=100)
    
    plt.title(f'{stock_code} 股价走势与ML预测信号')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制预测信号与实际信号对比
    plt.subplot(2, 1, 2)
    plt.plot(df['trade_date'], df['signal'], label='实际信号', color='blue')
    plt.plot(df['trade_date'], df['predicted_signal'], label='预测信号', color='orange')
    plt.title('实际信号 vs 预测信号')
    plt.xlabel('日期')
    plt.ylabel('信号 (1=买入, 0=持有, -1=卖出)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/{stock_code}_ml_prediction.png')
    plt.close()

def evaluate_prediction_performance(df):
    """评估预测性能"""
    # 获取最近的预测信号
    last_signal = df['predicted_signal'].iloc[-1]
    signal_text = "买入" if last_signal == 1 else ("卖出" if last_signal == -1 else "观望")
    
    print(f"\n最新交易信号: {signal_text}")
    
    # 计算预测准确率
    correct_predictions = (df['predicted_signal'] == df['signal']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    
    print(f"预测准确率: {accuracy:.4f}")
    
    # 计算买入信号准确率
    buy_signals = df[df['predicted_signal'] == 1]
    if len(buy_signals) > 0:
        correct_buys = (buy_signals['signal'] == 1).sum()
        buy_accuracy = correct_buys / len(buy_signals)
        print(f"买入信号准确率: {buy_accuracy:.4f}")
    
    # 计算卖出信号准确率
    sell_signals = df[df['predicted_signal'] == -1]
    if len(sell_signals) > 0:
        correct_sells = (sell_signals['signal'] == -1).sum()
        sell_accuracy = correct_sells / len(sell_signals)
        print(f"卖出信号准确率: {sell_accuracy:.4f}")
    
    # 分析预测错误的情况
    error_cases = df[df['predicted_signal'] != df['signal']]
    error_rate = len(error_cases) / total_predictions
    print(f"预测错误率: {error_rate:.4f}")
    
    # 计算平均收益率
    buy_returns = []
    for idx in buy_signals.index:
        if idx + 5 < len(df):  # 确保有足够的未来数据计算收益
            future_return = df['close'].iloc[idx + 5] / df['close'].iloc[idx] - 1
            buy_returns.append(future_return)
    
    if buy_returns:
        avg_buy_return = np.mean(buy_returns) * 100
        print(f"买入信号5天后平均收益率: {avg_buy_return:.2f}%")
    
    return signal_text

def main():
    """主函数"""
    # 初始化tushare
    pro = init_tushare()
    if not pro:
        print("初始化Tushare失败")
        return
    
    # 设置股票代码和日期范围
    stock_code = "002179.SZ"  # 恒瑞医药
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')  # 90天数据
    
    print(f"获取 {stock_code} 从 {start_date} 到 {today} 的数据用于机器学习预测")
    
    # 获取股票数据
    df = get_stock_data(pro, stock_code, start_date, today)
    if df is None or df.empty:
        print("未能获取有效数据，退出分析")
        return
    
    # 准备特征
    df_features = prepare_features(df)
    print("特征准备完成")
    
    # 创建标签
    df_labeled = create_labels(df_features)
    print("标签创建完成")
    
    # 删除含有NaN的行
    df_clean = df_labeled.dropna()
    if len(df_clean) < 30:  # 确保有足够的数据进行训练
        print("有效数据不足，无法进行可靠的机器学习预测")
        return
    
    print(f"数据准备完成，共 {len(df_clean)} 条有效记录")
    
    # 训练模型
    print("\n开始训练机器学习模型...")
    model, scaler, features = train_ml_model(df_clean)
    
    # 使用模型预测整个数据集的信号
    df_predicted = predict_signals(model, scaler, features, df_clean)
    
    # 绘制预测结果
    plot_predictions(df_predicted, stock_code)
    
    # 评估预测性能
    print("\n预测性能评估:")
    signal = evaluate_prediction_performance(df_predicted)
    
    print(f"\n分析完成！图表已保存至 images/{stock_code}_ml_prediction.png")
    print(f"机器学习预测的最新交易信号: {signal}")

if __name__ == "__main__":
    main()
