#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程模块 - 负责特征计算、选择和处理

该模块实现了股票数据的特征工程，包括技术指标计算、特征选择等功能。
"""

import numpy as np
import pandas as pd
import traceback
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA


def calculate_technical_indicators(df):
    """
    计算广泛应用于股票分析的技术指标
    
    该函数基于输入的股票数据DataFrame，计算并添加15类常用技术指标，包括移动平均线、
    MACD、RSI、布林带、动量指标等。这些指标可用于后续的机器学习模型训练，以预测股票
    价格走势。
    
    参数:
        df (pandas.DataFrame): 股票数据DataFrame，必须包含以下列：
                             - 'close': 收盘价
                             - 'high': 最高价
                             - 'low': 最低价
                             - 'open': 开盘价
                             - 'volume': 成交量
                             可选列：
                             - 'peTTM': 市盈率TTM
                             - 'pbMRQ': 市净率
    
    返回:
        pandas.DataFrame: 添加了以下技术指标的DataFrame：
                         1. 移动平均线 (MA) - 5/10/20/30/60日
                         2. 价格相对于移动平均线的百分比变化
                         3. 布林带 (BB) - 中轨、上轨、下轨、宽度、百分比位置
                         4. 相对强弱指标 (RSI) - 14日
                         5. MACD - 差值、信号线、柱状图
                         6. 动量指标 - 5/10/20日
                         7. 变化率 (ROC) - 5/10/20日
                         8. 真实波幅 (ATR) - 14日
                         9. 能量潮指标 (OBV) 及其变化率
                         10. KDJ指标 - K/D/J值
                         11. 价格波动范围及平均波动率
                         12. 成交量变化率 - 5/10/20日
                         13. 价格与成交量相关性 - 10/20/30日
                         14. PE百分位 (如有PE数据) - 20/60日
                         15. PB百分位 (如有PB数据) - 60日
    
    技术指标说明:
        - 移动平均线(MA): 反映价格平均走势，平滑价格波动
        - 布林带(BB): 反映价格波动区间，中轨为均线，上下轨为均线±2倍标准差
        - RSI: 相对强弱指标，衡量价格上涨动力，通常>70为超买，<30为超卖
        - MACD: 移动平均收敛/发散，用于判断趋势转变，金叉看涨，死叉看跌
        - 动量指标: 价格在特定周期内的变化率，反映上涨/下跌趋势的强度
        - ATR: 真实波幅，反映价格波动性的指标，常用于止损设置
        - OBV: 能量潮指标，结合价格和成交量判断买卖力量平衡
        - KDJ: 随机指标，用于判断价格走势超买/超卖状态
    
    示例:
        >>> import pandas as pd
        >>> # 创建示例股票数据
        >>> data = {
        ...     'date': pd.date_range(start='2023-01-01', periods=100),
        ...     'open': [100+i*0.1 for i in range(100)],
        ...     'high': [105+i*0.1 for i in range(100)],
        ...     'low': [95+i*0.1 for i in range(100)],
        ...     'close': [101+i*0.1 for i in range(100)],
        ...     'volume': [1000000+i*10000 for i in range(100)]
        ... }
        >>> df = pd.DataFrame(data)
        >>> # 计算技术指标
        >>> df_with_indicators = calculate_technical_indicators(df)
        >>> print(f"原始列数: {len(df.columns)}, 添加指标后列数: {len(df_with_indicators.columns)}")
        >>> print("新增指标:", set(df_with_indicators.columns) - set(df.columns))
    
    注意:
        1. 指标计算需要一定历史数据，前N个样本的某些指标将包含NaN值
        2. 建议在计算指标前确保数据已按日期排序（升序）
        3. 该函数会创建原始DataFrame的副本，不会修改输入数据
        4. 计算较多指标会显著增加DataFrame的列数(约65个特征)
        5. 部分指标（如MACD）需要较长时间窗口数据才能产生有意义的信号
    """
    try:
        # 创建副本避免修改原始数据
        result_df = df.copy()
        
        # 确保所需列存在
        required_columns = ['close', 'high', 'low', 'open', 'volume']
        for col in required_columns:
            if col not in result_df.columns:
                print(f"警告: 缺少计算技术指标所需的列: {col}")
                return df
        
        # 1. 移动平均线 (5, 10, 20, 30, 60天)
        for window in [5, 10, 20, 30, 60]:
            result_df[f'ma_{window}'] = result_df['close'].rolling(window=window).mean()
            
        # 2. 价格相对移动平均线的百分比变化
        for window in [5, 10, 20, 30, 60]:
            ma_col = f'ma_{window}'
            result_df[f'close_vs_{ma_col}'] = (result_df['close'] / result_df[ma_col] - 1) * 100
        
        # 3. 布林带指标 (20日移动平均线 +/- 2倍标准差)
        window = 20
        result_df['bb_middle'] = result_df['close'].rolling(window=window).mean()
        result_df['bb_std'] = result_df['close'].rolling(window=window).std()
        result_df['bb_upper'] = result_df['bb_middle'] + 2 * result_df['bb_std']
        result_df['bb_lower'] = result_df['bb_middle'] - 2 * result_df['bb_std']
        result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_pct'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        # 4. RSI (相对强弱指标) - 14天
        delta = result_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 5. MACD (移动平均收敛/发散)
        result_df['ema_12'] = result_df['close'].ewm(span=12, adjust=False).mean()
        result_df['ema_26'] = result_df['close'].ewm(span=26, adjust=False).mean()
        result_df['macd'] = result_df['ema_12'] - result_df['ema_26']
        result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
        result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
        
        # 6. 动量指标 (5, 10, 20日)
        for window in [5, 10, 20]:
            result_df[f'momentum_{window}'] = result_df['close'].pct_change(periods=window) * 100
        
        # 7. ROC (变化率)
        for window in [5, 10, 20]:
            result_df[f'roc_{window}'] = (result_df['close'] / result_df['close'].shift(window) - 1) * 100
        
        # 8. ATR (真实波幅)
        tr1 = result_df['high'] - result_df['low']
        tr2 = abs(result_df['high'] - result_df['close'].shift())
        tr3 = abs(result_df['low'] - result_df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df['atr_14'] = tr.rolling(window=14).mean()
        
        # 9. OBV (能量潮指标)
        result_df['daily_ret'] = result_df['close'].pct_change()
        result_df['obv'] = np.where(result_df['daily_ret'] > 0, result_df['volume'], 
                               np.where(result_df['daily_ret'] < 0, -result_df['volume'], 0)).cumsum()
        result_df['obv_change'] = result_df['obv'].pct_change(5)
        
        # 10. KDJ指标
        result_df['low_9'] = result_df['low'].rolling(9).min()
        result_df['high_9'] = result_df['high'].rolling(9).max()
        result_df['rsv'] = 100 * ((result_df['close'] - result_df['low_9']) / 
                              (result_df['high_9'] - result_df['low_9']))
        result_df['kdj_k'] = result_df['rsv'].ewm(com=2).mean()
        result_df['kdj_d'] = result_df['kdj_k'].ewm(com=2).mean()
        result_df['kdj_j'] = 3 * result_df['kdj_k'] - 2 * result_df['kdj_d']
        
        # 11. 价格波动范围
        result_df['volatility'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100
        for window in [5, 10, 20]:
            result_df[f'volatility_{window}d'] = result_df['volatility'].rolling(window).mean()
            
        # 12. 交易量变化
        for window in [5, 10, 20]:
            result_df[f'volume_change_{window}d'] = result_df['volume'].pct_change(window) * 100
            
        # 13. 价格与成交量相关性
        for window in [10, 20, 30]:
            result_df[f'price_volume_corr_{window}d'] = result_df['close'].rolling(window).corr(result_df['volume'])
        
        # 14. 如果有市盈率数据，添加PE相关指标
        if 'peTTM' in result_df.columns:
            for window in [20, 60]:
                result_df[f'pe_percentile_{window}d'] = result_df['peTTM'].rolling(window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
        
        # 15. 如果有市净率数据，添加PB相关指标
        if 'pbMRQ' in result_df.columns:
            result_df['pb_percentile'] = result_df['pbMRQ'].rolling(60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
        
        return result_df
        
    except Exception as e:
        print(f"计算技术指标时出错: {e}")
        traceback.print_exc()
        return df


def prepare_features(df, prediction_days=5, feature_selection=True, variance_threshold=0.001, 
                    correlation_threshold=0.85, use_pca=False, pca_components=0.95, 
                    feature_selection_method='variance'):
    """
    准备机器学习模型训练所需的特征数据和标签
    
    该函数基于股票历史数据，构建预测未来股价走势所需的特征矩阵和目标变量。
    支持特征筛选、高相关性特征去除和PCA降维，以优化特征集并减少过拟合风险。
    
    参数:
        df (pandas.DataFrame): 包含原始股票数据的DataFrame，必须包含'close'列用于
                             创建目标变量，且所有作为特征的列必须为数值类型。
        
        prediction_days (int): 预测未来多少天的涨跌，默认为5天。此参数决定了从当前
                             时间点看多远的未来，较大的值适合中长期预测，较小的值
                             适合短期预测。
        
        feature_selection (bool): 是否进行特征选择，默认为True。启用后会根据
                               feature_selection_method参数选择的方法筛选特征。
        
        variance_threshold (float): 方差阈值，用于去除低方差特征，默认为0.001。
                                  方差低于此阈值的特征将被视为提供信息量较少。
                                  仅当feature_selection为True且方法包含'variance'时使用。
        
        correlation_threshold (float): 相关性阈值，默认为0.85。当两个特征间的相关系数
                                     绝对值超过此阈值时，将移除其中方差较低的特征，以
                                     减少信息冗余。范围为0-1，较高的值更为保守。
        
        use_pca (bool): 是否使用主成分分析(PCA)进行降维，默认为False。PCA可以减少
                      特征数量同时保留大部分信息，但会影响特征的可解释性。
        
        pca_components (float or int): PCA保留的信息量比例(0-1)或组件数量(>1)，
                                     默认为0.95，表示保留95%的方差信息。
        
        feature_selection_method (str): 特征选择方法，可选值：
                                      - 'variance': 基于方差的选择，移除低方差特征
                                      - 'kbest': 使用F统计量选择K个最优特征
                                      - 'combined': 同时使用以上两种方法
                                      默认为'variance'。
    
    返回:
        tuple: 包含以下3个元素的元组：
            - X (numpy.ndarray): 特征矩阵，形状为(n_samples, n_features)
            - y (numpy.ndarray): 目标变量向量，形状为(n_samples,)，值为0(下跌)或1(上涨)
            - feature_names (list): 特征名列表，与X的列一一对应
    
    特征处理逻辑：
        1. 创建目标变量：根据prediction_days设置的未来时间点与当前价格的比较结果
        2. 排除不应用作特征的列（如日期、目标变量等）
        3. 数据类型检查与转换：确保所有特征为数值类型
        4. 特征筛选（可选）：
           - 方差筛选：移除方差过低的特征
           - 相关性筛选：移除高度相关的冗余特征
           - K优特征：基于统计检验选择最相关的特征
        5. PCA降维（可选）：将特征投影到主成分空间以减少维度
        6. 数据清理：处理缺失值，通常采用前向填充策略
    
    示例：
        >>> # 假设df是已经包含各种技术指标的股票数据DataFrame
        >>> X, y, feature_names = prepare_features(
        ...     df, 
        ...     prediction_days=10,  # 预测10天后的涨跌
        ...     feature_selection=True,
        ...     correlation_threshold=0.8,  # 较低的相关性阈值，更积极地移除相关特征
        ...     feature_selection_method='combined'  # 使用组合方法选择特征
        ... )
        >>> print(f"特征数量: {len(feature_names)}")
        >>> print(f"样本数量: {len(y)}")
        >>> print(f"上涨样本比例: {sum(y)/len(y)*100:.2f}%")
    
    注意：
        1. 模型训练前应将数据分割为训练集和测试集，本函数不执行此操作
        2. 数据应预先按时间顺序排序（升序），确保未来数据不会泄露到特征计算中
        3. NaN值处理采用前向填充策略，如有特殊需求可能需要在外部预处理
        4. 当样本数量小于特征数量时，会发出警告，此时应考虑更激进的特征选择
        5. 过度的特征选择可能导致信息丢失，建议根据实际情况调整阈值参数
        6. 返回的特征矩阵X已经过特征选择/降维处理，可能不包含原始DataFrame的所有特征
    """
    try:
        # 复制数据集以避免修改原始数据
        data = df.copy()
        
        # 检查预测天数是否有效
        if prediction_days <= 0:
            print("警告: 预测天数必须为正整数，将使用默认值5")
            prediction_days = 5
            
        # 创建目标变量: N天后的涨跌
        # 1表示上涨，0表示下跌或持平
        data['future_close'] = data['close'].shift(-prediction_days)
        data['target'] = (data['future_close'] > data['close']).astype(int)
        
        # 移除NaN值
        data.dropna(inplace=True)
        
        if len(data) == 0:
            print("错误: 处理后数据为空，无法继续")
            return None, None, None
            
        # 排除不作为特征的列
        exclude_columns = ['date', 'code', 'future_close', 'target', 'adjustflag',
                          'tradestatus', 'isST', 'signal', 'future_return']
        
        # 获取特征列
        feature_cols = [col for col in data.columns if col not in exclude_columns]
        
        print(f"初始特征数量: {len(feature_cols)}")
        
        # 检查数据是否足够
        if len(data) < len(feature_cols):
            print(f"警告: 样本数量({len(data)})小于特征数量({len(feature_cols)})，可能导致过拟合")
            
        # 检查并确保所有特征都是数值型的
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"警告: 特征 '{col}' 不是数值类型，尝试转换为数值")
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    print(f"无法将特征 '{col}' 转换为数值类型: {e}")
                    # 从特征列表中移除
                    feature_cols.remove(col)
                    
        # 再次检查是否有任何非数值特征或含有NaN的特征
        non_numeric_cols = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[col]) or data[col].isna().any():
                non_numeric_cols.append(col)
                
        if non_numeric_cols:
            print(f"警告: 以下特征包含非数值数据或NaN值，将被排除: {non_numeric_cols}")
            for col in non_numeric_cols:
                feature_cols.remove(col)
                
        # 确保所有NaN都被填充或行被丢弃
        data = data[feature_cols + ['target']].dropna()
        
        # 初始化标签和特征
        y = data['target'].values
        X = data[feature_cols].values
        
        # 确保所有值都是数值型的
        X = X.astype(float)
        
        print(f"最终特征数量: {len(feature_cols)}")
        
        # 进行特征选择
        if feature_selection and len(feature_cols) > 5:
            try:
                print(f"使用特征选择方法: {feature_selection_method}")
                
                # 方差选择 - 移除低方差特征
                if feature_selection_method in ['variance', 'combined']:
                    print(f"使用方差阈值 {variance_threshold} 进行特征选择")
                    var_selector = VarianceThreshold(threshold=variance_threshold)
                    X_var = var_selector.fit_transform(X)
                    
                    # 获取保留的特征索引
                    var_support = var_selector.get_support()
                    
                    # 更新特征列表
                    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if var_support[i]]
                    
                    if len(selected_features) > 0:
                        feature_cols = selected_features
                        X = X_var
                        print(f"方差选择后的特征数量: {len(feature_cols)}")
                    else:
                        print("方差选择后没有保留特征，跳过此步骤")
                
                # K优特征选择 - 使用F统计量
                if feature_selection_method in ['kbest', 'combined'] and len(feature_cols) > 2:
                    # 确定特征选择的k值 (选择至少50%的特征)
                    k = max(min(len(feature_cols) // 2, X.shape[1]), 1)
                    print(f"使用 SelectKBest 选择 {k} 个最佳特征")
                    
                    # 特征选择
                    k_selector = SelectKBest(score_func=f_classif, k=k)
                    X_k = k_selector.fit_transform(X, y)
                    
                    # 获取保留的特征索引
                    k_support = k_selector.get_support()
                    
                    # 更新特征列表
                    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if k_support[i]]
                    
                    if len(selected_features) > 0:
                        feature_cols = selected_features
                        X = X_k
                        print(f"K优特征选择后的特征数量: {len(feature_cols)}")
                    else:
                        print("K优特征选择后没有保留特征，跳过此步骤")
                
                print(f"特征选择后的最终特征数量: {len(feature_cols)}")
            except Exception as e:
                print(f"特征选择时出错，将使用所有特征: {e}")
        
        return X, y, feature_cols
        
    except Exception as e:
        print(f"准备特征时出错: {e}")
        traceback.print_exc()
        return None, None, None


def get_important_features(model, feature_names, threshold=0.01):
    """
    获取重要特征
    
    参数:
        model (object): 训练好的模型
        feature_names (list): 特征名称列表
        threshold (float): 重要性阈值，低于该值的特征将被忽略
    
    返回:
        list: 重要特征列表，每个元素为(特征名, 重要性)元组
    """
    try:
        # 检查模型是否有feature_importances_属性
        if not hasattr(model, 'feature_importances_'):
            print("模型没有feature_importances_属性，无法获取特征重要性")
            return []
            
        # 检查特征名列表长度是否与特征重要性长度一致
        if len(feature_names) != len(model.feature_importances_):
            print(f"特征名列表长度({len(feature_names)})与特征重要性长度({len(model.feature_importances_)})不一致")
            return []
            
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 将特征名和重要性匹配并排序
        feature_importance = [(feature_names[i], importance) for i, importance in enumerate(importances) if importance > threshold]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
        
    except Exception as e:
        print(f"获取重要特征时出错: {e}")
        traceback.print_exc()
        return []
