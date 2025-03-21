#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易 - 增强版机器学习预测模型
包含多种改进:
1. 可调整预测时间周期(1天/10天/20天)
2. 增强特征工程(MACD, ATR等)
3. 多种机器学习算法支持
4. 交叉验证和涨跌停优化
5. 集成学习模型(投票和堆叠)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import baostock as bs
import argparse
import os
import traceback
import warnings
import time
from datetime import datetime, timedelta
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 检查是否有XGBoost和LightGBM
has_xgboost = False
has_lightgbm = False

try:
    import xgboost as xgb

    has_xgboost = True
except ImportError:
    print("XGBoost未安装，将使用随机森林作为默认模型")
    has_xgboost = False

try:
    import lightgbm as lgb

    has_lightgbm = True
except ImportError:
    print("LightGBM未安装，将使用随机森林作为默认模型")
    has_lightgbm = False

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 用于显示负号

# 创建目录
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("predictions"):
    os.makedirs("predictions")


def get_stock_data(
    stock_code, start_date=None, end_date=None, retry_count=3, add_basic_data=True
):
    """
    获取股票历史数据，包含重试机制和增强的数据清洗

    参数:
    stock_code: 股票代码，如sh.600036
    start_date: 开始日期，如2020-01-01
    end_date: 结束日期，如2020-12-31
    retry_count: 重试次数
    add_basic_data: 是否添加更多基本面数据

    返回:
    包含历史数据的DataFrame
    """
    today = datetime.now()
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 确保日期格式正确 (YYYY-MM-DD)
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")

    print(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据")

    for attempt in range(retry_count):
        try:
            # 登录baostock
            login_result = bs.login()
            if login_result.error_code != "0":
                print(f"登录失败: {login_result.error_msg}")
                time.sleep(1)
                continue

            # 获取历史A股K线数据
            fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

            # 如果需要添加更多基本面数据
            if add_basic_data:
                # 尝试添加更多财务指标
                extended_fields = fields + ",roeTTM,roa,netMargin,grossMargin"
                try:
                    rs_test = bs.query_history_k_data_plus(
                        stock_code,
                        extended_fields,
                        start_date=start_date,
                        end_date=start_date,
                        frequency="d",
                        adjustflag="3",
                    )
                    if rs_test.error_code == "0":
                        fields = extended_fields
                        print("成功添加扩展财务指标")
                except:
                    print("扩展财务指标不可用，使用基本字段")

            rs = bs.query_history_k_data_plus(
                stock_code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )  # 3表示复权

            if rs.error_code != "0":
                print(f"查询失败: {rs.error_msg}, 尝试 {attempt+1}/{retry_count}")
                bs.logout()
                time.sleep(2)
                continue

            # 生成DataFrame
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            # 检查是否获取到数据
            if not data_list:
                print(f"未获取到数据，尝试 {attempt+1}/{retry_count}")
                bs.logout()
                time.sleep(2)
                continue

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 登出系统
            bs.logout()

            # 数据清洗和预处理
            if df.empty:
                print(f"未获取到 {stock_code} 的数据")
                continue

            print("成功获取", len(df), "条数据")
            print("数据列名:", df.columns.tolist())

            # 日期格式标准化
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            # 转换数值列为数值类型
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turn",
                "pctChg",
                "peTTM",
                "pbMRQ",
                "psTTM",
                "pcfNcfTTM",
            ]

            # 尝试转换扩展财务指标
            if "roeTTM" in df.columns:
                numeric_columns.extend(["roeTTM", "roa", "netMargin", "grossMargin"])

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 处理缺失值: 使用前值填充，更好地保持时间序列连续性
            df = df.ffill()  # 使用ffill()替代fillna(method='ffill')

            # 排序和重置索引
            if "date" in df.columns:
                df.sort_values("date", inplace=True)
                df.reset_index(drop=True, inplace=True)

            # 检测异常值并处理（例如价格跳跃异常）
            if len(df) > 3:
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        # 计算滚动中位数
                        med = df[col].rolling(5, min_periods=1).median()
                        # 计算滚动中位数绝对偏差
                        mad = np.abs(df[col] - med).rolling(5, min_periods=1).median()
                        # 如果偏离中位数太大（例如>10倍MAD），则替换为中位数
                        outlier_idx = df[np.abs(df[col] - med) > 10 * mad].index
                        if len(outlier_idx) > 0:
                            print(f"检测到 {len(outlier_idx)} 个异常值，进行修正")
                            df.loc[outlier_idx, col] = med[outlier_idx]

            return df

        except Exception as e:
            print(f"获取股票数据时出错: {e}")
            traceback.print_exc()
            try:
                bs.logout()
            except:
                pass
            if attempt < retry_count - 1:
                print(f"尝试重新获取数据 ({attempt+1}/{retry_count})")
                time.sleep(2)

    print("所有重试均失败")
    return None


def get_index_data(index_code, days=365):
    """
    获取指数历史数据

    参数:
    index_code: 指数代码
    days: 获取多少天的历史数据，默认365天

    返回:
    包含指数数据的DataFrame
    """
    # 计算开始日期和结束日期
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        # 登录系统
        lg = bs.login()
        if lg.error_code != "0":
            print(f"登录失败: {lg.error_msg}")
            return None

        # 获取指数日K数据
        rs = bs.query_history_k_data_plus(
            index_code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",  # 不复权
        )

        if rs.error_code != "0":
            print(f"获取指数数据失败: {rs.error_msg}")
            bs.logout()
            return None

        # 处理数据
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        # 登出系统
        bs.logout()

        if len(data_list) == 0:
            print(f"未获取到指数 {index_code} 的数据")
            return None

        # 转换为DataFrame
        result = pd.DataFrame(data_list, columns=rs.fields)

        # 转换数据类型
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        # 将日期设为索引
        result["date"] = pd.to_datetime(result["date"])
        result.rename(columns={"date": "trade_date"}, inplace=True)
        result.set_index("trade_date", inplace=True)

        return result

    except Exception as e:
        print(f"获取指数数据时发生错误: {e}")
        bs.logout()
        return None


def prepare_features(
    df,
    prediction_days=5,
    feature_selection=True,
    variance_threshold=0.001,
    correlation_threshold=0.85,
    use_pca=False,
    pca_components=0.95,
    feature_selection_method="variance",
):
    """
    准备模型训练所需的特征

    参数:
    df: 包含原始数据的DataFrame
    prediction_days: 预测未来多少天的涨跌
    feature_selection: 是否进行特征选择
    variance_threshold: 方差阈值，用于去除低方差特征
    correlation_threshold: 相关性阈值，超过此阈值的特征对将被视为高度相关
    use_pca: 是否使用PCA进行降维
    pca_components: PCA保留的方差比例(0-1)或组件数量(>1)
    feature_selection_method: 特征选择方法，可选：'variance'(方差选择), 'kbest'(K优特征), 'combined'(两者结合)

    返回:
    X: 特征DataFrame
    y: 目标变量Series
    feature_names: 特征名列表
    """
    result_df = df.copy()

    # 将日期转换为日期特征
    if "date" in result_df.columns:
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df["day_of_week"] = result_df["date"].dt.dayofweek
        result_df["day_of_month"] = result_df["date"].dt.day
        result_df["month"] = result_df["date"].dt.month
        result_df["quarter"] = result_df["date"].dt.quarter

        # 周期性特征的三角变换
        result_df["day_of_week_sin"] = np.sin(2 * np.pi * result_df["day_of_week"] / 5)
        result_df["day_of_week_cos"] = np.cos(2 * np.pi * result_df["day_of_week"] / 5)
        result_df["month_sin"] = np.sin(2 * np.pi * result_df["month"] / 12)
        result_df["month_cos"] = np.cos(2 * np.pi * result_df["month"] / 12)

    # 1. 基础特征：涨跌幅
    result_df["change"] = result_df["close"].pct_change()

    # 2. 价格类特征
    result_df["price_range"] = (result_df["high"] - result_df["low"]) / result_df[
        "low"
    ]  # 当日价格区间
    result_df["price_momentum"] = (
        result_df["close"] / result_df["open"] - 1
    )  # 当日价格动量

    # 3. 加入技术指标
    technical_indicators_df = calculate_technical_indicators(result_df)
    if technical_indicators_df is not None:
        result_df = technical_indicators_df

    # 4. 添加滞后特征 (过去N天的涨跌幅)
    for i in range(1, 10):  # 增加到9天的滞后特征
        result_df[f"lag_change_{i}"] = result_df["change"].shift(i)

    # 5. 添加周期性变动特征
    result_df["weekly_effect"] = result_df["change"].rolling(window=5).mean()  # 周效应
    result_df["monthly_effect"] = (
        result_df["change"].rolling(window=20).mean()
    )  # 月效应

    # 6. 添加移动平均线
    for window in [5, 10, 20, 30, 60]:
        result_df[f"sma_{window}"] = result_df["close"].rolling(window=window).mean()
        result_df[f"sma_change_{window}"] = result_df[f"sma_{window}"].pct_change()

        # 指数移动平均
        result_df[f"ema_{window}"] = (
            result_df["close"].ewm(span=window, adjust=False).mean()
        )
        result_df[f"ema_change_{window}"] = result_df[f"ema_{window}"].pct_change()

    # 7. 价格与均线的相对关系
    for window in [5, 10, 20, 30, 60]:
        result_df[f"close_to_sma_{window}"] = (
            result_df["close"] / result_df[f"sma_{window}"] - 1
        )
        result_df[f"close_to_ema_{window}"] = (
            result_df["close"] / result_df[f"ema_{window}"] - 1
        )

    # 8. 均线交叉信号
    result_df["sma_5_10_cross"] = (result_df["sma_5"] > result_df["sma_10"]).astype(int)
    result_df["sma_10_20_cross"] = (result_df["sma_10"] > result_df["sma_20"]).astype(
        int
    )
    result_df["ema_5_10_cross"] = (result_df["ema_5"] > result_df["ema_10"]).astype(int)
    result_df["ema_10_20_cross"] = (result_df["ema_10"] > result_df["ema_20"]).astype(
        int
    )

    # 9. 波动性指标
    result_df["volatility_5d"] = result_df["change"].rolling(window=5).std()
    result_df["volatility_10d"] = result_df["change"].rolling(window=10).std()
    result_df["volatility_20d"] = result_df["change"].rolling(window=20).std()
    result_df["volatility_ratio_5_20"] = (
        result_df["volatility_5d"] / result_df["volatility_20d"]
    )

    # 10. 成交量特征
    result_df["volume_change"] = result_df["volume"].pct_change()
    result_df["volume_ma_5"] = result_df["volume"].rolling(window=5).mean()
    result_df["volume_ma_10"] = result_df["volume"].rolling(window=10).mean()
    result_df["volume_ma_20"] = result_df["volume"].rolling(window=20).mean()
    result_df["volume_ma_ratio_5"] = result_df["volume"] / result_df["volume_ma_5"]
    result_df["volume_ma_ratio_10"] = result_df["volume"] / result_df["volume_ma_10"]
    result_df["volume_ma_ratio_20"] = result_df["volume"] / result_df["volume_ma_20"]

    # 11. 价格-成交量组合指标
    result_df["price_volume_change"] = result_df["change"] * result_df["volume_change"]
    result_df["obv"] = (np.sign(result_df["change"]) * result_df["volume"]).cumsum()
    result_df["obv_change"] = result_df["obv"].pct_change()

    # 12. 趋势特征
    result_df["trend_5d"] = (result_df["close"] > result_df["close"].shift(5)).astype(
        int
    )
    result_df["trend_10d"] = (result_df["close"] > result_df["close"].shift(10)).astype(
        int
    )
    result_df["trend_20d"] = (result_df["close"] > result_df["close"].shift(20)).astype(
        int
    )

    # 13. 价格突破特征
    for window in [5, 10, 20, 30]:
        # 上轨突破
        upper_band = result_df["close"].rolling(window=window).max()
        result_df[f"upper_breakout_{window}"] = (
            result_df["close"] >= upper_band
        ).astype(int)

        # 下轨突破
        lower_band = result_df["close"].rolling(window=window).min()
        result_df[f"lower_breakout_{window}"] = (
            result_df["close"] <= lower_band
        ).astype(int)

    # 14. 相对强弱指标的衍生指标
    result_df["rsi_diff"] = result_df["rsi_14"] - result_df["rsi_14"].shift(1)
    result_df["rsi_ma5"] = result_df["rsi_14"].rolling(window=5).mean()
    result_df["rsi_ma10"] = result_df["rsi_14"].rolling(window=10).mean()
    result_df["rsi_cross"] = (result_df["rsi_14"] > result_df["rsi_ma5"]).astype(int)

    # 15. MACD指标的衍生指标
    if "macd" in result_df.columns and "macd_signal" in result_df.columns:
        result_df["macd_diff"] = result_df["macd"] - result_df["macd_signal"]
        result_df["macd_pos"] = (result_df["macd"] > 0).astype(int)
        result_df["macd_cross"] = (result_df["macd"] > result_df["macd_signal"]).astype(
            int
        )
        result_df["macd_cross_change"] = result_df["macd_cross"].diff()

    # 16. 布林带相关指标
    if "upper_band" in result_df.columns and "lower_band" in result_df.columns:
        result_df["bb_width"] = (
            result_df["upper_band"] - result_df["lower_band"]
        ) / result_df["middle_band"]
        result_df["bb_position"] = (result_df["close"] - result_df["lower_band"]) / (
            result_df["upper_band"] - result_df["lower_band"]
        )
        result_df["bb_squeeze"] = (
            (result_df["bb_width"] < result_df["bb_width"].shift(5))
            & (result_df["bb_width"] < 0.1)
        ).astype(int)

    # 17. 星期几特征 (将日期转换为独热编码)
    if "day_of_week" in result_df.columns:
        # 使用独热编码处理星期几
        for i in range(5):  # 只考虑工作日（0-4对应周一至周五）
            result_df[f"day_{i}"] = (result_df["day_of_week"] == i).astype(int)

    # 18. 基本面特征比率
    if "peTTM" in result_df.columns and "pbMRQ" in result_df.columns:
        result_df["pe_to_pb"] = result_df["peTTM"] / result_df["pbMRQ"]
        result_df["pe_percentile"] = (
            result_df["peTTM"]
            .rolling(window=60)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        result_df["pb_percentile"] = (
            result_df["pbMRQ"]
            .rolling(window=60)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

    # 19. 指数和板块影响
    if "index_close" in result_df.columns:
        # 与大盘的相对表现
        result_df["relative_to_index"] = (
            result_df["close"].pct_change() - result_df["index_close"].pct_change()
        )
        result_df["relative_strength_5d"] = result_df["close"].pct_change(
            5
        ) - result_df["index_close"].pct_change(5)
        result_df["relative_strength_10d"] = result_df["close"].pct_change(
            10
        ) - result_df["index_close"].pct_change(10)

    if "sector_close" in result_df.columns:
        # 与板块的相对表现
        result_df["relative_to_sector"] = (
            result_df["close"].pct_change() - result_df["sector_close"].pct_change()
        )
        result_df["sector_relative_5d"] = result_df["close"].pct_change(5) - result_df[
            "sector_close"
        ].pct_change(5)
        result_df["sector_relative_10d"] = result_df["close"].pct_change(
            10
        ) - result_df["sector_close"].pct_change(10)

    # 创建目标变量：未来N天是否上涨
    result_df["future_change"] = (
        result_df["close"].shift(-prediction_days) / result_df["close"] - 1
    )
    result_df["future_up"] = result_df["future_change"].apply(
        lambda x: -1 if x <= -0.02 else 1 if x >= 0.02 else 0
    )

    # 丢弃包含NaN的行
    result_df = result_df.dropna()

    # 确保我们有足够的数据
    if len(result_df) == 0:
        print("警告: 处理后数据为空，无法继续特征工程和模型训练")
        return pd.DataFrame(), pd.Series(), []

    # 移除不需要的特征列
    features_to_remove = [
        "date",
        "code",
        "future_change",
        "open",
        "high",
        "low",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
    ]
    for col in features_to_remove:
        if col in result_df.columns:
            result_df = result_df.drop(col, axis=1)

    # 分离特征和目标变量
    X = result_df.drop("future_up", axis=1)
    y = result_df["future_up"]

    # 处理非数值特征（如果有的话）
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X = X[numeric_cols]

    # 特征名列表
    feature_names = X.columns.tolist()

    # 记录原始特征数量
    original_feature_count = len(feature_names)
    print(f"原始特征数量: {original_feature_count}")

    # 特征选择（可选）
    if feature_selection and len(feature_names) > 0:
        try:
            print("开始进行特征选择...")

            # 1. 移除高度相关的特征
            if correlation_threshold < 1.0:
                print(f"检测并移除高相关性特征(阈值: {correlation_threshold})...")
                # 计算相关性矩阵
                corr_matrix = X.corr().abs()

                # 获取上三角矩阵
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # 找出相关性大于阈值的特征
                to_drop = [
                    column
                    for column in upper.columns
                    if any(upper[column] > correlation_threshold)
                ]

                if to_drop:
                    print(f"移除 {len(to_drop)} 个高相关性特征:")
                    for col in to_drop[:5]:  # 只打印前5个，避免输出过多
                        print(f"  - {col}")
                    if len(to_drop) > 5:
                        print(f"  - ... 还有 {len(to_drop)-5} 个")

                    X = X.drop(columns=to_drop)
                    feature_names = X.columns.tolist()
                    print(f"移除高相关特征后剩余特征数量: {len(feature_names)}")

            # 2. 根据所选方法进行特征选择
            if feature_selection_method in ["variance", "combined"]:
                # 移除低方差特征
                from sklearn.feature_selection import VarianceThreshold

                selector = VarianceThreshold(variance_threshold)
                X_new = selector.fit_transform(X)

                # 获取被选择的特征索引
                selected_indices = selector.get_support(indices=True)

                # 确保我们至少保留一些特征
                if len(selected_indices) > 0:
                    selected_features = [feature_names[i] for i in selected_indices]

                    # 更新特征集
                    X = pd.DataFrame(X_new, columns=selected_features, index=X.index)
                    feature_names = selected_features
                    print(f"方差筛选后剩余特征数量: {len(feature_names)}")
                else:
                    print("警告: 方差筛选后没有留下任何特征，跳过此步骤")

            if (
                feature_selection_method in ["kbest", "combined"]
                and len(feature_names) > 0
            ):
                # 使用SelectKBest选择最相关的特征
                from sklearn.feature_selection import SelectKBest, f_classif

                # 确定要选择的特征数量 (最多选择100个特征或60%的特征，以较小者为准)
                k = min(
                    100, max(int(len(feature_names) * 0.6), min(30, len(feature_names)))
                )

                # 确保k至少为1，且不大于特征数量
                k = max(1, min(k, len(feature_names)))

                selector_k = SelectKBest(f_classif, k=k)
                X_new = selector_k.fit_transform(X, y)

                # 获取被选择的特征
                selected_indices_k = selector_k.get_support(indices=True)

                # 确保我们至少保留一些特征
                if len(selected_indices_k) > 0:
                    selected_features = [feature_names[i] for i in selected_indices_k]

                    # 更新特征集
                    X = pd.DataFrame(X_new, columns=selected_features, index=X.index)
                    feature_names = selected_features
                    print(f"K-Best筛选后剩余特征数量: {len(feature_names)}")
                else:
                    print("警告: K-Best筛选后没有留下任何特征，跳过此步骤")

            # 3. 应用PCA降维（如果需要）
            if use_pca and len(feature_names) > 3:  # 至少需要3个特征才值得做PCA
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA

                # 标准化特征
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 应用PCA
                if isinstance(pca_components, float) and 0 < pca_components < 1:
                    pca = PCA(n_components=pca_components)  # 保留指定比例的方差
                else:
                    pca = PCA(
                        n_components=min(int(pca_components), len(feature_names))
                    )  # 使用指定数量的组件

                X_pca = pca.fit_transform(X_scaled)

                # 创建新的特征名
                new_feature_names = [
                    f"PCA_Component_{i+1}" for i in range(X_pca.shape[1])
                ]

                # 更新特征集
                X = pd.DataFrame(X_pca, columns=new_feature_names, index=X.index)
                feature_names = new_feature_names

                # 打印PCA信息
                print(f"PCA降维后的特征数量: {len(feature_names)}")
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                print(f"PCA解释的累计方差: {cumulative_variance[-1]:.4f}")

            print(
                f"特征选择后，保留了 {len(feature_names)} 个特征 (从 {original_feature_count} 减少)"
            )
        except Exception as e:
            print(f"特征选择过程出错: {e}")
            traceback.print_exc()

    return X, y, feature_names


def get_important_features(model, feature_names, threshold=0.01):
    """
    获取重要特征

    参数:
    model: 训练好的模型
    feature_names: 特征名称列表
    threshold: 重要性阈值，低于该值的特征将被忽略

    返回:
    重要特征列表
    """
    # 获取特征重要性
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("模型不支持特征重要性")
        return feature_names

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # 筛选重要特征
    important_features = feature_importance[
        feature_importance["importance"] > threshold
    ]["feature"].tolist()

    return important_features


def train_ml_model(
    X,
    y,
    model_type="rf",
    feature_names=None,
    use_cv=True,
    use_feature_selection=True,
    use_hyperopt=False,
):
    """
    训练并评估机器学习模型

    参数:
    X: 特征DataFrame
    y: 标签Series
    model_type: 模型类型，可选 'rf' (随机森林)、'xgb' (XGBoost)、'lgbm' (LightGBM)
    feature_names: 特征名列表
    use_cv: 是否使用交叉验证
    use_feature_selection: 是否使用特征选择
    use_hyperopt: 是否使用超参数优化

    返回:
    (model, predictions, accuracy, confusion_matrix, feature_importance): 训练好的模型、预测结果、准确率、混淆矩阵和特征重要性
    """
    # 检查数据是否足够
    if X.empty or len(X) < 10 or len(y) < 10:
        print("错误: 数据量不足，无法训练模型。至少需要10条有效数据。")
        return None, None, 0, None, {}

    # 检查特征数量
    if X.shape[1] == 0:
        print("错误: 没有可用特征进行训练。")
        return None, None, 0, None, {}

    try:
        # 检查是否存在NaN或无穷值
        if X.isnull().any().any() or np.isinf(X).any().any():
            print("警告: 数据中存在NaN或无穷值，尝试修复")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())

        # 数据类型转换
        X = X.astype(float)

        # 确保我们有足够的数据进行训练和测试
        if len(X) < 10:  # 使用一个合理的最小值
            print("警告: 数据量不足，无法进行训练/测试划分")
            return None, None, 0, None, {}

        # 划分训练集和测试集 (保留末尾20%作为测试集)
        split_idx = int(len(X) * 0.8)
        if split_idx < 5:  # 确保至少有5个训练样本
            split_idx = len(X) - 1  # 至少留一个测试样本

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 检查划分的有效性
        if len(X_train) == 0 or len(X_test) == 0:
            print("错误: 无法有效划分训练集和测试集，可能数据量不足")
            return None, None, 0, None, {}

        # 初始化变量
        X_train_selected = X_train
        X_test_selected = X_test
        selected_features = feature_names if feature_names is not None else []

        # 特征选择
        if use_feature_selection and feature_names is not None:
            try:
                selector = VarianceThreshold(threshold=0.01)
                X_train_selected = selector.fit_transform(X_train)
                X_test_selected = selector.transform(X_test)

                # 获取选中的特征索引
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]

                # 显示被保留和移除的特征数
                print(
                    f"选中了 {len(selected_features)} 个特征，移除了 {len(feature_names) - len(selected_features)} 个低方差特征。"
                )

                if len(selected_features) == 0:
                    print("警告: 所有特征都被移除了，改用原始特征")
                    X_train_selected = X_train
                    X_test_selected = X_test
                    selected_features = feature_names
            except Exception as e:
                print(f"特征选择失败: {e}")
                X_train_selected = X_train
                X_test_selected = X_test
                selected_features = feature_names

        # 处理XGBoost和LightGBM的标签
        if model_type in ["xgb", "lgbm"]:
            # 将[-1, 0, 1]映射到[0, 1, 2]
            y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
            y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})
        else:
            y_train_mapped = y_train
            y_test_mapped = y_test

        # 模型选择和超参数调优
        if model_type == "xgb":
            print(f"准备训练XGBoost模型...")
            try:
                import xgboost as xgb

                if use_hyperopt:
                    print("执行XGBoost超参数优化...")
                    param_grid = {
                        "n_estimators": [100, 200],
                        "max_depth": [4, 6, 8],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "subsample": [0.7, 0.8, 0.9],
                        "colsample_bytree": [0.7, 0.8, 0.9],
                    }

                    base_model = xgb.XGBClassifier(
                        objective="multi:softmax",
                        num_class=3,  # 对应标签[0, 1, 2]
                        random_state=42,
                    )

                    tscv = TimeSeriesSplit(n_splits=3)  # 时间序列交叉验证
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        cv=tscv,
                        scoring="accuracy",
                        n_jobs=-1,
                        verbose=1,
                    )

                    grid_search.fit(X_train_selected, y_train_mapped)
                    model = grid_search.best_estimator_
                    print(f"XGBoost最佳参数: {grid_search.best_params_}")
                    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softmax",
                        num_class=3,  # 对应标签[0, 1, 2]
                        random_state=42,
                    )
            except ImportError:
                print("未安装XGBoost，将使用随机森林作为默认模型")
                model_type = "rf"
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=4, random_state=42
                )
                y_train_mapped = y_train
                y_test_mapped = y_test
        elif model_type == "lgbm":
            print(f"准备训练LightGBM模型...")
            try:
                import lightgbm as lgb

                if use_hyperopt:
                    print("执行LightGBM超参数优化...")
                    param_grid = {
                        "n_estimators": [100, 200],
                        "max_depth": [4, 6, 8],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "subsample": [0.7, 0.8, 0.9],
                        "colsample_bytree": [0.7, 0.8, 0.9],
                        "reg_alpha": [0, 0.1, 0.5],
                        "reg_lambda": [0, 1.0, 5.0],
                    }

                    base_model = lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=3,  # 对应标签[0, 1, 2]
                        random_state=42,
                    )

                    tscv = TimeSeriesSplit(n_splits=3)  # 时间序列交叉验证
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        cv=tscv,
                        scoring="accuracy",
                        n_jobs=-1,
                        verbose=1,
                    )

                    grid_search.fit(X_train_selected, y_train_mapped)
                    model = grid_search.best_estimator_
                    print(f"LightGBM最佳参数: {grid_search.best_params_}")
                    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
                else:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multiclass",
                        num_class=3,  # 对应标签[0, 1, 2]
                        random_state=42,
                    )
            except ImportError:
                print("未安装LightGBM，将使用随机森林作为默认模型")
                model_type = "rf"
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=4, random_state=42
                )
                y_train_mapped = y_train
                y_test_mapped = y_test
        else:  # 默认使用随机森林
            print(f"准备训练随机森林模型...")

            if use_hyperopt:
                print("执行随机森林超参数优化...")
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [4, 6, 8, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                }

                base_model = RandomForestClassifier(random_state=42)

                tscv = TimeSeriesSplit(n_splits=3)  # 时间序列交叉验证
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1,
                )

                grid_search.fit(X_train_selected, y_train_mapped)
                model = grid_search.best_estimator_
                print(f"随机森林最佳参数: {grid_search.best_params_}")
                print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=4, random_state=42
                )

        # 交叉验证
        if use_cv and not use_hyperopt:  # 如果使用了超参数优化，已经做过交叉验证了
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(
                    model, X_train_selected, y_train_mapped, cv=tscv, scoring="accuracy"
                )
                print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                print(f"交叉验证时出错: \n{e}")
                return None

        # 训练最终模型
        print(f"训练最终{model_type}模型...")
        if not use_hyperopt:  # 如果没有使用超参数优化，需要再次训练模型
            model.fit(X_train_selected, y_train_mapped)

        # 预测测试集
        y_pred = model.predict(X_test_selected)

        # 如果是XGBoost或LightGBM，需要将预测值映射回[-1, 0, 1]
        if model_type in ["xgb", "lgbm"]:
            y_pred = np.array([{0: -1, 1: 0, 2: 1}[x] for x in y_pred])
            y_test_original = y_test  # 保存原始测试标签
        else:
            y_test_original = y_test

        # 计算准确率
        accuracy = accuracy_score(y_test_original, y_pred) * 100

        # 创建混淆矩阵
        cm = confusion_matrix(y_test_original, y_pred)

        # 计算更多评估指标
        f1 = f1_score(y_test_original, y_pred, average="weighted")
        print(f"F1 分数: {f1:.4f}")

        # 二分类问题时计算ROC-AUC
        if len(np.unique(y_test_original)) == 2:
            try:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test_selected)
                    if model_type in ["xgb", "lgbm"]:
                        # 获取正类的概率
                        if y_pred_proba.shape[1] >= 3:
                            positive_class_proba = y_pred_proba[
                                :, 2
                            ]  # 第3列对应标签1（上涨）
                        else:
                            positive_class_proba = y_pred_proba[:, 1]  # 二分类情况
                    else:
                        positive_class_proba = y_pred_proba[:, 1]

                    # 将标签映射为二分类形式（1为正类，其他为负类）
                    binary_test = (y_test_original == 1).astype(int)
                    roc_auc = roc_auc_score(binary_test, positive_class_proba)
                    print(f"ROC-AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"计算ROC-AUC时出错: {e}")

        # 提取特征重要性
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_importance = sorted(
                zip(selected_features, importance), key=lambda x: x[1], reverse=True
            )
        else:
            feature_importance = []

        return model, y_pred, accuracy, cm, feature_importance

    except Exception as e:
        print(f"训练模型时出错: {e}")
        traceback.print_exc()
        return None


def train_ensemble_model(
    X, y, ensemble_type="voting", base_models=None, feature_names=None
):
    """
    训练集成模型

    参数:
    X: 特征数据
    y: 标签数据
    ensemble_type: 集成类型，'voting' 或 'stacking'
    base_models: 基础模型字典，如未提供则使用默认的RF、XGB和LGBM
    feature_names: 特征名称，用于特征重要性分析

    返回:
    训练好的集成模型和相关评估指标
    """
    try:
        # 检查是否存在NaN或无穷值
        if X.isnull().any().any() or np.isinf(X).any().any():
            print("警告: 数据中存在NaN或无穷值，尝试修复")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())

        # 数据类型转换
        X = X.astype(float)

        # 确保我们有足够的数据进行训练和测试
        if len(X) < 10:  # 使用一个合理的最小值
            print("警告: 数据量不足，无法进行训练/测试划分")
            return None, None, 0, None, {}

        # 划分训练集和测试集 (保留末尾20%作为测试集)
        split_idx = int(len(X) * 0.8)
        if split_idx < 5:  # 确保至少有5个训练样本
            split_idx = len(X) - 1  # 至少留一个测试样本

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 检查划分的有效性
        if len(X_train) == 0 or len(X_test) == 0:
            print("错误: 无法有效划分训练集和测试集，可能数据量不足")
            return None, None, 0, None, {}

        # 特征缩放 (对于集成模型，特别是使用多种不同的基础模型时非常重要)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 如果没有提供基础模型，创建默认的基础模型集
        if base_models is None:
            base_models = {}

            # 随机森林
            base_models["rf"] = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
            )

            # XGBoost
            try:
                import xgboost as xgb

                base_models["xgb"] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,  # L1正则化
                    reg_lambda=1.0,  # L2正则化
                    gamma=0.1,  # 最小损失减少
                    objective="multi:softmax",
                    num_class=3,
                    random_state=42,
                )
            except ImportError:
                print("XGBoost未安装，将不使用XGBoost模型")

            # LightGBM
            try:
                import lightgbm as lgb

                base_models["lgbm"] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,  # L1正则化
                    reg_lambda=1.0,  # L2正则化
                    min_child_samples=5,
                    objective="binary",
                    class_weight="balanced",
                    random_state=42,
                )
            except ImportError:
                print("LightGBM未安装，将不使用LightGBM模型")

            # 添加SVM分类器
            from sklearn.svm import SVC

            base_models["svm"] = SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )

            # 添加逻辑回归
            from sklearn.linear_model import LogisticRegression

            base_models["lr"] = LogisticRegression(
                C=0.1, max_iter=1000, class_weight="balanced", random_state=42
            )

            # 添加KNN分类器
            base_models["knn"] = KNeighborsClassifier(
                n_neighbors=5, weights="distance", p=2  # 欧几里得距离
            )

            # 添加高斯朴素贝叶斯
            from sklearn.naive_bayes import GaussianNB

            base_models["gnb"] = GaussianNB()

            # 添加神经网络
            from sklearn.neural_network import MLPClassifier

            base_models["mlp"] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                random_state=42,
            )

        # 二分类问题，不需要映射
        if len(np.unique(y)) == 2:
            y_train_mapped = y_train
            y_test_mapped = y_test
            is_binary = True
        else:
            # 多分类问题的映射
            y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
            y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})
            is_binary = False

        if ensemble_type == "voting":
            print("训练投票分类器集成模型...")
            # 创建分类器列表
            estimators = [(name, model) for name, model in base_models.items()]

            if len(estimators) < 2:
                print("警告: 投票分类器需要至少2个基础模型，回退到随机森林")
                return train_ml_model(
                    X, y, model_type="rf", feature_names=feature_names
                )

            # 创建投票分类器，设置权重
            weights = {
                "rf": 2,  # 随机森林更高权重
                "xgb": 2,  # XGBoost更高权重
                "lgbm": 2,  # LightGBM更高权重
                "svm": 1,  # SVM较低权重
                "lr": 1,  # 逻辑回归较低权重
                "knn": 1,  # KNN较低权重
                "gnb": 1,  # 朴素贝叶斯较低权重
                "mlp": 1.5,  # 神经网络中等权重
            }

            # 提取当前模型的权重
            model_weights = [weights.get(name, 1) for name, _ in estimators]

            # 创建投票分类器
            ensemble_model = VotingClassifier(
                estimators=estimators,
                voting="soft",  # 使用概率投票
                weights=model_weights,  # 设置权重
                flatten_transform=True,
            )

            # 训练模型
            ensemble_model.fit(X_train_scaled, y_train_mapped)

            # 预测
            y_pred_proba = ensemble_model.predict_proba(X_test_scaled)
            y_pred = ensemble_model.predict(X_test_scaled)

            # 如果是多分类，将预测结果映射回原始类别
            if not is_binary:
                y_pred = np.array([{0: -1, 1: 0, 2: 1}[x] for x in y_pred])

        elif ensemble_type == "stacking":
            print("训练堆叠分类器集成模型...")
            # 创建分类器列表
            estimators = [(name, model) for name, model in base_models.items()]

            if len(estimators) < 2:
                print("警告: 堆叠分类器需要至少2个基础模型，回退到随机森林")
                return train_ml_model(
                    X, y, model_type="rf", feature_names=feature_names
                )

            # 创建堆叠分类器，使用LightGBM或XGBoost作为元分类器
            meta_classifier = None
            try:
                import lightgbm as lgb

                meta_classifier = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
                )
                print("使用LightGBM作为元分类器")
            except ImportError:
                try:
                    import xgboost as xgb

                    meta_classifier = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=42,
                    )
                    print("使用XGBoost作为元分类器")
                except ImportError:
                    # 如果两者都不可用，使用随机森林
                    meta_classifier = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=4,
                        min_samples_split=5,
                        class_weight="balanced",
                        random_state=42,
                    )
                    print("使用随机森林作为元分类器")

            ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_classifier,
                cv=5,  # 5折交叉验证
                stack_method="predict_proba",
                n_jobs=-1,  # 使用所有CPU核心
            )

            # 训练模型
            ensemble_model.fit(X_train_scaled, y_train_mapped)

            # 预测
            y_pred = ensemble_model.predict(X_test_scaled)

            # 如果是多分类，将预测结果映射回原始类别
            if not is_binary:
                y_pred = np.array([{0: -1, 1: 0, 2: 1}[x] for x in y_pred])

        else:
            print(f"不支持的集成类型: {ensemble_type}，回退到随机森林")
            return train_ml_model(X, y, model_type="rf", feature_names=feature_names)

        # 计算准确率
        y_test_original = y_test  # 保存原始测试标签
        accuracy = accuracy_score(y_test_original, y_pred) * 100

        # 创建混淆矩阵
        cm = confusion_matrix(y_test_original, y_pred)

        # 计算F1分数
        f1 = f1_score(y_test_original, y_pred, average="weighted")
        print(f"F1 分数: {f1:.4f}")

        # 二分类问题时计算ROC-AUC
        if len(np.unique(y_test_original)) == 2 and hasattr(
            ensemble_model, "predict_proba"
        ):
            try:
                # 对于二分类问题，获取正类的概率
                y_pred_proba = ensemble_model.predict_proba(X_test_scaled)

                # 确定正类的索引
                if not is_binary and y_pred_proba.shape[1] >= 3:  # 多分类映射情况
                    positive_class_idx = 2  # 类别1（上涨）的索引是2
                else:
                    positive_class_idx = 1  # 二分类中，正类通常是索引1

                if y_pred_proba.shape[1] > positive_class_idx:
                    # 获取正类概率
                    positive_class_proba = y_pred_proba[:, positive_class_idx]

                    # 将原始测试标签转换为二分类形式
                    binary_test = (y_test_original == 1).astype(int)

                    # 计算ROC-AUC
                    roc_auc = roc_auc_score(binary_test, positive_class_proba)
                    print(f"ROC-AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"计算ROC-AUC时出错: {e}")

        # 尝试获取特征重要性
        feature_importance = []
        try:
            if ensemble_type == "voting":
                # 对于投票分类器，尝试从第一个基础模型获取特征重要性
                for name, model in ensemble_model.estimators_:
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        feature_importance = sorted(
                            zip(feature_names, importances),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        break
            elif ensemble_type == "stacking":
                # 对于堆叠分类器，尝试从基础模型获取平均特征重要性
                importances = np.zeros(len(feature_names))
                count = 0
                for name, model in ensemble_model.estimators_:
                    if hasattr(model, "feature_importances_"):
                        importances += model.feature_importances_
                        count += 1
                if count > 0:
                    importances /= count
                    feature_importance = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1],
                        reverse=True,
                    )
        except Exception as e:
            print(f"无法提取特征重要性: {e}")
            feature_importance = []

        print(f"集成模型 {ensemble_type} 训练完成")
        print(f"模型组合: {', '.join(base_models.keys())}")
        print(f"准确率: {accuracy:.2f}%")

        return ensemble_model, y_pred, accuracy, cm, feature_importance

    except Exception as e:
        print(f"训练集成模型时出错: {e}")
        traceback.print_exc()
        print("回退到随机森林模型")
        return train_ml_model(X, y, model_type="rf", feature_names=feature_names)


def predict_signals(features, predictions, stock_code):
    """
    根据模型预测结果生成交易信号并可视化

    参数:
    features: 特征数据
    predictions: 预测结果
    stock_code: 股票代码

    返回:
    交易信号图表
    """
    try:
        # 创建结果图表
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

        # 获取测试数据部分 (最后20%的数据)
        test_size = len(predictions)
        test_start_idx = len(features) - test_size

        # 获取测试数据的日期和价格
        dates = None
        if "date" in features.columns:
            dates = features["date"].iloc[test_start_idx:].reset_index(drop=True)
        else:
            # 如果没有日期列，使用索引
            dates = np.arange(test_size)

        close_prices = features["close"].iloc[test_start_idx:].reset_index(drop=True)

        # 绘制价格走势图
        ax.plot(dates, close_prices, label="收盘价", color="blue", linewidth=2)

        # 提取交易信号
        buy_signals = dates[predictions == 1].values
        sell_signals = dates[predictions == -1].values

        # 获取对应日期的价格
        buy_prices = close_prices.iloc[predictions == 1].values
        sell_prices = close_prices.iloc[predictions == -1].values

        # 绘制交易信号
        ax.scatter(
            buy_signals,
            buy_prices,
            color="green",
            label="买入信号(多)",
            marker="^",
            s=100,
        )
        ax.scatter(
            sell_signals,
            sell_prices,
            color="red",
            label="卖出信号(空)",
            marker="v",
            s=100,
        )

        # 获取置信度信息
        if "预测概率" in features.columns:
            high_conf = (
                features["预测概率"].iloc[test_start_idx:].reset_index(drop=True)
            )
            high_conf_signals = dates[high_conf > 0.7].values
            high_conf_prices = close_prices.iloc[high_conf > 0.7].values
            ax.scatter(
                high_conf_signals,
                high_conf_prices,
                color="blue",
                label="高置信度预测",
                marker="*",
                s=120,
                alpha=0.5,
            )

        # 添加均线作为参考
        if "ma20" in features.columns:
            ax.plot(
                dates,
                features["ma20"].iloc[test_start_idx:].reset_index(drop=True),
                "--",
                color="purple",
                alpha=0.7,
                linewidth=1,
                label="20日均线",
            )
        if "ma10" in features.columns:
            ax.plot(
                dates,
                features["ma10"].iloc[test_start_idx:].reset_index(drop=True),
                "--",
                color="orange",
                alpha=0.7,
                linewidth=1,
                label="10日均线",
            )

        # 添加标题和标签
        plt.title(f"{stock_code} 股价走势与交易信号", fontsize=15)
        plt.xlabel("日期", fontsize=12)
        plt.ylabel("价格", fontsize=12)

        # 日期格式化
        if hasattr(dates, "iloc") and len(dates) > 0:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, len(dates) // 10))
            )
            plt.gcf().autofmt_xdate()

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # 创建保存目录
        os.makedirs("predictions", exist_ok=True)

        # 保存图表
        save_path = f"predictions/{stock_code}_prediction.png"
        plt.savefig(save_path, dpi=300)
        print(f"预测结果图表已保存至: {save_path}")

        # 如果在notebook中，显示图表
        plt.show()

        return None

    except Exception as e:
        print(f"生成交易信号图表时出错: {e}")
        traceback.print_exc()
        return None


def visualize_prediction(df, stock_code, prediction_days, model_type=None):
    """
    绘制预测结果与股价走势

    参数:
    df: 包含预测结果的数据框
    stock_code: 股票代码
    prediction_days: 预测天数
    model_type: 使用的模型类型
    """
    # 检查数据框是否包含必需列
    required_cols = ["close", "预测信号"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"数据缺少必要列：{missing_cols}，无法绘制预测图表")
        return

    try:
        # 设置中文字体支持
        import matplotlib as mpl
        from matplotlib.font_manager import FontProperties

        # 使用预先配置的字体 (install_fonts.py已经配置好了matplotlibrc)

        # 备选中文字体列表
        chinese_fonts = [
            "Arial Unicode MS",
            "PingFang HK",
            "Heiti TC",
            "STHeiti",
            "SimSong",
            "STFangsong",
        ]

        # 方法1: 从重要字体中找到第一个可用的
        font_found = False

        for font_name in chinese_fonts:
            try:
                # 尝试获取这个字体的字体文件路径
                font_path = mpl.font_manager.findfont(
                    font_name, fallback_to_default=False
                )
                if font_path:
                    # 创建字体属性对象
                    font_prop = FontProperties(fname=font_path)
                    print(f"使用中文字体: {font_name}")
                    font_found = True
                    break
            except:
                continue

        # 如果没有找到适合的字体，使用回退方案
        if not font_found:
            print("警告: 未找到合适的中文字体，将使用系统默认字体")
            font_prop = None

        # 确保负号能正确显示
        mpl.rcParams["axes.unicode_minus"] = False

        # 日期格式转换 - 确保日期是datetime类型，便于格式化
        df = df.copy()  # 创建副本，避免修改原始数据
        if isinstance(df["trade_date"].iloc[0], str):
            df["trade_date"] = pd.to_datetime(df["trade_date"])

        plt.figure(figsize=(16, 10))  # 稍微增加宽度

        # 绘制股价走势
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(
            df["trade_date"],
            df["close"],
            label="Close" if font_prop is None else "收盘价",
            linewidth=1.5,
        )

        # 标记买入信号
        buy_signals = df[df["预测信号"] == 1]
        plt.scatter(
            buy_signals["trade_date"],
            buy_signals["close"],
            color="green",
            label="Buy Signal" if font_prop is None else "买入信号",
            marker="^",
            s=100,
        )

        # 标记卖出信号
        sell_signals = df[df["预测信号"] == -1]
        plt.scatter(
            sell_signals["trade_date"],
            sell_signals["close"],
            color="red",
            label="Sell Signal" if font_prop is None else "卖出信号",
            marker="v",
            s=100,
        )

        # 如果有置信度信息，显示置信度高的点
        if "预测概率" in df.columns:
            high_conf = df[df["预测概率"] > 0.7]
            plt.scatter(
                high_conf["trade_date"],
                high_conf["close"],
                color="blue",
                label="High Confidence" if font_prop is None else "高置信度预测",
                marker="*",
                s=120,
                alpha=0.5,
            )

        # 添加均线作为参考
        plt.plot(
            df["trade_date"],
            df["SMA20"],
            "--",
            color="purple",
            alpha=0.7,
            linewidth=1,
            label="20-day MA" if font_prop is None else "20日均线",
        )
        plt.plot(
            df["trade_date"],
            df["SMA10"],
            "--",
            color="orange",
            alpha=0.7,
            linewidth=1,
            label="10-day MA" if font_prop is None else "10日均线",
        )

        # 添加标题和标签 - 使用正确的语言版本
        model_info = f"Model: {model_type}" if model_type else ""
        if font_prop:
            title = (
                f"{stock_code} 股价预测 (预测天数: {prediction_days}天) {model_info}"
            )
            xlabel = "日期"
            ylabel = "价格"
        else:
            title = (
                f"{stock_code} Stock Prediction ({prediction_days}-day) {model_info}"
            )
            xlabel = "Date"
            ylabel = "Price"

        plt.title(title, fontproperties=font_prop, fontsize=15)
        plt.xlabel(xlabel, fontproperties=font_prop, fontsize=12)
        plt.ylabel(ylabel, fontproperties=font_prop, fontsize=12)

        # 日期格式化
        date_format = mdates.DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_format)

        # 自动选择合适的日期间隔
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30, ha="right", fontsize=10)

        # 添加网格线
        plt.grid(True, which="major", linestyle="-", alpha=0.3)
        plt.grid(True, which="minor", linestyle="--", alpha=0.1)

        # 更好的图例位置
        if font_prop:
            plt.legend(loc="best", fontsize=10, prop=font_prop)
        else:
            plt.legend(loc="best", fontsize=10)

        # 绘制预测信号与实际信号对比
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(
            df["trade_date"],
            df["signal"],
            label="Actual Signal" if font_prop is None else "实际信号",
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            df["trade_date"],
            df["预测信号"],
            label="Predicted Signal" if font_prop is None else "预测信号",
            color="orange",
            linewidth=1.5,
        )

        # 使用字体属性设置标题和标签
        if font_prop:
            plt.title(
                "Actual vs Predicted Signals", fontproperties=font_prop, fontsize=14
            )
            plt.xlabel("日期", fontproperties=font_prop, fontsize=12)
            plt.ylabel(
                "信号 (1=买入, 0=持有, -1=卖出)", fontproperties=font_prop, fontsize=12
            )
        else:
            plt.title("Actual vs Predicted Signals", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Signal (1=Buy, 0=Hold, -1=Sell)", fontsize=12)

        # 设置Y轴刻度为整数
        ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax2.set_ylim(-1.2, 1.2)  # 稍微扩展y轴范围，使点不会太靠近边缘

        # 设置y轴刻度标签
        if font_prop:
            plt.yticks(
                [-1, 0, 1], ["卖出(-1)", "持有(0)", "买入(1)"], fontproperties=font_prop
            )
        else:
            plt.yticks([-1, 0, 1], ["Sell(-1)", "Hold(0)", "Buy(1)"])

        # 使用相同的日期格式
        ax2.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30, ha="right", fontsize=10)

        # 明确标记y轴值，便于阅读
        plt.yticks([-1, 0, 1], ["卖出(-1)", "持有(0)", "买入(1)"])

        # 添加网格线
        plt.grid(True, which="major", linestyle="-", alpha=0.3)

        plt.legend(loc="best", fontsize=10)

        plt.tight_layout(pad=2.0)  # 增加边距

        # 尝试创建images目录(如果不存在)
        if not os.path.exists("images"):
            os.makedirs("images")

        # 保存高分辨率图像
        plt.savefig(
            f"images/{stock_code}_ml_prediction_{prediction_days}days.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"图表已保存至: images/{stock_code}_ml_prediction_{prediction_days}days.png"
        )
        plt.close()

    except Exception as e:
        print(f"可视化预测结果时出错: {e}")
        traceback.print_exc()


def evaluate_prediction_performance(df, prediction_days):
    """
    评估预测性能

    参数:
    df: 包含预测结果的数据框
    prediction_days: 预测天数

    返回:
    最新的交易信号
    """
    # 获取最近的预测信号
    last_signal = df["预测信号"].iloc[-1]
    last_confidence = df["预测概率"].iloc[-1] if "预测概率" in df.columns else 0

    signal_text = (
        "买入" if last_signal == 1 else ("卖出" if last_signal == -1 else "观望")
    )
    confidence_text = (
        f" (置信度: {last_confidence:.2f})" if "预测概率" in df.columns else ""
    )

    print(f"\n最新交易信号: {signal_text}{confidence_text}")

    # 计算预测准确率
    correct_predictions = (df["预测信号"] == df["signal"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"预测准确率: {accuracy:.4f}")

    # 计算买入信号准确率
    buy_signals = df[df["预测信号"] == 1]
    if len(buy_signals) > 0:
        correct_buys = (buy_signals["signal"] == 1).sum()
        buy_accuracy = correct_buys / len(buy_signals)
        print(f"买入信号准确率: {buy_accuracy:.4f}")

    # 计算卖出信号准确率
    sell_signals = df[df["预测信号"] == -1]
    if len(sell_signals) > 0:
        correct_sells = (sell_signals["signal"] == -1).sum()
        sell_accuracy = correct_sells / len(sell_signals)
        print(f"卖出信号准确率: {sell_accuracy:.4f}")

    # 分析预测错误的情况
    error_cases = df[df["预测信号"] != df["signal"]]
    error_rate = len(error_cases) / total_predictions
    print(f"预测错误率: {error_rate:.4f}")

    # 计算买入信号收益率
    buy_returns = []
    for idx in buy_signals.index:
        if idx + prediction_days < len(df.index):  # 确保有足够的未来数据计算收益
            future_idx = df.index[list(df.index).index(idx) + prediction_days]
            if future_idx in df.index:
                future_return = df["close"].loc[future_idx] / df["close"].loc[idx] - 1
                buy_returns.append(future_return)

    if buy_returns:
        avg_buy_return = np.mean(buy_returns) * 100
        print(f"买入信号{prediction_days}天后平均收益率: {avg_buy_return:.2f}%")

    # 计算止损止盈效果
    if len(buy_returns) > 0:
        stop_loss = -0.05  # 5%止损
        stop_profit = 0.08  # 8%止盈

        # 模拟止损止盈
        adjusted_returns = []
        for ret in buy_returns:
            if ret <= stop_loss:
                adjusted_returns.append(stop_loss)
            elif ret >= stop_profit:
                adjusted_returns.append(stop_profit)
            else:
                adjusted_returns.append(ret)

        avg_adjusted_return = np.mean(adjusted_returns) * 100
        print(
            f"应用止损({stop_loss*100}%)止盈({stop_profit*100}%)后的平均收益率: {avg_adjusted_return:.2f}%"
        )

    # 显示混淆矩阵
    conf_matrix = confusion_matrix(df["signal"], df["预测信号"])
    print("\n混淆矩阵:")
    print(conf_matrix)

    return signal_text


def add_index_features(df, index_code):
    """
    添加指数相关特征

    参数:
    df: 股票数据DataFrame
    index_code: 指数代码

    返回:
    添加了指数特征的DataFrame
    """
    print(f"添加指数特征: {index_code}")

    try:
        # 获取指数数据
        first_date = df["date"].min()
        last_date = df["date"].max()

        index_df = get_stock_data(index_code, first_date, last_date)

        if index_df is None or len(index_df) == 0:
            print(f"未能获取{index_code}数据")
            return df

        # 确保两个数据框都有日期列
        if "date" not in index_df.columns:
            print("index_df missing date column")
            return df

        # 使用日期列进行合并
        result_df = df.merge(
            index_df[["date", "close", "volume", "pctChg"]],
            on="date",
            how="left",
            suffixes=("", "_index"),
        )

        # 重命名指数特征
        result_df = result_df.rename(
            columns={
                "close_index": "index_close",
                "volume_index": "index_volume",
                "pctChg_index": "index_pctChg",
            }
        )

        # 填充缺失值
        for col in ["index_close", "index_volume", "index_pctChg"]:
            if col in result_df.columns:
                result_df[col] = result_df[col].ffill()

        # 计算指数的一些技术指标
        # 1. 指数涨跌幅
        result_df["index_change"] = result_df["index_close"].pct_change()

        # 2. 指数5日、10日、20日涨跌幅
        result_df["index_change_5d"] = result_df["index_close"].pct_change(5)
        result_df["index_change_10d"] = result_df["index_close"].pct_change(10)
        result_df["index_change_20d"] = result_df["index_close"].pct_change(20)

        # 3. 指数RSI
        delta = result_df["index_close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result_df["index_rsi"] = 100 - (100 / (1 + rs))

        return result_df

    except Exception as e:
        print(f"添加指数特征时出错: {e}")
        traceback.print_exc()
        return df


def add_sector_features(df, sector_code):
    """
    添加相关板块指数特征

    参数:
    df: 原始数据
    sector_code: 板块指数代码

    返回:
    添加了板块特征的DataFrame
    """
    try:
        # 复制数据以避免修改原始数据
        result_df = df.copy()

        print(f"获取板块数据: {sector_code}")

        # 获取板块指数数据
        first_date = df["date"].min()
        last_date = df["date"].max()

        sector_df = get_stock_data(sector_code, first_date, last_date)

        if sector_df is None or len(sector_df) == 0:
            print(f"无法获取板块指数数据: {sector_code}")
            return result_df

        # 确保两个数据框都有日期列
        if "date" not in sector_df.columns:
            print("sector_df missing date column")
            return result_df

        # 使用日期列进行合并
        result_df = result_df.merge(
            sector_df[["date", "close", "volume", "pctChg"]],
            on="date",
            how="left",
            suffixes=("", "_sector"),
        )

        # 重命名板块特征
        result_df = result_df.rename(
            columns={
                "close_sector": "sector_close",
                "volume_sector": "sector_volume",
                "pctChg_sector": "sector_pctChg",
            }
        )

        # 填充缺失值
        for col in ["sector_close", "sector_volume", "sector_pctChg"]:
            if col in result_df.columns:
                result_df[col] = result_df[col].ffill()

        # 计算板块指数的一些技术指标
        # 1. 板块指数涨跌幅
        result_df["sector_change"] = result_df["sector_close"].pct_change()

        # 2. 板块5日、10日、20日涨跌幅
        result_df["sector_change_5d"] = result_df["sector_close"].pct_change(5)
        result_df["sector_change_10d"] = result_df["sector_close"].pct_change(10)
        result_df["sector_change_20d"] = result_df["sector_close"].pct_change(20)

        # 3. 板块RSI
        delta = result_df["sector_close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result_df["sector_rsi"] = 100 - (100 / (1 + rs))

        return result_df

    except Exception as e:
        print(f"添加板块特征时出错: {e}")
        traceback.print_exc()
        return df


def generate_trading_advice(predictions, stock_code):
    """
    基于模型预测结果生成交易建议

    参数:
    predictions: 预测结果数组
    stock_code: 股票代码

    返回:
    交易建议字符串
    """
    # 获取最近的预测
    if len(predictions) == 0:
        return "无法生成建议：预测结果为空"

    latest_prediction = predictions[-1]

    # 基于最新预测生成建议
    if latest_prediction == 1:
        advice = (
            f"🟢 买入建议 🟢\n股票: {stock_code}\n理由: 模型预测价格将在未来上涨超过2%"
        )
    elif latest_prediction == -1:
        advice = (
            f"🔴 卖出建议 🔴\n股票: {stock_code}\n理由: 模型预测价格将在未来下跌超过2%"
        )
    else:
        advice = f"🟡 观望建议 🟡\n股票: {stock_code}\n理由: 模型预测价格将在未来波动不超过2%"

    # 添加模型准确率的警告
    advice += "\n\n注意：此建议仅供参考，模型预测准确率有限"

    return advice


def calculate_technical_indicators(df):
    """
    计算技术指标

    参数:
    df: 股票数据DataFrame

    返回:
    添加了技术指标的DataFrame
    """
    try:
        # 复制数据以避免修改原始数据
        result_df = df.copy()

        # 1. 移动平均线指标
        result_df["ma5"] = result_df["close"].rolling(window=5).mean()
        result_df["ma10"] = result_df["close"].rolling(window=10).mean()
        result_df["ma20"] = result_df["close"].rolling(window=20).mean()
        result_df["ma30"] = result_df["close"].rolling(window=30).mean()
        result_df["ma60"] = result_df["close"].rolling(window=60).mean()
        result_df["ma120"] = result_df["close"].rolling(window=120).mean()

        # 2. MACD指标
        exp1 = result_df["close"].ewm(span=12, adjust=False).mean()
        exp2 = result_df["close"].ewm(span=26, adjust=False).mean()
        result_df["macd"] = exp1 - exp2
        result_df["macd_signal"] = result_df["macd"].ewm(span=9, adjust=False).mean()
        result_df["macd_hist"] = result_df["macd"] - result_df["macd_signal"]

        # 3. RSI指标
        delta = result_df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result_df["rsi_14"] = 100 - (100 / (1 + rs))

        # 4. Bollinger带指标
        result_df["bollinger_mid"] = result_df["close"].rolling(window=20).mean()
        result_df["bollinger_std"] = result_df["close"].rolling(window=20).std()
        result_df["bollinger_upper"] = (
            result_df["bollinger_mid"] + 2 * result_df["bollinger_std"]
        )
        result_df["bollinger_lower"] = (
            result_df["bollinger_mid"] - 2 * result_df["bollinger_std"]
        )
        result_df["bollinger_width"] = (
            result_df["bollinger_upper"] - result_df["bollinger_lower"]
        ) / result_df["bollinger_mid"]

        # 5. 相对于均线的位置百分比
        result_df["ma5_position"] = (
            (result_df["close"] - result_df["ma5"]) / result_df["ma5"] * 100
        )
        result_df["ma10_position"] = (
            (result_df["close"] - result_df["ma10"]) / result_df["ma10"] * 100
        )
        result_df["ma20_position"] = (
            (result_df["close"] - result_df["ma20"]) / result_df["ma20"] * 100
        )
        result_df["ma30_position"] = (
            (result_df["close"] - result_df["ma30"]) / result_df["ma30"] * 100
        )
        result_df["ma60_position"] = (
            (result_df["close"] - result_df["ma60"]) / result_df["ma60"] * 100
        )

        # 6. ATR指标 (真实波幅)
        high_low = result_df["high"] - result_df["low"]
        high_close = np.abs(result_df["high"] - result_df["close"].shift())
        low_close = np.abs(result_df["low"] - result_df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result_df["atr"] = true_range.rolling(window=14).mean()

        # 7. 价格动量指标
        result_df["momentum_5d"] = result_df["close"] / result_df["close"].shift(5) - 1
        result_df["momentum_10d"] = (
            result_df["close"] / result_df["close"].shift(10) - 1
        )
        result_df["momentum_20d"] = (
            result_df["close"] / result_df["close"].shift(20) - 1
        )

        # 8. 价格变化率
        result_df["price_change_1d"] = result_df["close"].pct_change()
        result_df["price_change_5d"] = result_df["close"].pct_change(5)
        result_df["price_change_10d"] = result_df["close"].pct_change(10)
        result_df["price_change_20d"] = result_df["close"].pct_change(20)

        # 9. 成交量特征
        result_df["volume_change"] = result_df["volume"].pct_change()
        result_df["volume_ma_5"] = result_df["volume"].rolling(window=5).mean()
        result_df["volume_ma_ratio"] = result_df["volume"] / result_df["volume_ma_5"]

        # 10. 星期几特征
        if "date" in result_df.columns:
            result_df["dayofweek"] = pd.to_datetime(result_df["date"]).dt.dayofweek
            # 使用独热编码处理星期几
            for i in range(5):  # 只考虑工作日（0-4对应周一至周五）
                result_df[f"day_{i}"] = (result_df["dayofweek"] == i).astype(int)

        # 11. KDJ指标
        low_min = result_df["low"].rolling(9).min()
        high_max = result_df["high"].rolling(9).max()
        result_df["kdj_k"] = 50.0
        result_df["kdj_d"] = 50.0

        for i in range(9, len(result_df)):
            if (
                i < len(result_df)
                and i - 1 >= 0
                and i < len(high_max)
                and i < len(low_min)
            ):
                if high_max.iloc[i] - low_min.iloc[i] > 0:
                    if i < len(result_df.index) and i - 1 < len(result_df.index):
                        result_df.loc[result_df.index[i], "kdj_k"] = (
                            result_df.loc[result_df.index[i - 1], "kdj_k"] * 2 / 3
                            + 100
                            * (
                                result_df.loc[result_df.index[i], "close"]
                                - low_min.iloc[i]
                            )
                            / (high_max.iloc[i] - low_min.iloc[i])
                            * 1
                            / 3
                        )
                        result_df.loc[result_df.index[i], "kdj_d"] = (
                            result_df.loc[result_df.index[i - 1], "kdj_d"] * 2 / 3
                            + result_df.loc[result_df.index[i], "kdj_k"] * 1 / 3
                        )

        result_df["kdj_j"] = 3 * result_df["kdj_k"] - 2 * result_df["kdj_d"]

        # 12. Williams %R指标
        result_df["williams_r"] = (
            (high_max - result_df["close"]) / (high_max - low_min) * -100
        )

        # 13. CCI（商品通道指数）
        typical_price = (result_df["high"] + result_df["low"] + result_df["close"]) / 3
        tp_ma = typical_price.rolling(window=20).mean()
        tp_std = typical_price.rolling(window=20).std()
        result_df["cci"] = (typical_price - tp_ma) / (0.015 * tp_std)

        # 14. Ichimoku云图指标
        result_df["ichimoku_tenkan"] = (
            result_df["high"].rolling(window=9).max()
            + result_df["low"].rolling(window=9).min()
        ) / 2
        result_df["ichimoku_kijun"] = (
            result_df["high"].rolling(window=26).max()
            + result_df["low"].rolling(window=26).min()
        ) / 2

        # 处理财务指标
        if "peTTM" in result_df.columns:
            result_df["pe_percentile"] = (
                result_df["peTTM"]
                .rolling(60)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
            )
        if "pbMRQ" in result_df.columns:
            result_df["pb_percentile"] = (
                result_df["pbMRQ"]
                .rolling(60)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
            )

        return result_df

    except Exception as e:
        print(f"计算技术指标时出错: {e}")
        traceback.print_exc()
        return df


def main(args=None):
    """
    主函数，处理参数和运行预测
    """
    parser = argparse.ArgumentParser(description="A股量化交易 - 增强版机器学习预测模型")
    parser.add_argument(
        "-s",
        "--stock",
        type=str,
        required=True,
        help="股票代码，如 sh.600036",
    )
    parser.add_argument(
        "-d", "--days", type=int, default=365, help="历史数据天数，默认365天"
    )
    parser.add_argument(
        "-f", "--forecast", type=int, default=5, help="预测未来几天的走势，默认5天"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="xgb",
        choices=["rf", "xgb", "lgbm", "voting", "stacking"],
        help="使用的模型类型: rf=随机森林, xgb=XGBoost, lgbm=LightGBM, voting=投票集成, stacking=堆叠集成",
    )
    parser.add_argument("--no-cv", action="store_true", help="不使用交叉验证")
    parser.add_argument("--feature-selection", action="store_true", help="使用特征选择")
    parser.add_argument(
        "--feature-method",
        type=str,
        default="variance",
        choices=["variance", "kbest", "combined"],
        help="特征选择方法: variance=方差选择, kbest=K优特征, combined=两者结合",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.001,
        help="方差特征选择阈值，默认0.001",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.85,
        help="相关性特征选择阈值，默认0.85，数值越小越严格",
    )
    parser.add_argument("--use-pca", action="store_true", help="使用PCA降维")
    parser.add_argument(
        "--pca-components",
        type=float,
        default=0.95,
        help="PCA保留方差比例(0-1)或组件数量(>1)",
    )
    parser.add_argument("--hyperopt", action="store_true", help="使用超参数优化")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="学习率，用于XGBoost和LightGBM",
    )
    parser.add_argument(
        "--num-leaves", type=int, default=31, help="叶子节点数，用于LightGBM"
    )
    parser.add_argument("--max-depth", type=int, default=6, help="树的最大深度")

    # 解析参数
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # 显示设置参数
    print("===== A股量化交易预测分析工具增强版 =====")
    print(f"股票代码: {args.stock}")
    print(f"历史数据天数: {args.days}")
    print(f"预测天数: {args.forecast}")
    print(f"模型类型: {args.model}")
    print(f"特征选择: {'启用' if args.feature_selection else '禁用'}")
    if args.feature_selection:
        print(f"特征选择方法: {args.feature_method}")
        print(f"相关性阈值: {args.correlation_threshold}")
    print(f"使用PCA降维: {'是' if args.use_pca else '否'}")
    if args.use_pca:
        print(f"PCA组件/方差: {args.pca_components}")
    print(f"超参数优化: {'启用' if args.hyperopt else '禁用'}")
    print("=====================================")

    # 设置当前日期
    end_date = datetime.now().strftime("%Y-%m-%d")
    # 计算开始日期 (当前日期往前推days天)
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"获取 {args.stock} 从 {start_date} 到 {end_date} 的数据")

    # 获取股票数据
    df = get_stock_data(args.stock, start_date, end_date)
    if df is None or len(df) == 0:
        print(f"无法获取股票 {args.stock} 的数据")
        return

    print(f"获取到 {len(df)} 条股票数据")

    # 添加上证指数作为参考
    print("添加指数特征: sh.000001")
    df = add_index_features(df, "sh.000001")

    # 添加板块指数特征 (以上证50为例)
    print("获取板块数据: sh.000016")
    df = add_sector_features(df, "sh.000016")

    # 准备特征和标签
    print("准备特征和标签...")
    X, y, feature_names = prepare_features(
        df,
        prediction_days=args.forecast,
        feature_selection=args.feature_selection,
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        feature_selection_method=args.feature_method,
    )

    # 训练模型
    if args.model in ["voting", "stacking"]:
        # 集成模型
        model, predictions, accuracy, cm, feature_importance = train_ensemble_model(
            X,
            y,
            ensemble_type=args.model,
            base_models=None,
            feature_names=feature_names,
        )
    else:
        # 单一模型
        model, predictions, accuracy, cm, feature_importance = train_ml_model(
            X,
            y,
            model_type=args.model,
            feature_names=feature_names,
            use_cv=not args.no_cv,
            use_feature_selection=args.feature_selection,
            use_hyperopt=args.hyperopt,
        )

    if model is None:
        print("模型训练失败")
        return

    # 显示结果
    print(f"\n预测准确率: {accuracy:.2f}%\n")

    print("混淆矩阵:")
    print(cm)
    print()

    # 显示特征重要性
    if feature_importance and len(feature_importance) > 0:
        print("特征重要性 (前10个):")
        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
        print()

    # 生成交易建议
    advice = generate_trading_advice(predictions, args.stock)
    print("=== 当前交易建议 ===")
    print(advice)
    print("=====================")

    df["预测信号"] = predictions.map({0: -1, 1: 0, 2: 1})

    # 绘制预测结果
    visualize_prediction(df, args.stock, args.forecast, args.model)


if __name__ == "__main__":
    main()
