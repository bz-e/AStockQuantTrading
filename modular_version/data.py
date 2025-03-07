#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Module - Responsible for stock data retrieval, preprocessing and cleaning

This module contains functions for retrieving stock and index data from BaoStock API,
and performs necessary data cleaning and preprocessing for subsequent feature engineering
and model training.
"""

import numpy as np
import pandas as pd
import baostock as bs
import traceback
import time
from datetime import datetime, timedelta
import logging


def get_stock_data(stock_code, start_date=None, end_date=None, retry_count=3, add_basic_data=True):
    """
    获取股票历史数据并进行全面的数据清洗和预处理
    
    该函数通过BaoStock API获取指定股票的历史K线数据，并执行一系列数据清洗和预处理步骤，
    包括异常值处理、缺失值填充、数据类型转换等，以确保数据适用于后续的特征工程和模型训练。
    支持自动重试机制，增强了数据获取的稳定性。
    
    参数：
        stock_code (str): 股票代码，格式为交易所代码+股票代码，例如：
                        - 'sh.600036'：招商银行（上海证券交易所）
                        - 'sz.000001'：平安银行（深圳证券交易所）
                        - 'sh.000001'：上证指数
                        - 'sz.399001'：深证成指
        
        start_date (str or datetime, optional): 开始日期，格式为'YYYY-MM-DD'或datetime对象。
                                             如未指定，默认为当前日期前365天。
        
        end_date (str or datetime, optional): 结束日期，格式为'YYYY-MM-DD'或datetime对象。
                                           如未指定，默认为当前日期。
        
        retry_count (int): 数据获取失败时的重试次数，默认为3次。当网络不稳定或遇到临时服务
                         中断时，重试机制能提高数据获取成功率。
        
        add_basic_data (bool): 是否尝试获取额外的基本面数据，例如ROE、净利润率等，默认为True。
                             请注意，部分股票可能不提供这些额外数据。
    
    返回：
        pandas.DataFrame or None: 包含以下列的DataFrame：
                                - 'date': 交易日期，已转换为datetime格式
                                - 'code': 股票代码
                                - 'open': 开盘价，数值类型
                                - 'high': 最高价，数值类型
                                - 'low': 最低价，数值类型
                                - 'close': 收盘价，数值类型
                                - 'volume': 成交量，数值类型
                                - 'amount': 成交额，数值类型
                                - 'adjustflag': 复权类型，3表示后复权
                                - 'turn': 换手率，数值类型
                                - 'tradestatus': 交易状态
                                - 'pctChg': 涨跌幅(%)，数值类型
                                - 'peTTM': 市盈率TTM，数值类型
                                - 'pbMRQ': 市净率，数值类型
                                - 'psTTM': 市销率TTM，数值类型
                                - 'pcfNcfTTM': 市现率TTM，数值类型
                                - 'isST': 是否ST股，1表示是，0表示否
                                
                                如果add_basic_data=True且数据可获取，还可能包含：
                                - 'roeTTM': 净资产收益率TTM
                                - 'roa': 总资产收益率
                                - 'netMargin': 净利率
                                - 'grossMargin': 毛利率
                                
                                如果数据获取失败，返回None。
    
    示例：
        >>> # 获取招商银行2022年全年数据
        >>> df = get_stock_data('sh.600036', '2022-01-01', '2022-12-31')
        >>> print(f"获取到{len(df)}条交易记录")
        >>> print(df[['date', 'close', 'volume']].head())
        
        >>> # 获取平安银行最近一年数据（默认），并尝试包含额外基本面数据
        >>> df = get_stock_data('sz.000001', add_basic_data=True)
        >>> print(f"数据列: {df.columns.tolist()}")
        >>> # 检查是否包含基本面数据列
        >>> has_fundamentals = any(col in df.columns for col in ['roeTTM', 'netMargin'])
        >>> print(f"是否包含基本面数据: {has_fundamentals}")
    
    异常处理：
        - 如果登录BaoStock失败，将重试指定次数
        - 如果查询数据失败，将重试指定次数
        - 如果经过所有重试仍未获取到数据，返回None
        - 函数内部会处理所有异常，不会向上抛出异常
    
    注意：
        1. 函数会自动转换日期格式和数值类型，并处理缺失值和异常值
        2. 对于异常值（如价格跳跃），会使用中值滤波方法进行替换
        3. 返回的数据已排序（按日期升序）并重置索引
        4. 所有数据使用后复权价格，适合用于历史分析
        5. 至少需要获取30条记录才能计算大部分技术指标，建议查询至少60个交易日
        6. BaoStock API需要网络连接，如在无网络环境下使用，请考虑本地缓存数据
    """
    today = datetime.now()
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Ensure date format is correct (YYYY-MM-DD)
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
        
    print(f"Retrieving data for {stock_code} from {start_date} to {end_date}")
    
    for attempt in range(retry_count):
        try:
            # Login to baostock
            login_result = bs.login()
            if login_result.error_code != '0':
                print(f"Login failed: {login_result.error_msg}")
                time.sleep(1)
                continue
            
            # Get historical A-share K-line data
            fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
            
            # If additional fundamental data is needed
            if add_basic_data:
                # Try to add more financial indicators
                extended_fields = fields + ",roeTTM,roa,netMargin,grossMargin"
                try:
                    rs_test = bs.query_history_k_data_plus(stock_code, extended_fields, 
                                                        start_date=start_date, 
                                                        end_date=start_date,
                                                        frequency="d", 
                                                        adjustflag="3")
                    if rs_test.error_code == '0':
                        fields = extended_fields
                        print("Successfully added extended financial indicators")
                except:
                    print("Extended financial indicators are not available, using basic fields")
            
            rs = bs.query_history_k_data_plus(stock_code,
                                            fields,
                                            start_date=start_date,
                                            end_date=end_date,
                                            frequency="d",
                                            adjustflag="3")  # 3 indicates adjustment
            
            if rs.error_code != '0':
                print(f"Query failed: {rs.error_msg}, attempt {attempt+1}/{retry_count}")
                bs.logout()
                time.sleep(2)
                continue
            
            # Generate DataFrame
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            # Check if data is retrieved
            if not data_list:
                print(f"No data retrieved, attempt {attempt+1}/{retry_count}")
                bs.logout()
                time.sleep(2)
                continue
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # Logout of the system
            bs.logout()
            
            # Data cleaning and preprocessing
            if df.empty:
                print(f"No data retrieved for {stock_code}")
                continue
            
            print("Successfully retrieved", len(df), "records")
            print("Data columns:", df.columns.tolist())
            
            # Standardize date format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Convert numeric columns to numeric type
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                             'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            
            # Try to convert extended financial indicators
            if 'roeTTM' in df.columns:
                numeric_columns.extend(['roeTTM', 'roa', 'netMargin', 'grossMargin'])
                
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values: fill with previous values to maintain time series continuity
            df = df.ffill() 
            
            # Sort and reset index
            if 'date' in df.columns:
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
            # Detect and handle outliers (e.g., price jumps)
            if len(df) > 3:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        # Calculate rolling median
                        med = df[col].rolling(5, min_periods=1).median()
                        # Calculate rolling median absolute deviation
                        mad = np.abs(df[col] - med).rolling(5, min_periods=1).median()
                        # If deviation from median is too large (e.g., >10*MAD), replace with median
                        df.loc[np.abs(df[col] - med) > 10 * mad, col] = med
            
            # Ensure at least 30 records for indicator calculation
            if len(df) < 30:
                print(f"Warning: only {len(df)} records retrieved, less than 30")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving data: {e}")
            traceback.print_exc()
    
    print(f"Failed to retrieve data for {stock_code} after {retry_count} attempts")
    return None


def get_index_data(index_code, days=365):
    """
    Get index historical data
    
    Parameters:
        index_code (str): Index code, e.g., sh.000001 (Shanghai Composite Index)
        days (int): Number of days of historical data to retrieve, default 365
    
    Returns:
        pandas.DataFrame: DataFrame containing index data
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return get_stock_data(index_code, start_date, end_date)


def add_index_features(df, index_df, logger=None):
    """
    Add index-related features to stock data
    
    Parameters:
        df (pandas.DataFrame): Stock data DataFrame
        index_df (pandas.DataFrame): Index data DataFrame
        logger (logging.Logger, optional): Logger for output messages
    
    Returns:
        pandas.DataFrame: DataFrame with added index features
    """
    try:
        if logger:
            logger.info(f"Original stock data shape: {df.shape}, Index data shape: {index_df.shape}")
        else:
            print(f"Original stock data shape: {df.shape}, Index data shape: {index_df.shape}")
            
        # Ensure both DataFrames have a date column
        if 'date' not in df.columns or 'date' not in index_df.columns:
            if logger:
                logger.error("Missing date column in one of the DataFrames")
            else:
                print("Missing date column in one of the DataFrames")
            return df
            
        # Convert dates to string for safer merging if they're not already
        if not isinstance(df['date'].iloc[0], str):
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        if not isinstance(index_df['date'].iloc[0], str):
            index_df['date'] = index_df['date'].dt.strftime('%Y-%m-%d')
            
        # Merge on date column
        result_df = df.merge(index_df[['date', 'close', 'volume', 'pctChg']], 
                            on='date', 
                            how='left',
                            suffixes=('', '_index'))
        
        if logger:
            logger.info(f"After merge: data shape {result_df.shape}")
        
        # Rename index features
        result_df = result_df.rename(columns={
            'close_index': 'index_close',
            'volume_index': 'index_volume',
            'pctChg_index': 'index_pctChg'
        })
        
        # Fill missing values from the merge
        for col in ['index_close', 'index_volume', 'index_pctChg']:
            if col in result_df.columns:
                missing_before = result_df[col].isna().sum()
                result_df[col] = result_df[col].ffill().bfill()  # Forward then backward fill
                missing_after = result_df[col].isna().sum()
                
                if logger and missing_before > 0:
                    logger.info(f"Filled {missing_before - missing_after} missing values in {col}")
        
        # Count data points before rolling calculations
        valid_data_before = result_df.dropna().shape[0]
        if logger:
            logger.info(f"Valid data points before feature calculations: {valid_data_before}")
                
        # Calculate correlation features between stock and index
        # Beta calculation (using 20-day rolling window)
        stock_returns = result_df['close'].pct_change()
        index_returns = result_df['index_close'].pct_change()
        result_df['beta_20d'] = stock_returns.rolling(20).cov(index_returns) / index_returns.rolling(20).var()
        
        # Correlation calculation
        result_df['corr_with_index_20d'] = stock_returns.rolling(20).corr(index_returns)
        
        # Relative strength (stock vs index)
        result_df['rel_strength_5d'] = (result_df['close'].pct_change(5) - 
                                     result_df['index_close'].pct_change(5)) * 100
        result_df['rel_strength_10d'] = (result_df['close'].pct_change(10) - 
                                      result_df['index_close'].pct_change(10)) * 100
        result_df['rel_strength_20d'] = (result_df['close'].pct_change(20) - 
                                      result_df['index_close'].pct_change(20)) * 100
        
        # Index technical indicators
        # 1. Index price moving averages
        result_df['index_ma5'] = result_df['index_close'].rolling(5).mean()
        result_df['index_ma10'] = result_df['index_close'].rolling(10).mean()
        result_df['index_ma20'] = result_df['index_close'].rolling(20).mean()
        
        # 2. Index price change
        result_df['index_change'] = result_df['index_close'].pct_change()
        
        # 3. Index 5-day, 10-day, 20-day price change
        result_df['index_change_5d'] = result_df['index_close'].pct_change(5)
        result_df['index_change_10d'] = result_df['index_close'].pct_change(10)
        result_df['index_change_20d'] = result_df['index_close'].pct_change(20)
        
        # 4. Index RSI
        delta = result_df['index_close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result_df['index_rsi'] = 100 - (100 / (1 + rs))
        
        # 5. Index volatility
        result_df['index_volatility_10d'] = result_df['index_change'].rolling(10).std() * 100
        
        # 6. Market trend indicators (1: up, 0: down)
        result_df['market_trend_5d'] = (result_df['index_close'] > result_df['index_close'].shift(5)).astype(int)
        result_df['market_trend_10d'] = (result_df['index_close'] > result_df['index_close'].shift(10)).astype(int)
        
        # Handle NaN values due to rolling calculations
        # Count NaN values
        na_count = result_df.isna().sum().max()
        valid_data_after = result_df.dropna().shape[0]
        
        if logger:
            logger.info(f"Maximum NaN values in any column after feature calculations: {na_count}")
            logger.info(f"Valid data points after feature calculations: {valid_data_after}")
            logger.info(f"Data reduction due to feature calculations: {valid_data_before - valid_data_after} rows")
            logger.info(f"Percentage of data retained: {valid_data_after/len(result_df)*100:.2f}%")
        
        # Option 1: Fill NaNs with zeros - can be dangerous for some features
        # result_df = result_df.fillna(0)
        
        # Option 2: Filter out rows with NaNs - most conservative approach
        # result_df = result_df.dropna()
        
        # Option 3: Forward fill (copy previous values) - good for time series
        # This is generally safer than filling with zeros
        # result_df = result_df.fillna(method='ffill')
        
        # We'll keep NaN values as is, to let downstream functions handle them
        # This maintains data integrity but requires proper handling later
        
        return result_df
        
    except Exception as e:
        error_msg = f"Error adding index features: {e}"
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return df


def add_sector_features(df, sector_code):
    """
    Add sector-related features
    
    Parameters:
        df (pandas.DataFrame): Original data
        sector_code (str): Sector code, e.g., sh.000016 (Shanghai 50)
    
    Returns:
        pandas.DataFrame: DataFrame with added sector features
    """
    try:
        # Copy data to avoid modifying original data
        result_df = df.copy()
        
        print(f"Retrieving sector data: {sector_code}")
        
        # Get sector index data
        first_date = df['date'].min()
        last_date = df['date'].max()
        
        sector_df = get_stock_data(sector_code, first_date, last_date)
        
        if sector_df is None or len(sector_df) == 0:
            print(f"Failed to retrieve {sector_code} data")
            return df
            
        # Ensure both DataFrames have a date column
        if 'date' not in sector_df.columns:
            print("sector_df missing date column")
            return df
            
        # Merge on date column
        result_df = result_df.merge(sector_df[['date', 'close', 'volume', 'pctChg']], 
                           on='date', 
                           how='left',
                           suffixes=('', '_sector'))
        
        # Rename sector features
        result_df = result_df.rename(columns={
            'close_sector': 'sector_close',
            'volume_sector': 'sector_volume',
            'pctChg_sector': 'sector_pctChg'
        })
        
        # Fill missing values
        for col in ['sector_close', 'sector_volume', 'sector_pctChg']:
            if col in result_df.columns:
                result_df[col] = result_df[col].ffill()
                
        # Calculate some technical indicators for the sector
        # 1. Stock's relative strength to the sector
        # Compare the stock's price change to the sector's price change over the past N days
        for days in [5, 10, 20]:
            try:
                stock_change = result_df['close'].pct_change(days)
                sector_change = result_df['sector_close'].pct_change(days)
                result_df[f'rel_strength_{days}d'] = stock_change - sector_change
            except Exception as e:
                print(f"Error calculating relative strength indicator ({days}d): {e}")
                
        # 2. Sector 5-day, 10-day, 20-day price change
        result_df['sector_change_5d'] = result_df['sector_close'].pct_change(5)
        result_df['sector_change_10d'] = result_df['sector_close'].pct_change(10)
        result_df['sector_change_20d'] = result_df['sector_close'].pct_change(20)
        
        # 3. Sector's relative strength to the market
        for col in ['index_change_5d', 'index_change_10d', 'index_change_20d']:
            if col in result_df.columns:
                sector_col = col.replace('index', 'sector')
                if sector_col in result_df.columns:
                    result_df[f'sector_rel_{col[-4:]}'] = result_df[sector_col] - result_df[col]
        
        return result_df
        
    except Exception as e:
        print(f"Error adding sector features: {e}")
        traceback.print_exc()
        return df


def prepare_data(df, future_days=5, label_type='binary', test_size=0.2):
    """
    Prepare data and labels for machine learning models
    
    Parameters:
        df (pandas.DataFrame): Stock data, containing 'close' column
        future_days (int): Number of days to predict
        label_type (str): Label type, 'binary' (up/down) or 'regression' (price)
        test_size (float): Test set proportion
    
    Returns:
        tuple: (X, y, dates)
            - X: Feature data
            - y: Label data
            - dates: Corresponding dates
    """
    try:
        if df is None or len(df) < future_days + 10:
            print(f"Insufficient data, at least {future_days + 10} records required")
            return None, None, None
            
        # Ensure date column is datetime type
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Deep copy to avoid modifying original data
        data = df.copy()
        
        # Sort by date to ensure time series integrity
        if 'date' in data.columns:
            data = data.sort_values('date')
            
        # Ensure necessary columns exist
        if 'close' not in data.columns:
            print("Error: 'close' column missing from data")
            return None, None, None
            
        # Create labels - future price change
        if label_type == 'binary':
            # Create binary labels: 1 for up, 0 for down or flat
            data['future_price'] = data['close'].shift(-future_days)
            data['label'] = (data['future_price'] > data['close']).astype(int)
        else:
            # Regression labels: future price
            data['label'] = data['close'].shift(-future_days)
            
        # Remove NaN values
        data = data.dropna()
        
        if len(data) == 0:
            print("Processed data is empty")
            return None, None, None
            
        # Extract features and labels
        if 'date' in data.columns:
            dates = data['date'].values
        else:
            dates = np.arange(len(data))
            
        # Use only basic features as input
        X = data[['close']].values
        y = data['label'].values
        
        return X, y, dates
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        traceback.print_exc()
        return None, None, None
