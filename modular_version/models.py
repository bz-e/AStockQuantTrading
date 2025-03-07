#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型模块 - 负责模型训练、评估和预测

该模块实现了多种机器学习模型的训练、评估和预测功能，
包括随机森林、XGBoost、LightGBM以及集成模型。
"""

import numpy as np
import pandas as pd
import time
import traceback
import matplotlib.pyplot as plt
from contextlib import contextmanager
import signal
import gc
import psutil
import warnings

# 基础机器学习模型和工具
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import parallel_backend
from sklearn.dummy import DummyClassifier

# 专用模型库导入
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("XGBoost未安装，将不使用XGBoost模型")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    has_lightgbm = True
except ImportError:
    has_lightgbm = False
    print("LightGBM未安装，将不使用LightGBM模型")

# 自定义超时处理器
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("运行时间超出限制")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

# 内存监控和优化函数
def monitor_memory_usage():
    """返回当前进程的内存使用情况（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def optimize_memory(X=None, y=None, force_gc=False):
    """优化内存使用，返回节省的内存量（MB）"""
    initial_memory = monitor_memory_usage()
    
    if force_gc:
        gc.collect()
        
    final_memory = monitor_memory_usage()
    return initial_memory - final_memory

warnings.filterwarnings('ignore')

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

# 实现超时功能的上下文管理器
@contextmanager
def time_limit(seconds):
    """
    创建一个超时上下文管理器，用于限制函数执行时间
    
    参数：
        seconds (int): 超时时间（秒）
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"执行超时，超过 {seconds} 秒")
        
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # 恢复闹钟

def get_adaptive_model_params(X, model_type):
    """
    根据数据集特征自动调整模型参数，提高计算效率
    
    参数：
        X (pd.DataFrame/np.ndarray): 特征数据
        model_type (str): 模型类型，可选值: 'rf', 'xgb', 'lgb', 'knn', 'svm'
    
    返回：
        dict: 适应性参数字典
    """
    n_samples = len(X)
    n_features = X.shape[1]
    params = {}
    
    # 根据数据集大小动态调整参数
    if model_type == 'rf':
        # 随机森林参数
        params['n_estimators'] = min(300, max(50, int(n_samples / 5)))
        params['max_depth'] = min(20, max(5, int(np.log2(n_features) * 2)))
        params['min_samples_split'] = max(2, int(np.sqrt(n_samples) / 10))
        
        # 大数据集使用更少的树，小数据集使用更多的树
        if n_samples > 10000:
            params['n_estimators'] = min(params['n_estimators'], 150)
            params['bootstrap'] = True
            params['oob_score'] = True  # 使用袋外样本评估，避免额外的交叉验证
        
    elif model_type == 'xgb':
        # XGBoost参数
        params['n_estimators'] = min(300, max(50, int(n_samples / 5)))
        params['max_depth'] = min(8, max(3, int(np.log2(n_features))))
        
        # 对于大数据集，使用直方图近似方法
        if n_samples > 5000:
            params['tree_method'] = 'hist'
        else:
            params['tree_method'] = 'exact'
            
        # 更大的数据集使用更小的学习率
        if n_samples > 10000:
            params['learning_rate'] = 0.05
        else:
            params['learning_rate'] = 0.1
            
    elif model_type == 'lgb':
        # LightGBM参数
        params['n_estimators'] = min(300, max(50, int(n_samples / 5)))
        params['max_depth'] = min(8, max(3, int(np.log2(n_features))))
        
        # 特征和样本抽样比例
        params['feature_fraction'] = 0.8
        params['bagging_fraction'] = 0.8
        params['bagging_freq'] = 5
        
        # 大数据集使用更小的学习率和更少的叶子
        if n_samples > 10000:
            params['learning_rate'] = 0.05
            params['num_leaves'] = min(64, 2**params['max_depth'])
        else:
            params['learning_rate'] = 0.1
            params['num_leaves'] = min(127, 2**(params['max_depth']+1))
            
    elif model_type == 'knn':
        # KNN参数
        params['n_neighbors'] = min(20, max(3, int(np.sqrt(n_samples))))
        
        # 大数据集使用基于树的快速KNN
        if n_samples > 5000:
            params['algorithm'] = 'kd_tree'
        else:
            params['algorithm'] = 'auto'
            
    elif model_type == 'svm':
        # SVM参数
        # 大数据集使用线性核，小数据集使用RBF核
        if n_samples > 5000:
            params['kernel'] = 'linear'
            params['C'] = 1.0
        else:
            params['kernel'] = 'rbf'
            params['C'] = 10.0
            params['gamma'] = 'scale'
    
    return params

def check_progress_callback(iteration):
    """
    进度回调函数，用于监控超长时间训练的模型
    """
    global last_callback_time
    current_time = time.time()
    
    # 初始化全局变量
    if 'last_callback_time' not in globals():
        global last_callback_time
        last_callback_time = current_time
        return
    
    # 每10次迭代或者至少3秒打印一次进度
    if iteration % 10 == 0 or (current_time - last_callback_time) > 3:
        print(f"训练进度：迭代 {iteration} 完成")
        last_callback_time = current_time

def train_ml_model(X, y, model_type='rf', feature_names=None, use_cv=True, use_feature_selection=True, use_hyperopt=False, memory_optimization=True, timeout=300, **kwargs):
    """
    训练并评估机器学习模型用于股票涨跌预测
    
    该函数实现了多种机器学习模型的训练、评估和预测功能，
    包括随机森林、XGBoost、LightGBM以及集成模型。
    
    参数：
        X (numpy.ndarray): 特征矩阵，形状为(n_samples, n_features)，其中n_samples是样本数量，
                          n_features是特征数量。样本应按时间顺序排列（从最早到最近）。
        
        y (numpy.ndarray): 标签向量，形状为(n_samples,)，值应为二分类标签（0或1），
                          其中0表示下跌，1表示上涨。长度必须与X的样本数相同。
        
        model_type (str): 模型类型，可选值：
                        - 'rf': 随机森林分类器（默认），稳定且适用于大多数情况
                        - 'xgb': XGBoost分类器，通常在有足够数据时性能更佳
                        - 'lgbm': LightGBM分类器，计算效率高，适合较大数据集
                        如指定的模型依赖不可用，将自动回退到随机森林。
        
        feature_names (list, optional): 特征名列表，长度应等于X的特征数量。用于特征重要性
                                      分析和结果解释。如未提供，将使用'feature_0'、
                                      'feature_1'等默认名称。
        
        use_cv (bool): 是否使用时间序列交叉验证评估模型，默认为True。启用后将使用
                      TimeSeriesSplit进行5折交叉验证，这对时间序列数据更合适。
        
        use_feature_selection (bool): 是否使用特征选择来减少特征数量，默认为True。
                                    目前此参数在函数内部未实际使用，保留作未来扩展。
        
        use_hyperopt (bool): 是否使用超参数优化来提升模型性能，默认为False。启用后会
                            显著增加计算时间，但可能提高模型准确率。对于RF使用GridSearchCV，
                            对于XGBoost和LightGBM使用RandomizedSearchCV。
    
    返回：
        tuple: 包含以下5个元素的元组：
            - model (object): 训练好的模型实例，可用于后续预测
            - predictions (numpy.ndarray): 对所有输入数据的预测结果数组
            - accuracy (float): 模型在测试集上的预测准确率（百分比，0-100）
            - confusion_matrix (numpy.ndarray): 混淆矩阵，形状为(2, 2)，用于评估模型性能
            - feature_importance (list): 特征重要性列表，每个元素为(特征名, 重要性)元组，
                                      按重要性降序排列
    
    示例：
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 特征数据
        >>> y_train = np.array([0, 0, 1, 1])  # 标签数据（0=下跌，1=上涨）
        >>> feature_names = ['price_momentum', 'volume_change']
        >>> model, preds, acc, cm, importance = train_ml_model(
        ...     X_train, y_train, model_type='rf', feature_names=feature_names, 
        ...     use_cv=True, use_hyperopt=False
        ... )
        >>> print(f"模型准确率: {acc:.2f}%")
        >>> print("前3个重要特征:", importance[:3])
    
    注意：
        1. 对于小数据集(<100样本)，建议使用随机森林('rf')以避免过拟合
        2. 启用超参数优化(use_hyperopt=True)会显著增加训练时间，但可能提高模型性能
        3. 函数会计算基准模型(DummyClassifier)准确率作为参考，帮助判断模型是否有实际预测能力
        4. 如果对指定模型类型的依赖库未安装，会自动回退到随机森林模型
        5. 时间序列数据不应随机打乱，本函数使用shuffle=False进行训练测试集分割
        6. 返回的predictions是对完整数据集的预测，而不仅是测试集
    """
    try:
        # 确保X和y是numpy数组并强制转换类型
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        # 数据预处理：处理NaN和无穷值
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("警告: 特征矩阵包含NaN或无穷值，正在处理...")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.nanmax(X), neginf=np.nanmin(X))
        
        # 标签验证：确保标签为二分类（0和1）
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            print(f"错误: 标签应为二分类 (当前类别: {unique_labels})")
            return None, None, 0, None, []
        
        if np.any(np.isnan(y)):
            print("错误: 标签包含NaN值")
            return None, None, 0, None, []
            
        # 检查数据维度
        if len(X) != len(y):
            print(f"错误: 特征矩阵与标签向量长度不匹配 (X: {len(X)}, y: {len(y)})")
            return None, None, 0, None, []
            
        if len(X) == 0:
            print("错误: 特征矩阵为空")
            return None, None, 0, None, []
            
        # 检查特征名是否提供
        if feature_names is None or len(feature_names) == 0:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 检查特征名长度是否与特征数匹配
        if len(feature_names) != X.shape[1]:
            print(f"警告: 特征名数量({len(feature_names)})与特征数({X.shape[1]})不匹配")
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 特征选择（如果启用）
        if use_feature_selection:
            print("执行特征选择...")
            
            # 1. 移除低方差特征
            var_threshold = 0.01  # 方差阈值，可根据需要调整
            var_selector = VarianceThreshold(threshold=var_threshold)
            
            try:
                X_filtered = var_selector.fit_transform(X)
                var_mask = var_selector.get_support()
                
                # 保存被选中的特征名
                selected_features = [feature_names[i] for i in range(len(feature_names)) if var_mask[i]]
                
                print(f"低方差特征过滤: 从 {X.shape[1]} 个特征减少到 {X_filtered.shape[1]} 个")
                
                # 更新数据和特征名
                X = X_filtered
                feature_names = selected_features
                
                # 2. 基于F值选择TopK特征
                if X.shape[1] > 10:  # 如果特征数量还是很多，继续筛选
                    k = min(50, X.shape[1])  # 最多选择50个特征
                    f_selector = SelectKBest(f_classif, k=k)
                    X_filtered = f_selector.fit_transform(X, y)
                    f_mask = f_selector.get_support()
                    
                    # 保存被选中的特征名
                    selected_features = [feature_names[i] for i in range(len(feature_names)) if f_mask[i]]
                    
                    print(f"Top-K特征选择: 从 {X.shape[1]} 个特征减少到 {X_filtered.shape[1]} 个")
                    
                    # 更新数据和特征名
                    X = X_filtered
                    feature_names = selected_features
                
            except Exception as e:
                print(f"特征选择出错: {e}")
                print("跳过特征选择步骤")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 确保数据集大小足够
        if len(X_train) < 10 or len(X_test) < 5:
            print(f"警告: 训练集或测试集样本数过少 (训练集: {len(X_train)}, 测试集: {len(X_test)})")
        
        print(f"使用模型: {model_type}")
        
        # 基准模型 (总是预测多数类)
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_acc = accuracy_score(y_test, baseline_pred) * 100
        print(f"基准模型准确率: {baseline_acc:.2f}%")
        
        # 选择并训练模型
        model = None
        
        # 内存优化
        if memory_optimization:
            initial_memory = monitor_memory_usage()
            print(f"初始内存使用: {initial_memory:.2f} MB")
            current_memory, memory_saved = optimize_memory(X, y)
            print(f"内存优化节省: {memory_saved:.2f} MB, 当前内存使用: {current_memory:.2f} MB")
        
        # 获取适应性模型参数
        adaptive_params = get_adaptive_model_params(X, model_type)
        print(f"使用适应性参数: {adaptive_params}")
            
        if model_type == 'rf':
            # 随机森林
            if use_hyperopt:
                print("对随机森林进行超参数优化...")
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
                
                # 使用GridSearchCV进行超参数优化
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=TimeSeriesSplit(n_splits=3),  # 减少交叉验证折数以加快速度
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42
                )
                
                try:
                    with time_limit(timeout // 2):  # 超参数优化使用一半的超时时间
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        print(f"最佳参数: {grid_search.best_params_}")
                except TimeoutError:
                    print(f"超参数优化超时，使用默认参数...")
                    # 超时后使用默认配置
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_split=5,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
            else:
                # 优化的随机森林
                model = RandomForestClassifier(
                    **adaptive_params,
                    n_jobs=-1,  # 并行训练
                    random_state=42,
                    verbose=0
                )
                model.fit(X_train, y_train)
                
        elif model_type == 'xgb' and has_xgboost:
            # XGBoost
            try:
                if use_hyperopt:
                    print("对XGBoost进行超参数优化...")
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
                    
                    # 使用更高效的RandomizedSearchCV进行超参数优化
                    random_search = RandomizedSearchCV(
                        xgb.XGBClassifier(
                            objective='binary:logistic', 
                            random_state=42,
                            n_jobs=-1,  # 并行训练
                            tree_method='hist'  # 更快的直方图算法
                        ),
                        param_distributions=param_grid,
                        n_iter=10,  # 减少迭代次数
                        cv=TimeSeriesSplit(n_splits=3),  # 减少交叉验证折数
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    try:
                        with time_limit(timeout // 2):  # 超参数优化使用一半的超时时间
                            random_search.fit(X_train, y_train)
                            model = random_search.best_estimator_
                            print(f"最佳参数: {random_search.best_params_}")
                    except TimeoutError:
                        print(f"超参数优化超时，使用默认参数...")
                        # 超时后使用默认配置
                        model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            n_estimators=100,
                            max_depth=5,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                            tree_method='hist',
                            verbosity=0
                        )
                        model.fit(X_train, y_train)
                else:
                    # 分出一部分训练数据用于早停验证
                    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                    
                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        **adaptive_params,
                        random_state=42,
                        n_jobs=-1,  # 并行训练
                        tree_method='hist',  # 更快的直方图算法
                        verbosity=0  # 减少输出
                    )
                    
                    # 使用验证集实现早停，避免过拟合同时减少训练时间
                    print("训练XGBoost(使用早停机制)...")
                    try:
                        with time_limit(timeout):
                            model.fit(
                                X_train_fit, y_train_fit,
                                eval_set=[(X_val, y_val)],
                                eval_metric='logloss',
                                early_stopping_rounds=20,  # 20轮没有提升则停止
                                verbose=False,
                                callbacks=[check_progress_callback]  # 添加进度回调函数
                            )
                            
                            # 记录最佳迭代次数
                            print(f"最佳迭代次数: {model.best_iteration}")
                            
                            # 使用最佳迭代次数在完整训练集上重新训练
                            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                                final_n_estimators = model.best_iteration
                                print(f"使用最佳迭代次数重新训练: {final_n_estimators}")
                                
                                # 限时训练最终模型
                                try:
                                    with time_limit(timeout // 2):  # 使用一半的超时时间
                                        model = xgb.XGBClassifier(
                                            objective='binary:logistic',
                                            **adaptive_params,
                                            n_estimators=final_n_estimators,
                                            random_state=42,
                                            n_jobs=-1,
                                            tree_method='hist',
                                            verbosity=0
                                        )
                                        model.fit(X_train, y_train)
                                except TimeoutError:
                                    print("最终XGBoost模型训练超时，使用之前的模型...")
                                    # 保持之前训练的模型
                    except TimeoutError:
                        print(f"XGBoost训练超时，使用快速配置...")
                        # 超时后使用更快的配置重试
                        model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            n_estimators=50,  # 使用更少的树
                            max_depth=3,  # 更浅的树
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                            tree_method='hist',
                            verbosity=0
                        )
                        model.fit(X_train, y_train)
            except Exception as e:
                print(f"XGBoost训练失败: {e}")
                print("回退到随机森林模型")
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
                
        elif model_type == 'lgb' and has_lightgbm:
            # LightGBM
            try:
                if use_hyperopt:
                    print("对LightGBM进行超参数优化...")
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [15, 31, 63],
                        'feature_fraction': [0.8, 0.9, 1.0],
                        'bagging_fraction': [0.8, 0.9, 1.0],
                    }
                    
                    # 随机搜索法更快
                    random_search = RandomizedSearchCV(
                        lgb.LGBMClassifier(
                            objective='binary',
                            verbosity=-1,
                            random_state=42,
                            n_jobs=-1
                        ),
                        param_grid,
                        n_iter=10,  # 只尝试10种组合
                        cv=TimeSeriesSplit(n_splits=3),  # 减少交叉验证折数以加快速度
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    try:
                        with time_limit(timeout // 2):  # 超参数优化使用一半的超时时间
                            random_search.fit(X_train, y_train)
                            model = random_search.best_estimator_
                            print(f"最佳参数: {random_search.best_params_}")
                    except TimeoutError:
                        print(f"LightGBM超参数优化超时，使用默认参数...")
                        # 超时后使用默认配置
                        model = lgb.LGBMClassifier(
                            objective='binary',
                            n_estimators=100,
                            max_depth=5,
                            num_leaves=31,
                            learning_rate=0.1,
                            feature_fraction=0.8,
                            bagging_fraction=0.8,
                            bagging_freq=5,
                            verbosity=-1,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                else:
                    # 分出一部分训练数据用于早停验证
                    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                    
                    model = lgb.LGBMClassifier(
                        objective='binary',
                        **adaptive_params,
                        random_state=42,
                        n_jobs=-1,  # 并行训练
                        verbosity=-1,  # 减少输出
                        feature_fraction=0.8,  # 每次迭代时随机选择80%的特征，提高速度和防止过拟合
                        bagging_fraction=0.8,  # 类似随机森林的bagging方法
                        bagging_freq=5  # 每5次迭代执行bagging
                    )
                    
                    # 使用验证集实现早停，避免过拟合同时减少训练时间
                    print("训练LightGBM(使用早停机制)...")
                    try:
                        with time_limit(timeout):
                            model.fit(
                                X_train_fit, y_train_fit,
                                eval_set=[(X_val, y_val)],
                                eval_metric='binary_logloss',
                                early_stopping_rounds=20,  # 20轮没有提升则停止
                                verbose=False,
                                callbacks=[check_progress_callback]  # 添加进度回调函数
                            )
                            
                            # 记录最佳迭代次数
                            print(f"最佳迭代次数: {model.best_iteration_}")
                            
                            # 使用最佳迭代次数在完整训练集上重新训练
                            if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                                final_n_estimators = model.best_iteration_
                                print(f"使用最佳迭代次数重新训练: {final_n_estimators}")
                                
                                # 限时训练最终模型
                                try:
                                    with time_limit(timeout // 2):  # 使用一半的超时时间
                                        model = lgb.LGBMClassifier(
                                            objective='binary',
                                            **adaptive_params,
                                            n_estimators=final_n_estimators,
                                            random_state=42,
                                            n_jobs=-1,
                                            verbosity=-1,
                                            feature_fraction=0.8,
                                            bagging_fraction=0.8,
                                            bagging_freq=5
                                        )
                                        model.fit(X_train, y_train)
                                except TimeoutError:
                                    print("最终LightGBM模型训练超时，使用之前的模型...")
                                    # 保持之前训练的模型
                    except TimeoutError:
                        print(f"LightGBM训练超时，使用快速配置...")
                        # 超时后使用更快的配置重试
                        model = lgb.LGBMClassifier(
                            objective='binary',
                            n_estimators=50,  # 使用更少的树
                            max_depth=3,  # 更浅的树
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=-1,
                            feature_fraction=0.8,
                            bagging_fraction=0.8,
                            bagging_freq=5
                        )
                        model.fit(X_train, y_train)
            except Exception as e:
                print(f"LightGBM训练失败: {e}")
                print("回退到随机森林模型")
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
        else:
            # 默认使用随机森林
            print(f"未识别的模型类型或依赖库未安装: {model_type}，使用随机森林")
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            
        # 使用交叉验证评估模型
        if use_cv:
            print("使用时间序列交叉验证评估模型...")
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            print(f"交叉验证准确率: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
            
        # 在测试集上评估模型
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        cm = confusion_matrix(y_test, predictions)
        
        # 计算ROC-AUC分数（如果模型支持predict_proba）
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
                print(f"ROC-AUC分数: {roc_auc:.4f}")
            except Exception as e:
                print(f"计算ROC-AUC时出错: {e}")
                
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, predictions))
        
        # 获取特征重要性（如果模型支持）
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = [(feature_names[i], importance) for i, importance in enumerate(importances)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
        # 对测试数据进行完整预测，用于进一步分析
        all_predictions = model.predict(X)
        
        # 创建结果字典
        results = {
            'model': model,
            'predictions': all_predictions,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
        
        # 评估收益率（如果提供了收益率数据）
        if 'returns' in kwargs and kwargs['returns'] is not None:
            returns_data = kwargs['returns']
            if len(returns_data) >= len(y):
                # 确保返回数据与标签长度一致（使用最近的数据）
                returns_data = returns_data[-len(y):]
                cum_returns, annual_return, sharpe, max_dd, win_rate = evaluate_returns(y_test, predictions, returns_data[-len(y_test):])
                
                # 将收益评估结果添加到返回结果中
                results['returns_evaluation'] = {
                    'cumulative_returns': cum_returns,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'win_rate': win_rate
                }
            else:
                print(f"警告: 提供的收益率数据长度({len(returns_data)})小于标签数据长度({len(y)})")
                
        return results
        
    except Exception as e:
        print(f"训练模型时出错: {e}")
        traceback.print_exc()
        return None, None, 0, None, []


def train_ensemble_model(X, y, ensemble_type='voting', base_models=None, feature_names=None, test_size=0.3, 
                      auto_model_selection=True, optimize=True, 
                      random_state=42, verbose=1, timeout=600, **kwargs):
    """
    训练集成模型

    参数:
        X (array-like): 特征数据
        y (array-like): 目标变量
        ensemble_type (str): 集成类型，可选 'voting' 或 'stacking'
        base_models (dict): 基础模型字典，格式为 {model_name: model_instance}
        feature_names (list): 特征名称列表
        test_size (float): 测试集比例
        auto_model_selection (bool): 是否启用智能模型选择
        optimize (bool): 是否优化集成模型超参数
        random_state (int): 随机种子
        verbose (int): 详细程度
        timeout (int): 训练超时时间(秒)
        **kwargs: 其他参数

    返回:
        dict: 包含模型、预测结果和评估指标的字典
    """
    if verbose:
        print("训练" + ensemble_type + "集成模型...")
    
    start_time = time.time()
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 动态调整n_estimators参数，根据训练数据规模
    n_estimators = min(300, max(50, int(len(X_train) / 5)))
    if verbose:
        print(f"根据训练集大小({len(X_train)})，动态设置n_estimators={n_estimators}")
    
    # 如果没有提供基础模型，使用默认模型
    if base_models is None:
        base_models = {
            'rf': RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
            'xgb': XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=random_state),
            'lgb': LGBMClassifier(n_estimators=n_estimators, random_state=random_state),
            'knn': KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))) if len(X_train) > 100 else 5),
            'svm': SVC(probability=True, random_state=random_state)
        }
    
    # 智能模型选择 - 根据数据集特征自动调整模型组合
    if auto_model_selection:
        if verbose:
            print("启用自动模型选择...")
        
        # 保存原始基础模型组合
        original_base_models = base_models.copy()
        models_to_remove = []
        models_to_modify = {}
        
        # 小数据集(<1000样本)：移除LightGBM避免过拟合，调整SVM为线性核
        if len(X_train) < 1000:
            if 'lgb' in base_models:
                models_to_remove.append('lgb')
                if verbose:
                    print("样本量较小 (<1000)，移除LightGBM以避免过拟合")
            
            # 小数据集上使用线性SVM
            if 'svm' in base_models:
                models_to_modify['svm'] = SVC(kernel='linear', probability=True, random_state=random_state)
        
        # 高维数据(>50特征)：调整模型
        if X.shape[1] > 50:
            # 高维数据上SVM表现不佳且计算成本高，移除或调整
            if 'svm' in base_models and X.shape[1] > 100:
                models_to_remove.append('svm')
                if verbose:
                    print("特征维度过高 (>100)，移除SVM以提高效率")
            elif 'svm' in base_models:
                # 特征数量适中(50-100)时使用RBF核
                models_to_modify['svm'] = SVC(kernel='rbf', probability=True, random_state=random_state)
        
        # 大数据集(>10000样本)：移除KNN提高训练效率
        if len(X_train) > 10000 and 'knn' in base_models:
            models_to_remove.append('knn')
            if verbose:
                print("样本量较大 (>10000)，移除KNN以提高训练效率")
                
        # 现在应用我们的更改
        for model_name in models_to_remove:
            if model_name in base_models:
                del base_models[model_name]
                
        for model_name, new_model in models_to_modify.items():
            base_models[model_name] = new_model
        
        # 确保至少有2个模型用于集成
        if len(base_models) < 2:
            if verbose:
                print("自动模型选择后模型数量不足，添加额外模型...")
            
            # 根据数据集大小选择适合的补充模型
            if len(X_train) <= 1000:
                # 小数据集优先使用简单模型
                if 'rf' not in base_models:
                    base_models['rf'] = original_base_models.get('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
                if len(base_models) < 2 and 'xgb' not in base_models:
                    base_models['xgb'] = original_base_models.get('xgb', XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=random_state))
                if len(base_models) < 2 and 'knn' not in base_models:
                    base_models['knn'] = original_base_models.get('knn', KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))) if len(X_train) > 100 else 5))
            else:
                # 大数据集优先使用复杂模型
                if 'xgb' not in base_models:
                    base_models['xgb'] = original_base_models.get('xgb', XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=random_state))
                if len(base_models) < 2 and 'rf' not in base_models:
                    base_models['rf'] = original_base_models.get('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
                if len(base_models) < 2 and 'lgb' not in base_models:
                    base_models['lgb'] = original_base_models.get('lgb', LGBMClassifier(n_estimators=n_estimators, random_state=random_state))
        
        if verbose:
            print(f"最终选择的基础模型: {list(base_models.keys())}")
    
    # 训练每个基础模型
    model_results = {}
    for name, base_model in list(base_models.items()):  # 使用list()创建副本进行遍历
        if verbose:
            print(f"训练基础模型 {name}...")
        start_model_time = time.time()
        
        try:
            # 为每个基础模型应用超时保护
            with time_limit(timeout // (len(base_models) + 1)):  # 平均分配时间给每个模型，预留额外时间给集成步骤
                base_model.fit(X_train, y_train)
            
            if name in ['xgb', 'lgb'] and hasattr(base_model, 'best_iteration_'):
                if verbose:
                    print(f"模型 {name} 最佳迭代次数: {base_model.best_iteration_}")
            
            # 使用训练好的模型进行预测
            y_pred = base_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if verbose:
                print(f"模型 {name} 验证集准确率: {accuracy:.4f}")
            
            # 评估预测表现
            model_results[name] = {
                'model': base_model,
                'accuracy': accuracy,
                'training_time': time.time() - start_model_time
            }
        except TimeoutException:
            if verbose:
                print(f"模型 {name} 训练超时，跳过该模型")
            # 从基础模型字典中移除超时的模型
            del base_models[name]
        except Exception as e:
            if verbose:
                print(f"训练模型 {name} 时出错: {e}")
                traceback.print_exc()
            # 从基础模型字典中移除出错的模型
            if name in base_models:
                del base_models[name]
    
    # 使用超参数优化
    model = None
    if optimize:
        t_start = time.time()
        try:
            with time_limit(timeout // 3):  # 为超参数优化分配较少的超时时间，避免卡住
                if ensemble_type == 'voting':
                    # 首先创建基础投票集成模型
                    estimators = [(name, model) for name, model in base_models.items()]
                    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                    
                    # 投票集成超参优化
                    parameters = {
                        'voting': ['soft', 'hard'],
                        'weights': [None, [1] * len(base_models), 
                                   list(range(1, len(base_models) + 1)), 
                                   list(range(len(base_models), 0, -1))]
                    }
                    if verbose:
                        print(f"开始超参数优化，优化参数: {parameters}")
                    grid = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    if verbose:
                        print(f"最佳投票集成参数: {grid.best_params_}")
                
                elif ensemble_type == 'stacking':
                    # 首先创建基础堆叠集成模型
                    estimators = [(name, model) for name, model in base_models.items()]
                    model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1),
                        cv=3,
                        n_jobs=-1,
                        passthrough=False
                    )
                    
                    # 堆叠集成超参优化，使用随机搜索节省时间
                    final_estimator_params = {
                        'final_estimator__C': [0.1, 1.0, 10.0],
                        'final_estimator__max_iter': [500, 1000],
                        'passthrough': [True, False]
                    }
                    if verbose:
                        print(f"开始超参数优化，优化参数: {final_estimator_params}")
                    random_search = RandomizedSearchCV(
                        model, final_estimator_params, 
                        n_iter=5,  # 限制迭代次数，防止过长时间
                        cv=3, random_state=42, n_jobs=-1
                    )
                    random_search.fit(X_train, y_train)
                    model = random_search.best_estimator_
                    if verbose:
                        print(f"最佳堆叠集成参数: {random_search.best_params_}")
                        
                if verbose:
                    print(f"集成模型超参数优化耗时: {time.time() - t_start:.2f}秒")
        except TimeoutException:
            if verbose:
                print(f"集成模型超参数优化超时 (>{timeout//3}秒)，将使用默认参数")
            # 在超时情况下，我们保留当前模型，不做进一步优化
            model = None
        except Exception as e:
            if verbose:
                print(f"集成模型超参数优化出错: {e}，将使用默认参数")
            model = None
            traceback.print_exc()
    
    # 如果超参数优化失败或未启用，使用默认设置
    if model is None:
        if verbose:
            print("创建默认集成模型配置...")
        # 创建集成模型
        if ensemble_type == 'voting':
            # 投票集成
            if verbose:
                print(f"创建投票集成模型，基础模型: {list(base_models.keys())}")
            estimators = [(name, model) for name, model in base_models.items()]
            model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        elif ensemble_type == 'stacking':
            # 堆叠集成
            if verbose:
                print(f"创建堆叠集成模型，基础模型: {list(base_models.keys())}")
            estimators = [(name, model) for name, model in base_models.items()]
            try:
                with time_limit(timeout // 2):  # 为堆叠模型训练指定超时时间
                    model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1),
                        cv=3,  # 减少交叉验证折数提高速度
                        n_jobs=-1,  # 并行处理提高训练速度
                        passthrough=False  # 不传递原始特征，减少计算量
                    )
                    if verbose:
                        print("开始训练堆叠集成模型...")
                    model.fit(X_train, y_train)
            except TimeoutException:
                if verbose:
                    print("堆叠模型创建超时，使用简化的投票集成模型...")
                # 先尝试创建投票集成作为备选
                try:
                    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                    if verbose:
                        print("使用投票集成模型作为备选...")
                    model.fit(X_train, y_train)
                except Exception as e:
                    if verbose:
                        print(f"投票集成模型失败: {e}，使用随机森林作为最终备选")
                    # 创建简化的随机森林模型作为后备
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
            except Exception as e:
                if verbose:
                    print(f"堆叠模型创建失败: {e}，使用投票集成模型作为备选")
                traceback.print_exc()
                # 先尝试创建投票集成作为备选
                try:
                    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                    if verbose:
                        print("使用投票集成模型作为备选...")
                    model.fit(X_train, y_train)
                except Exception as ve:
                    if verbose:
                        print(f"投票集成模型失败: {ve}，使用随机森林作为最终备选")
                    # 创建简化的随机森林模型作为后备
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
        else:
            if verbose:
                print(f"未知的集成类型: {ensemble_type}，使用投票集成")
            estimators = [(name, model) for name, model in base_models.items()]
            model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    # 训练模型
    if verbose:
        print("训练集成模型...")
    try:
        # 使用小数据集进行初步训练
        if len(X_train) > 1000:
            print("第1阶段: 使用小数据集进行初步训练...")
            try:
                with time_limit(timeout // len(base_models)):  # 平均分配超时时间给每个模型
                    if not hasattr(model, 'fit') or not callable(getattr(model, 'fit')):
                        raise ValueError(f"模型不支持fit方法，重新创建集成模型")
                    model.fit(X_train[:1000], y_train[:1000])
            except TimeoutException:
                if verbose:
                    print(f"使用小数据集训练超时，尝试使用投票集成...")
                try:
                    # 尝试使用投票集成
                    estimators = [(name, base) for name, base in base_models.items()]
                    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                    model.fit(X_train[:1000], y_train[:1000])
                except Exception as e:
                    if verbose:
                        print(f"投票集成训练失败: {e}，使用随机森林替代...")
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    model.fit(X_train[:1000], y_train[:1000])
            except Exception as e:
                if verbose:
                    print(f"小数据集训练失败: {e}，使用随机森林替代...")
                traceback.print_exc()
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train[:1000], y_train[:1000])
                
            # 释放小训练集内存
            if 'memory_optimization' in kwargs and kwargs['memory_optimization']:
                del X_train[:1000], y_train[:1000]
                optimize_memory(force_gc=True)
                    
            # 使用完整训练集进行微调
            if verbose:
                print("第2阶段: 使用完整训练集进行微调...")
            try:
                with time_limit(timeout // 2):  # 最终训练使用一半的超时时间
                    model.fit(X_train, y_train)
            except TimeoutException:
                if verbose:
                    print("完整数据集训练超时，使用之前的模型...")
                # 保持之前训练的模型
        else:
            # 直接使用完整训练集训练
            try:
                with time_limit(timeout // 2):  # 最终训练使用一半的超时时间
                    if not hasattr(model, 'fit') or not callable(getattr(model, 'fit')):
                        raise ValueError(f"模型不支持fit方法，重新创建模型")
                    model.fit(X_train, y_train)
            except TimeoutException:
                if verbose:
                    print("完整数据集训练超时，尝试使用投票集成...")
                try:
                    # 尝试使用投票集成
                    estimators = [(name, base) for name, base in base_models.items()]
                    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                    model.fit(X_train, y_train)
                except Exception as e:
                    if verbose:
                        print(f"投票集成训练失败: {e}，使用随机森林作为备选...")
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)
            except Exception as e:
                if verbose:
                    print(f"模型训练失败: {e}，使用随机森林作为备选...")
                traceback.print_exc()
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
        if verbose:
            print(f"最终使用的模型类型: {type(model).__name__}")
        
        # 检查是否为集成模型
        if isinstance(model, (VotingClassifier, StackingClassifier)):
            if verbose:
                print("成功创建并训练集成模型!")
            if hasattr(model, 'estimators_'):
                if verbose:
                    print(f"集成模型包含以下子模型: {[type(est).__name__ for est in model.estimators_]}")
            elif hasattr(model, 'named_estimators_'):
                if verbose:
                    print(f"集成模型包含以下子模型: {[name + ':' + type(est).__name__ for name, est in model.named_estimators_.items()]}")
        else:
            if verbose:
                print(f"警告: 最终模型不是集成模型，而是 {type(model).__name__}")
        
        # 进行预测
        predictions = model.predict(X)
        test_predictions = model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, test_predictions) * 100
        if verbose:
            print(f"集成模型在测试集上的准确率: {accuracy:.2f}%")
        
        # 创建混淆矩阵
        cm = confusion_matrix(y_test, test_predictions)
        if verbose:
            print("\n混淆矩阵:")
            print(cm)
        
        # 计算特征重要性 (如果可用)
        feature_importance = []
        
        # 为不同类型的集成模型计算特征重要性
        if ensemble_type == 'voting':
            # 对投票集成模型，计算所有支持特征重要性的基础模型的平均重要性
            try:
                importances = np.zeros(X.shape[1])
                count = 0
                estimator_items = []
                
                if hasattr(model, 'estimators_'):
                    if isinstance(model.estimators_, list):
                        # 如果是列表格式，转为带名称的元组
                        if len(model.estimators_) == len(base_models):
                            estimator_items = [(name, model.estimators_[i]) 
                                             for i, name in enumerate(base_models.keys())]
                        else:
                            # 使用简单索引作为名称
                            estimator_items = [(f"model_{i}", est) for i, est in enumerate(model.estimators_)]
                    else:
                        if verbose:
                            print("model.estimators_不是预期的列表格式")
                elif hasattr(model, 'named_estimators_'):
                    if hasattr(model.named_estimators_, 'items') and callable(getattr(model.named_estimators_, 'items')):
                        estimator_items = list(model.named_estimators_.items())
                    else:
                        if verbose:
                            print("model.named_estimators_不是字典格式")
                else:
                    if verbose:
                        print("模型没有estimators_或named_estimators_属性")
                    raise AttributeError("无法访问模型的估计器列表")
                
                if verbose:
                    print(f"找到了{len(estimator_items)}个基础模型用于计算特征重要性")
                
                for name, est in estimator_items:
                    if hasattr(est, 'feature_importances_'):
                        # 确保特征重要性数组维度一致
                        if len(est.feature_importances_) == len(importances):
                            importances += est.feature_importances_
                            count += 1
                            if verbose:
                                print(f"添加了{name}模型的特征重要性，累计{count}个模型")
                
                if count > 0:
                    importances /= count
                    
                    # 创建特征重要性结果
                    if feature_names is None:
                        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                    
                    # 确保特征名称和重要性匹配
                    feature_importance = []
                    for i, importance in enumerate(importances):
                        if i < len(feature_names):
                            feature_importance.append((feature_names[i], float(importance)))
                    
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    if verbose:
                        print("\n平均特征重要性 (Top 10):")
                        # 安全获取前10个元素
                        top_features = feature_importance[:min(10, len(feature_importance))]
                        for i, feature_tuple in enumerate(top_features):
                            name, importance = feature_tuple
                            print(f"{name}: {importance:.4f}")
            except Exception as e:
                if verbose:
                    print(f"计算投票集成模型平均特征重要性时出错: {e}")
                traceback.print_exc()
                # 创建一个空的特征重要性列表，避免后续处理出错
                feature_importance = []
        
        elif ensemble_type == 'stacking':
            # 对于堆叠集成模型，如果最终估计器支持特征重要性，则使用它
            if hasattr(model.final_estimator_, 'feature_importances_'):
                meta_importance = model.final_estimator_.feature_importances_
                
                # 创建特征重要性结果
                meta_feature_names = [f'meta_{i}' for i in range(len(meta_importance))]
                feature_importance = [(meta_feature_names[i], imp) for i, imp in enumerate(meta_importance)]
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                if verbose:
                    print("\n元学习器特征重要性:")
                    for name, importance in feature_importance:
                        print(f"{name}: {importance:.4f}")
            else:
                # 使用每个基础模型的特征重要性
                if ensemble_type == 'voting':
                    # 投票集成的estimators_是(name, estimator)对的列表
                    for name, est in model.estimators_:
                        if hasattr(est, 'feature_importances_'):
                            est_importances = est.feature_importances_
                            est_feature_names = feature_names or [f'feature_{i}' for i in range(len(est_importances))]
                            est_importance = list(zip(est_feature_names, est_importances))
                            est_importance.sort(key=lambda x: x[1], reverse=True)
                            
                            if verbose:
                                print(f"\n{name} 模型特征重要性 (Top 5):")
                                for fname, imp in est_importance[:5]:
                                    print(f"{fname}: {imp:.4f}")
                elif ensemble_type == 'stacking':
                    # 堆叠集成的estimators_只是估计器列表
                    base_models_names = list(base_models.keys())
                    for i, est in enumerate(model.estimators_):
                        name = base_models_names[i] if i < len(base_models_names) else f"estimator_{i}"
                        if hasattr(est, 'feature_importances_'):
                            est_importances = est.feature_importances_
                            est_feature_names = feature_names or [f'feature_{i}' for i in range(len(est_importances))]
                            est_importance = list(zip(est_feature_names, est_importances))
                            est_importance.sort(key=lambda x: x[1], reverse=True)
                            
                            if verbose:
                                print(f"\n{name} 模型特征重要性 (Top 5):")
                                for fname, imp in est_importance[:5]:
                                    print(f"{fname}: {imp:.4f}")
        
        # 评估收益率（如果提供了收益率数据）
        returns_evaluation = None
        if 'returns' in kwargs and kwargs['returns'] is not None:
            returns_data = kwargs['returns']
            if len(returns_data) >= len(y):
                # 确保返回数据与标签长度一致（使用最近的数据）
                returns_data = returns_data[-len(y):]
                cum_returns, annual_return, sharpe, max_dd, win_rate = evaluate_returns(y_test, test_predictions, returns_data[-len(y_test):])
                
                # 创建收益评估结果
                returns_evaluation = {
                    'cumulative_returns': cum_returns,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'win_rate': win_rate
                }
            else:
                if verbose:
                    print(f"警告: 提供的收益率数据长度({len(returns_data)})小于标签数据长度({len(y)})")
                
        # 创建结果字典
        results = {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
        
        if returns_evaluation:
            results['returns_evaluation'] = returns_evaluation
            
        return results
        
    except Exception as e:
        if verbose:
            print(f"训练集成模型时出错: {e}")
        traceback.print_exc()
        return None, None, 0, None, []


def evaluate_returns(y_true, predictions, returns):
    """
    评估预测策略的收益表现
    
    参数：
        y_true (numpy.ndarray): 实际标签，用于分析预测准确率与收益的关系
        predictions (numpy.ndarray): 模型预测结果（0表示预测下跌，1表示预测上涨）
        returns (numpy.ndarray): 对应的实际收益率序列，与y_true长度相同
        
    返回：
        tuple: 包含以下元素：
            - cum_returns (numpy.ndarray): 策略累计收益率序列
            - annualized_return (float): 年化收益率
            - sharpe_ratio (float): 夏普比率
            - max_drawdown (float): 最大回撤
            - win_rate (float): 预测为上涨且实际上涨的比率
    """
    try:
        # 确保数据长度一致
        if len(predictions) != len(returns):
            if verbose:
                print(f"警告: 预测结果长度({len(predictions)})与收益率序列长度({len(returns)})不匹配")
            predictions = predictions[-len(returns):] if len(predictions) > len(returns) else predictions
            returns = returns[-len(predictions):] if len(returns) > len(predictions) else returns
            
        # 计算策略收益率：只在预测为1(买入)时获得收益
        strategy_returns = returns * predictions
        
        # 计算累计收益
        cum_returns = (1 + strategy_returns).cumprod() - 1
        
        # 计算年化收益率(假设252个交易日)
        days = len(returns)
        if days > 0:
            annualized_return = ((1 + cum_returns[-1]) ** (252 / days)) - 1
        else:
            annualized_return = 0
            
        # 计算夏普比率
        daily_returns = strategy_returns
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        peak = np.maximum.accumulate(1 + cum_returns)
        drawdown = (1 + cum_returns) / peak - 1
        max_drawdown = np.max(drawdown)
        
        # 计算胜率
        win_rate = np.mean((predictions == 1) & (y_true == 1)) if np.sum(predictions == 1) > 0 else 0
        
        # 打印结果
        if verbose:
            print("\n===== 策略收益评估 =====")
            print(f"策略累计收益率: {cum_returns[-1]*100:.2f}%")
            print(f"年化收益率: {annualized_return*100:.2f}%")
            print(f"夏普比率: {sharpe_ratio:.2f}")
            print(f"最大回撤: {max_drawdown*100:.2f}%")
            print(f"上涨预测胜率: {win_rate*100:.2f}%")
        
        return cum_returns, annualized_return, sharpe_ratio, max_drawdown, win_rate
        
    except Exception as e:
        if verbose:
            print(f"评估策略收益时出错: {e}")
        return np.array([0]), 0, 0, 0, 0


def plot_feature_importance(feature_importance, model_type, top_n=20, figsize=(12, 10)):
    """
    绘制特征重要性图并返回matplotlib Figure对象
    
    参数：
        feature_importance (list): 包含(特征名称, 重要性得分)元组的列表
        model_type (str): 模型类型
        top_n (int): 显示前N个重要特征
        figsize (tuple): 图像尺寸
        
    返回：
        matplotlib.figure.Figure: 图像对象
    """
    try:
        # 提取前N个特征
        n_features = min(top_n, len(feature_importance))
        top_features = feature_importance[:n_features]
        
        # 获取特征名称和得分
        feature_names = [feature[0] for feature in top_features]
        importance_scores = [feature[1] for feature in top_features]
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_names)))
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        bars = ax.barh(range(len(feature_names)), importance_scores, color=colors)
        
        # 设置Y轴标签
        max_text_len = max([len(name) for name in feature_names])
        fontsize = min(14, max(8, 400 / max_text_len))  # 动态调整字体大小
        
        # 反转顺序以便最重要的特征在顶部
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(reversed(feature_names), fontsize=fontsize)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            value = importance_scores[i]
            text_x = value + 0.005
            text_y = bar.get_y() + bar.get_height()/2
            text_color = 'black'
            ax.text(text_x, text_y, f'{value:.3f}', va='center', fontsize=fontsize-1, color=text_color)
        
        # 添加标题和标签
        if 'ensemble' in model_type.lower():
            title = f"特征重要性 - 集成模型 (Top {n_features})"
        else:
            title = f"特征重要性 - {model_type} (Top {n_features})"
            
        ax.set_title(title, fontsize=15, pad=20)
        ax.set_xlabel('重要性', fontsize=12)
        
        # 去掉顶部和右侧的轴线
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # 设置X轴范围，给文本留出空间
        max_score = max(importance_scores) if importance_scores else 0
        ax.set_xlim([0, max_score * 1.15])
        
        # 添加网格线
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        
        # 反转Y轴，使最重要的特征在顶部
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        if verbose:
            print(f"绘制特征重要性图时出错: {e}")
        traceback.print_exc()
        # 返回一个空图
        return plt.figure()


if __name__ == '__main__':
    try:
        # 创建一些简单的测试数据
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        feature_names = [f'feature_{i}' for i in range(10)]
        returns = np.random.normal(0.001, 0.02, 200)  # 随机收益率，均值0.1%，标准差2%
        
        if verbose:
            print("===== 测试单一模型训练 =====")
        results = train_ml_model(X, y, model_type='rf', feature_names=feature_names, returns=returns)
        
        if verbose:
            print("\n===== 测试集成模型训练 =====")
        ensemble_results = train_ensemble_model(X, y, ensemble_type='voting', feature_names=feature_names, 
                                             auto_model_selection=True, returns=returns)
                                             
        if verbose:
            print("\n测试完成!")
    except Exception as e:
        if verbose:
            print(f"测试时出错: {e}")
        traceback.print_exc()
