# 增强版A股预测模型（模块化版本）

这是增强版A股预测模型的模块化重构版本，旨在提高代码的可读性、可维护性和可扩展性。该模型使用多种机器学习算法和技术指标来预测A股股票的走势。

## 模块结构

项目采用了模块化的结构设计，将不同功能分离到独立的模块中：

- `__init__.py`: 包信息和元数据
- `data.py`: 数据获取和预处理
- `features.py`: 特征工程和特征选择
- `models.py`: 模型训练、评估和预测
- `plotting.py`: 数据可视化功能
- `utils.py`: 工具函数和交易策略
- `main.py`: 主程序入口和命令行界面

## 主要功能

1. **灵活的预测时间周期**:
   - 支持1天、5天、10天、20天等多种预测周期

2. **增强的特征工程**:
   - 技术指标 (MACD、ATR、RSI等)
   - 多周期移动平均线
   - 成交量变化率分析

3. **多种机器学习算法**:
   - 随机森林 (默认)
   - XGBoost (如已安装)
   - LightGBM (如已安装)
   - 集成模型 (投票和堆叠)

4. **高级功能**:
   - 特征重要性分析
   - 时间序列交叉验证
   - 交易信号生成
   - 止损止盈策略
   - 回测交易模拟

## 使用方法

### 基本用法

```bash
# 直接运行示例脚本
python run_stock_prediction.py

# 或使用命令行参数
python run_stock_prediction.py --code sh.600000 --predict_days 5 --model rf --technical_indicators --price_features --volume_features
```

### 主要参数

- `--code`: 股票代码，如 sh.600000 或 sz.000001
- `--start_date`: 开始日期，格式为YYYY-MM-DD
- `--end_date`: 结束日期，格式为YYYY-MM-DD
- `--predict_days`: 预测未来天数，默认为5
- `--model`: 使用的模型类型，可选 'rf'、'xgb'、'lgbm'或'ensemble'
- `--technical_indicators`: 是否使用技术指标特征
- `--volume_features`: 是否使用成交量特征
- `--price_features`: 是否使用价格特征
- `--feature_selection`: 是否使用特征选择
- `--cv`: 是否使用交叉验证
- `--optimize`: 是否进行超参数优化
- `--show_plots`: 是否显示图表

### 完整参数列表

运行以下命令查看所有可用参数:

```bash
python -m modular_version.main --help
```

## 输出结果

运行后将会生成以下输出:

1. **数据可视化**:
   - 股票历史走势图
   - 预测结果可视化
   - 特征重要性图表
   - 混淆矩阵

2. **性能指标**:
   - 准确率、精确率、召回率、F1分数
   - 模拟交易收益和收益率
   - 交易记录

3. **输出文件**:
   - 结果JSON文件
   - 交易记录CSV文件
   - 图表PNG文件

## 依赖库

- `numpy` 和 `pandas`: 数据处理
- `scikit-learn`: 机器学习算法
- `matplotlib` 和 `seaborn`: 数据可视化
- `baostock`: A股数据获取
- `xgboost` 和 `lightgbm` (可选): 高级模型

可以使用以下命令安装必要的依赖:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn baostock
```

## 示例

```python
# 导入必要的模块
from modular_version.data import fetch_stock_data
from modular_version.features import extract_features
from modular_version.models import train_ml_model
from modular_version.plotting import plot_prediction_results

# 获取股票数据
df = fetch_stock_data("sh.600000", "2023-01-01", "2023-12-31")

# 提取特征
X, feature_names = extract_features(df, use_technical=True, use_volume=True)

# 训练模型
model, predictions, accuracy, cm, feature_importance = train_ml_model(X, y, model_type='rf')

# 可视化结果
plot_prediction_results(df, predictions, 5, "sh.600000")
```
