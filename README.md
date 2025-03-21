# A股量化交易环境

这是一个用于A股量化交易的Python环境，包含了常用的量化分析和交易工具。

## 环境设置

### 激活虚拟环境

```bash
# 在macOS下
source venv/bin/activate

# 在Windows下
venv\Scripts\activate
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 关于TA-Lib

TA-Lib是一个技术分析库，需要额外安装C语言依赖。在macOS上，您可以使用以下命令安装：

```bash
# 使用Homebrew安装TA-Lib C语言库
brew install ta-lib

# 安装Python绑定
pip install TA-Lib
```

如果遇到安装问题，可以参考[TA-Lib官方文档](https://github.com/TA-Lib/ta-lib-python)。

## 包含的主要库

- **数据获取**: 
  - Tushare: 提供A股历史数据和基本面数据
  - Baostock: 提供A股、指数、基金的历史数据
  - AKShare: 全量金融数据接口

- **数据分析**:
  - Pandas: 数据分析和处理
  - NumPy: 科学计算
  - Matplotlib: 数据可视化
  - PyEcharts: 交互式图表库

- **量化分析**:
  - TA-Lib: 技术分析库
  - Backtrader: 回测框架
  - Alphalens: 因子分析
  - Pyfolio: 投资组合分析
  - Empyrical: 风险和绩效分析

## 使用示例

```python
# 创建简单的示例脚本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak

# 使用AKShare获取上证指数数据
df = ak.stock_zh_index_daily(symbol="sh000001")
print(df.head())

# 绘制上证指数走势图
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['close'])
plt.title('上证指数走势')
plt.xlabel('日期')
plt.ylabel('指数值')
plt.grid(True)
plt.show()
```

## 增强版机器学习预测模型

项目包含了一个增强版的机器学习预测模型，用于预测股票未来的涨跌。

### 核心功能

- **灵活的预测时间周期**: 支持1天(超短线)、5天、10天(中线)、20天(长线)预测
- **增强的特征工程**: 
  - MACD (动量指标)
  - ATR (平均真实波幅)
  - RSI (相对强弱指数)
  - 板块指数关联
  - 多周期移动平均线
  - 成交量变化率
  - 股票-板块相关性指标等
- **多种机器学习算法**:
  - 随机森林 (默认)
  - XGBoost
  - LightGBM
- **高级功能**:
  - 特征重要性分析和选择
  - 时间序列交叉验证
  - 预测结果可视化
  - 模型性能评估

### 使用方法

```bash
# 基本用法(使用默认参数)
python enhanced_ml_prediction.py

# 指定股票和预测天数
python enhanced_ml_prediction.py --stock 600519.SH --days 20

# 使用XGBoost模型，预测贵州茅台未来5天走势
python enhanced_ml_prediction.py --stock 600519.SH --days 5 --model xgb

# 完整参数说明
python enhanced_ml_prediction.py --help
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --stock | 股票代码 | 002179.SZ (中航光电) |
| --sector | 板块指数代码 | 399967.SZ (中证军工指数) |
| --days | 预测天数 | 10 |
| --period | 回溯历史数据天数 | 365 |
| --model | 模型类型 (rf/xgb/lgbm) | rf |
| --no-cv | 不使用交叉验证 | 默认使用 |
| --no-feature-select | 不使用特征选择 | 默认使用 |
| --no-sector | 不使用板块数据 | 默认使用 |

### 输出内容

- 预测信号: 买入/观望/卖出
- 预测准确率评估
- 预测收益率分析
- 可视化图表 (保存在images目录下)

## 注意事项

- 部分数据接口(如Tushare)可能需要注册获取token
- 某些库(如TA-Lib)可能需要特定的安装环境
- 机器学习预测仅供参考，不构成投资建议
- 实际交易中请结合基本面和其他分析方法

# A股量化交易机器学习预测系统

一个基于机器学习的股票量化交易预测系统，专为A股市场设计，集成了技术指标分析、板块关联分析与实时预测功能。

## 核心功能

### 1. 增强版机器学习预测模型 (`enhanced_ml_prediction.py`)

- **灵活的预测时间周期**: 支持1天、5天、10天、20天预测
- **增强的特征工程**:
  - MACD、ATR、RSI等技术指标
  - 板块指数关联性分析
  - 多周期移动平均线
  - 成交量变化分析
  - A股涨跌停特性优化
- **多种机器学习算法**:
  - 随机森林(默认)
  - XGBoost
  - LightGBM
- **高级功能**:
  - 特征重要性分析
  - 时间序列交叉验证
  - 可视化预测结果
  - 性能评估与止损止盈模拟

### 2. 实时股票分析 (`realtime_stock_analysis.py`)

- **实时数据分析**: 结合历史数据和实时行情，提供即时预测
- **多维度信号解读**: 包括趋势、RSI、MACD、成交量异常等
- **可视化分析**: 生成带有信号标记的K线图和技术指标
- **友好的参数配置**: 支持通过命令行灵活配置分析参数

## 安装要求

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 训练预测模型

```bash
python enhanced_ml_prediction.py  --stock 600797.SH --days 365  --model xgb -f 5  --feature-selection --feature-method combined  --use-pca --hyperopt
```

参数说明:
- `--stock`: 股票代码，默认002179.SZ (中航光电)
- `--days`: 预测天数，可选1/5/10/20，默认10
- `--model`: 模型类型，可选rf/xgb/lgbm，默认rf
- `--period`: 回溯历史数据天数，默认365
- `--no-cv`: 不使用交叉验证
- `--no-feature-select`: 不使用特征选择
- `--no-sector`: 不使用板块数据

### 2. 实时股票分析

```bash
python realtime_stock_analysis.py --stock 002179.SZ --token YOUR_TUSHARE_TOKEN
```

或手动输入实时数据:

```bash
python realtime_stock_analysis.py --stock 002179.SZ --manual
```

参数说明:
- `--token`: Tushare API 令牌
- `--stock`: 股票代码，默认002179.SZ
- `--model`: 模型文件路径，默认model_rf.pkl
- `--days`: 历史数据天数，默认100
- `--manual`: 使用手动输入的实时数据，而非API获取

## 输出示例

1. **预测结果**:
   - 买入/卖出/持有信号
   - 预测置信度
   - 技术指标分析

2. **可视化图表**:
   - 股价走势与预测信号
   - 移动平均线
   - 成交量分析
   - 信号强度标记

## 注意事项

- 需要Tushare Pro API接口权限
- 预测结果仅供参考，请勿直接用于实盘交易
- 涨跌停股票会自动调整为持有信号
- 模型训练后会自动保存，便于实时分析调用

## 技术栈

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, tushare
- joblib (模型保存与加载)
- XGBoost, LightGBM (可选)

## 性能指标参考

根据不同股票和市场环境，性能可能有所不同:
- 交叉验证平均准确率: 约51%-55%
- 买入信号5天后平均收益率: 约4%-7%
- 应用止损止盈后的平均收益率: 约3%-6%

## 开发者

- 作者: Wang Yan
- 日期: 2025年3月
