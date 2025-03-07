#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 成功提示
echo "A股量化交易环境设置完成！"
echo
echo "——————————————————————————————————————————"
echo "如需安装TA-Lib技术分析库，请执行以下步骤："
echo "1. 安装C语言库依赖：brew install ta-lib"
echo "2. 安装Python绑定：pip install TA-Lib"
echo "——————————————————————————————————————————"
echo
echo "你可以通过'source venv/bin/activate'命令激活虚拟环境"
echo "使用'python example.py'运行示例脚本"
