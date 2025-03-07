#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票预测模型运行示例

这个脚本展示了如何使用模块化的股票预测模型进行预测，
并提供基于预测结果的交易建议
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主模块
from modular_version.main import main, parse_arguments

# 设置日志
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('stock_prediction')

if __name__ == "__main__":
    # 如果直接运行该文件，将参数传递给main函数执行
    import sys
    import argparse
    
    # 创建一个临时解析器来检测是否显式提供了--code参数
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--code', type=str)
    temp_args, _ = temp_parser.parse_known_args()
    
    # 如果没有提供--code参数，则使用默认参数
    if temp_args.code is None:
        # 设置默认参数
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 移除已存在的参数（除了脚本名称）
        sys.argv = [sys.argv[0]]
        
        # 使用默认值：以上证指数为例
        sys.argv.extend([
            "--code", "sh.000001",  # 上证指数
            "--start_date", start_date,
            "--end_date", end_date,
            "--predict_days", "5",
            "--technical_indicators",
            "--volume_features",
            "--price_features",
            "--market_index",  # 添加大盘指数特征
            "--model", "rf",
            "--cv",
            "--save_data",
            "--show_plots"
        ])
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging()
    
    # 执行主函数 - 不传参数，main函数会自己解析命令行参数
    results = main()
    
    if results:
        print("\n" + "="*50)
        print("股票预测分析报告".center(46))
        print("="*50)
        
        # 显示基本信息
        print(f"\n📈 股票代码: {args.code}")
        print(f"📅 分析周期: {args.start_date} 至 {args.end_date}")
        print(f"🔮 预测天数: {args.predict_days}天")
        
        # 提取预测结果
        predictions = results.get('predictions', None)
        model = results.get('model', None)
        metrics = results.get('metrics', {})
        returns_eval = results.get('returns_evaluation', {})
        feature_importance = results.get('feature_importance', [])
        
        # 显示模型性能
        if metrics:
            print("\n📊 模型性能指标:")
            print(f"   准确率: {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   精确率: {metrics.get('precision', 0)*100:.2f}%")
            print(f"   召回率: {metrics.get('recall', 0)*100:.2f}%")
            print(f"   F1分数: {metrics.get('f1', 0):.4f}")
        
        # 显示策略回测结果
        if returns_eval:
            print("\n💰 策略回测结果:")
            print(f"   累计收益率: {returns_eval.get('cumulative_return', 0)*100:.2f}%")
            print(f"   年化收益率: {returns_eval.get('annual_return', 0)*100:.2f}%")
            print(f"   夏普比率: {returns_eval.get('sharpe_ratio', 0):.2f}")
            print(f"   最大回撤: {returns_eval.get('max_drawdown', 0)*100:.2f}%")
            print(f"   胜率: {returns_eval.get('win_rate', 0)*100:.2f}%")
        
        # 模型类型信息
        model_type = "集成模型" if args.model == 'ensemble' else "单一模型"
        if hasattr(model, 'estimator_names_') and len(model.estimator_names_) > 0:
            model_detail = f"({', '.join(model.estimator_names_)})"
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            model_detail = f"({len(model.estimators_)}个基础模型)"
        else:
            model_detail = f"({args.model})" if hasattr(args, 'model') else ""
        
        print(f"\n🤖 使用模型: {model_type}{model_detail}")
        
        # 特征重要性信息
        if feature_importance and len(feature_importance) > 0:
            print("\n🔑 关键影响因素 (Top 5):")
            for i, (feature, importance) in enumerate(feature_importance[:5]):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        # 提取最近的预测结果
        if predictions is not None and hasattr(predictions, '__len__') and len(predictions) >= args.predict_days:
            # 确保预测结果是列表类型而非NumPy数组
            recent_predictions = predictions[-args.predict_days:].tolist() if hasattr(predictions, 'tolist') else predictions[-args.predict_days:]
        else:
            recent_predictions = []
        
        if recent_predictions and len(recent_predictions) > 0:
            # 分析预测趋势
            up_days = sum(1 for p in recent_predictions if p > 0)
            down_days = sum(1 for p in recent_predictions if p < 0)
            
            # 计算信号强度 (预测值的绝对平均值)
            signal_strength = sum(abs(p) for p in recent_predictions) / len(recent_predictions)
            
            # 确定总体趋势
            if up_days > down_days:
                trend = "上涨"
                trend_emoji = "📈"
            elif down_days > up_days:
                trend = "下跌"
                trend_emoji = "📉"
            else:
                trend = "震荡"
                trend_emoji = "📊"
            
            # 强度评级
            if signal_strength > 0.015:
                strength = "强"
            elif signal_strength > 0.005:
                strength = "中"
            else:
                strength = "弱"
            
            print(f"\n{trend_emoji} 未来{args.predict_days}天预测趋势: {trend}({strength})")
            
            # 详细日预测
            print("\n📆 每日预测:")
            for i, pred in enumerate(recent_predictions):
                day = i + 1
                direction = "上涨" if pred > 0 else "下跌" if pred < 0 else "持平"
                emoji = "🔼" if pred > 0 else "🔽" if pred < 0 else "➡️"
                print(f"   第{day}天: {emoji} {direction} {abs(pred)*100:.2f}%")
            
            # 生成交易建议
            print("\n💡 交易建议:")
            
            # 根据趋势和强度生成不同建议
            if trend == "上涨":
                if strength == "强":
                    print("   ✅ 强烈推荐买入，市场看涨信号明确")
                    print("   📌 建议设置止损位: 当前价格的95%")
                elif strength == "中":
                    print("   ✅ 建议适量买入，市场有上升趋势")
                    print("   📌 建议设置止损位: 当前价格的97%")
                else:
                    print("   ✅ 可小额试探性买入，趋势略微偏向上涨")
                    print("   📌 建议严格设置止损位: 当前价格的98%")
            elif trend == "下跌":
                if strength == "强":
                    print("   ❌ 不建议现在买入，市场看跌信号明确")
                    print("   📌 建议持币观望或考虑做空")
                elif strength == "中":
                    print("   ⚠️ 暂时观望，市场下行趋势明显")
                    print("   📌 如已持有，建议设置止损位保护仓位")
                else:
                    print("   ⚠️ 保持谨慎，短期内市场可能有小幅回调")
                    print("   📌 可以制定分批减仓计划")
            else:  # 震荡
                print("   ⚖️ 市场处于震荡期，建议观望或做短线交易")
                print("   📌 避免重仓操作，控制单笔交易风险")
            
            # 风险警示
            print("\n⚠️ 风险提示:")
            print("   • 预测仅供参考，实际交易请结合基本面和技术面分析")
            print("   • 历史表现不代表未来，市场存在不可预见风险因素")
            print("   • 请根据个人风险承受能力做出投资决策")
            
        print("\n" + "="*50 + "\n")
    else:
        print("预测过程中出现错误，未能获取结果。")
