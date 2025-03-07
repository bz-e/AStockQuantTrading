#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Stock Prediction Model - Main Program

This program is the main entry point for the enhanced machine learning prediction model,
supporting various configuration parameters from the command line.
"""

import os
import sys
import argparse
import logging
import time
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import custom modules
from modular_version.data import get_stock_data, prepare_data, add_index_features
from modular_version.features import calculate_technical_indicators, prepare_features, get_important_features
from modular_version.models import train_ml_model, train_ensemble_model
from modular_version.plotting import plot_stock_data, plot_prediction_results, plot_feature_importance, plot_confusion_matrix
from modular_version.utils import generate_trading_signals, simulate_trading, calculate_metrics, save_results, create_stop_loss_profit_strategy, format_time, apply_stop_loss_take_profit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_prediction')


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Namespace containing all command line arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced Stock Prediction Model')
    
    # Basic parameters
    parser.add_argument('--code', type=str, required=True, help='Stock code, e.g., sh.600000 or sz.000001')
    parser.add_argument('--start_date', type=str, default=None, help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None, help='End date in format YYYY-MM-DD')
    parser.add_argument('--predict_days', type=int, default=5, help='Number of days to predict, default is 5')
    
    # Data related parameters
    parser.add_argument('--data_source', type=str, default='baostock', choices=['baostock', 'local'], help='Data source, default is baostock')
    parser.add_argument('--local_data', type=str, default=None, help='Local data file path (if using local data)')
    parser.add_argument('--save_data', action='store_true', help='Whether to save the retrieved data')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save data')
    
    # Feature engineering parameters
    parser.add_argument('--technical_indicators', action='store_true', help='Whether to use technical indicators as features')
    parser.add_argument('--volume_features', action='store_true', help='Whether to use volume features')
    parser.add_argument('--price_features', action='store_true', help='Whether to use price features')
    parser.add_argument('--market_index', action='store_true', help='Whether to include market index features')
    parser.add_argument('--market_index_code', type=str, default='sh.000001', help='Market index code, default is sh.000001 (Shanghai Index)')
    parser.add_argument('--rolling_windows', type=str, default='5,10,20', help='Rolling window sizes, comma-separated, e.g., 5,10,20')
    parser.add_argument('--feature_selection', action='store_true', help='Whether to perform feature selection')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'xgb', 'lgbm', 'ensemble'], help='Model type to use')
    parser.add_argument('--ensemble_type', type=str, default='voting', choices=['voting', 'stacking'], help='Ensemble model type')
    parser.add_argument('--cv', action='store_true', help='Whether to use cross-validation')
    parser.add_argument('--optimize', action='store_true', help='Whether to perform hyperparameter optimization')
    
    # Trading strategy parameters
    parser.add_argument('--stop_loss', type=float, default=0.05, help='Stop-loss percentage, default is 0.05 (5%)')
    parser.add_argument('--take_profit', type=float, default=0.1, help='Take-profit percentage, default is 0.1 (10%)')
    parser.add_argument('--initial_capital', type=float, default=100000, help='Initial capital, default is 100000')
    parser.add_argument('--commission_rate', type=float, default=0.0003, help='Commission rate, default is 0.0003 (0.03%)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--show_plots', action='store_true', help='Whether to display plots')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--verbose', action='store_true', help='Whether to display detailed information')
    
    return parser.parse_args()


def setup_output_directory(output_dir):
    """
    Set up the output directory, creating it if it does not exist
    
    Parameters:
        output_dir (str): Path to the output directory
    
    Returns:
        str: Path to the created output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
        
    # Create a subdirectory with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(output_dir, timestamp)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info(f"Created result directory: {result_dir}")
        
    return result_dir


def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up the output directory
    result_dir = setup_output_directory(args.output_dir)
    
    # Record the start time
    start_time = time.time()
    
    # Set the logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Retrieve stock data
        logger.info(f"Retrieving historical data for stock {args.code}")
        
        # If no date range is specified, use default values
        if args.start_date is None:
            args.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if args.end_date is None:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
            
        if args.data_source == 'baostock':
            df = get_stock_data(args.code, args.start_date, args.end_date)
        else:
            # Load data from a local file
            if args.local_data is None:
                logger.error("Local data source specified, but no data file path provided")
                return
            df = pd.read_csv(args.local_data, parse_dates=['date'])
            
        if df is None or len(df) == 0:
            logger.error("No data retrieved")
            return
            
        logger.info(f"Retrieved {len(df)} data records")
        
        # Save the raw data if requested
        if args.save_data:
            # Create data directory if it doesn't exist
            if not os.path.exists(args.data_dir):
                os.makedirs(args.data_dir)
                
            # Format the stock code for the filename
            formatted_code = args.code.replace('.', '_')
            
            # Save to CSV
            data_file = os.path.join(args.data_dir, f"{formatted_code}_{args.start_date}_{args.end_date}.csv")
            df.to_csv(data_file, index=False)
            logger.info(f"Raw data saved to: {data_file}")
        
        # Plot stock data
        stock_chart_path = os.path.join(result_dir, f"{args.code}_history.png")
        plot_stock_data(df, args.code, save_path=stock_chart_path, show=args.show_plots)
        
        # Retrieve market index data if requested
        if args.market_index:
            logger.info(f"Retrieving market index data: {args.market_index_code}")
            index_df = get_stock_data(args.market_index_code, args.start_date, args.end_date)
            
            if index_df is None or len(index_df) == 0:
                logger.warning(f"Could not retrieve market index data for {args.market_index_code}")
            else:
                logger.info(f"Retrieved {len(index_df)} market index data records")
                
                # Add index features to the stock dataframe
                df = add_index_features(df, index_df, logger)
                logger.info("Added market index features to the stock data")
        
        # Extract features
        if args.technical_indicators:
            logger.info("Calculating technical indicators...")
            df = calculate_technical_indicators(df)
            
        # Parse rolling window values
        if hasattr(args, 'rolling_windows') and args.rolling_windows:
            try:
                rolling_windows = [int(w) for w in args.rolling_windows.split(',')]
            except ValueError:
                logger.warning(f"Invalid rolling window values: {args.rolling_windows}, using default values")
                rolling_windows = [5, 10, 20]
            
        # Prepare features
        logger.info(f"Preparing feature data, predicting {args.predict_days} days...")
        X, y, feature_names = prepare_features(
            df, 
            prediction_days=args.predict_days,
            feature_selection=args.feature_selection
        )
        
        if X is None or len(X) == 0:
            logger.error("Feature extraction failed or generated feature set is empty")
            return None
            
        logger.info(f"Prepared {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature selection
        if args.feature_selection and X.shape[1] > 10:
            logger.info("Performing feature selection...")
            # Here, we would normally use a trained model to evaluate feature importance
            # However, we don't have a trained model yet, so we'll skip this step
            # Instead, we'll use the original feature set
            X_final = X
            final_feature_names = feature_names
        else:
            X_final = X
            final_feature_names = feature_names
            
        # Train the model
        logger.info(f"Training {args.model} model...")
        start_time = time.time()
        
        if args.model == 'ensemble':
            # 使用集成模型
            logger.info("Using ensemble model...")
            # 默认为投票集成，如果指定了ensemble_type则使用指定的类型
            ensemble_type = getattr(args, 'ensemble_type', 'voting')
            logger.info(f"Ensemble type: {ensemble_type}")
            
            result_dict = train_ensemble_model(
                X_final, y, 
                ensemble_type=ensemble_type,
                feature_names=final_feature_names,
                use_feature_selection=args.feature_selection,
                use_hyperopt=args.optimize,
                auto_model_selection=True
            )
        else:
            # 使用单一模型
            logger.info(f"Using single model: {args.model}")
            result_dict = train_ml_model(
                X_final, y, 
                model_type=args.model, 
                feature_names=final_feature_names,
                use_cv=args.cv, 
                use_feature_selection=args.feature_selection,
                use_hyperopt=args.optimize
            )
        
        if result_dict is None or 'model' not in result_dict:
            logger.error("Model training failed")
            return None
            
        # 将结果打包成字典返回
        results = {
            'code': args.code,
            'predict_days': args.predict_days,
            'model': str(result_dict['model'].__class__.__name__),
            'model_object': result_dict['model'],
            'accuracy': result_dict['accuracy'],
            'predictions': result_dict['predictions'],
            'confusion_matrix': result_dict['confusion_matrix'],
            'feature_importance': result_dict['feature_importance'],
            'profit': result_dict.get('total_profit', 0),
            'profit_rate': result_dict.get('profit_percentage', 0),
            'trade_count': result_dict.get('total_trades', 0),
        }
        
        # 处理特征重要性，转换为正确的格式
        try:
            if isinstance(result_dict['feature_importance'], (list, np.ndarray)) and not isinstance(result_dict['feature_importance'][0], tuple):
                if len(result_dict['feature_importance']) == len(final_feature_names):
                    # 创建元组列表
                    top_features = [(final_feature_names[i], float(importance)) for i, importance in 
                                   enumerate(result_dict['feature_importance'])]
                    # 排序
                    top_features.sort(key=lambda x: x[1], reverse=True)
                    results['top_features'] = top_features[:10]  # 只保留前10个最重要的特征
                else:
                    results['top_features'] = []
            elif isinstance(result_dict['feature_importance'], list) and isinstance(result_dict['feature_importance'][0], tuple):
                # 已经是元组列表格式
                results['top_features'] = sorted(result_dict['feature_importance'], key=lambda x: x[1], reverse=True)[:10]
            else:
                results['top_features'] = []
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"Error processing feature importance: {str(e)}")
            results['top_features'] = []
        
        # 如果结果字典中包含策略收益评估结果，添加到返回值中
        if result_dict.get('returns_evaluation'):
            results.update({
                'strategy_returns': result_dict['returns_evaluation'].get('cumulative_return', 0),
                'strategy_sharpe': result_dict['returns_evaluation'].get('sharpe_ratio', 0),
                'strategy_drawdown': result_dict['returns_evaluation'].get('max_drawdown', 0),
                'strategy_win_rate': result_dict['returns_evaluation'].get('win_rate', 0)
            })
        
        logger.info(f"Model training complete, accuracy: {result_dict['accuracy']:.2f}%")
        
        # 从结果字典中提取所需信息用于可视化和进一步处理
        model = result_dict['model']
        predictions = result_dict['predictions']
        accuracy = result_dict['accuracy']
        cm = result_dict['confusion_matrix']
        feature_importance = result_dict['feature_importance']
        
        # 绘制混淆矩阵
        cm_path = os.path.join(result_dir, f"{args.code}_{args.model}_confusion_matrix.png")
        plot_confusion_matrix(cm, class_names=['Down', 'Up'], save_path=cm_path, show=args.show_plots)
        
        # 绘制特征重要性
        if feature_importance is not None and len(feature_importance) > 0:
            fi_path = os.path.join(result_dir, f"{args.code}_{args.model}_feature_importance.png")
            try:
                # 将特征名称和重要性值整合为元组列表
                if isinstance(feature_importance, (list, np.ndarray)) and not isinstance(feature_importance[0], tuple):
                    # 数组格式的特征重要性需要与特征名称配对
                    if len(feature_importance) == len(final_feature_names):
                        feature_importance_tuples = [(final_feature_names[i], float(importance)) for i, importance in 
                                                    enumerate(feature_importance)]
                        # 按重要性排序
                        feature_importance_tuples.sort(key=lambda x: x[1], reverse=True)
                        plot_feature_importance(feature_importance_tuples, save_path=fi_path, show=args.show_plots)
                    else:
                        logger.warning(f"特征重要性长度 ({len(feature_importance)}) 与特征名称长度 ({len(final_feature_names)}) 不匹配")
                elif isinstance(feature_importance, list) and isinstance(feature_importance[0], tuple):
                    # 已经是元组列表格式，直接使用
                    plot_feature_importance(feature_importance, save_path=fi_path, show=args.show_plots)
                else:
                    logger.warning(f"不支持的特征重要性格式: {type(feature_importance)}")
            except Exception as e:
                logger.error(f"Error plotting feature importance: {str(e)}")
            
        # 生成交易信号
        logger.info("Generating trading signals...")
        try:
            # 确保数据长度匹配
            if len(predictions) != len(df):
                logger.warning(f"Data length mismatch: df={len(df)}, predictions={len(predictions)}")
                # 找出有效的数据点（没有NaN的行）
                valid_df = df.dropna()
                
                if len(valid_df) != len(predictions):
                    logger.warning(f"Valid data points ({len(valid_df)}) still don't match predictions ({len(predictions)})")
                    # 使用最后的预测数据点数量 - 这假设预测点是从最新的有效数据生成的
                    aligned_df = df.iloc[-len(predictions):].copy()
                    logger.info(f"Using the most recent {len(predictions)} data points for signal generation")
                else:
                    # 有效数据点数量和预测数量匹配，使用有效数据
                    aligned_df = valid_df.copy()
                    logger.info(f"Using {len(valid_df)} valid data points (without NaN) for signal generation")
                
                signal_df = generate_trading_signals(aligned_df, predictions)
            else:
                signal_df = generate_trading_signals(df, predictions)
                
            logger.info(f"Generated trading signals: Buy {signal_df['signal'].value_counts().get(1, 0)} times, Sell {signal_df['signal'].value_counts().get(-1, 0)} times")
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            traceback.print_exc()
            signal_df = None
        
        # 创建止损/止盈策略
        logger.info(f"Applying stop-loss ({args.stop_loss*100}%) and take-profit ({args.take_profit*100}%) strategy...")
        try:
            # 使用已经对齐的数据
            if signal_df is not None:
                close_prices = signal_df['close'].values
                dates = signal_df.index
                
                # 应用止损止盈策略
                signals = apply_stop_loss_take_profit(
                    signals=signal_df['signal'].values,
                    close_prices=close_prices,
                    stop_loss=args.stop_loss, 
                    take_profit=args.take_profit
                )
                
                # 更新信号
                signal_df['signal'] = signals
                logger.info(f"Applied stop-loss and take-profit strategy to {len(signals)} signals")
            else:
                logger.warning("Cannot apply stop-loss and take-profit: no valid signal dataframe")
                
        except Exception as e:
            logger.error(f"Error applying stop-loss and take-profit: {str(e)}")
            traceback.print_exc()
        
        # 模拟交易
        logger.info("Simulating trading...")
        try:
            if signal_df is not None:
                # 使用已对齐的信号数据框进行模拟交易
                profit, profit_rate, trades = simulate_trading(signal_df, initial_capital=args.initial_capital)
                logger.info(f"Trading simulation completed: Total return {profit:.2f}, Return rate {profit_rate:.2f}%, Number of trades {len(trades) if isinstance(trades, pd.DataFrame) else 0}")
                
                # 更新结果中的模拟交易收益
                results['profit'] = profit
                results['profit_rate'] = profit_rate
                results['trade_count'] = len(trades) if isinstance(trades, pd.DataFrame) else 0
                
                # 评估交易性能指标
                if 'signal' in signal_df.columns and 'actual_signal' in signal_df.columns:
                    # 计算准确率、精确率、召回率等指标
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                    
                    # 过滤掉信号为0的行，只评估有明确买入/卖出信号的决策
                    active_signals = signal_df[signal_df['signal'] != 0]
                    if len(active_signals) > 0:
                        # 将-1映射为0以匹配二分类指标计算要求
                        y_pred = (active_signals['signal'] > 0).astype(int)
                        y_true = (active_signals['actual_signal'] > 0).astype(int)
                        
                        if len(y_pred) > 0 and len(y_true) > 0:
                            accuracy = accuracy_score(y_true, y_pred) * 100
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            
                            logger.info(f"Calculated performance metrics: Accuracy={accuracy:.2f}%, F1 score={f1:.4f}")
                            
                            # 添加到结果中
                            results['signal_accuracy'] = accuracy
                            results['signal_f1'] = f1
            else:
                logger.error("No trading signal data available for simulation")
        except Exception as e:
            logger.error(f"Simulated trading failed: {str(e)}")
            traceback.print_exc()
        
        # 保存结果
        results_file = os.path.join(result_dir, f"{args.code}_{args.model}_results.json")
        
        # 计算指标
        try:
            metrics = calculate_metrics(predictions, y)
            results['metrics'] = metrics
            logger.info(f"Calculated model metrics")
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            traceback.print_exc()
        
        # 保存结果到文件
        try:
            save_results(results, results_file)
            logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            traceback.print_exc()
        
        # 绘制预测结果
        pred_path = os.path.join(result_dir, f"{args.code}_{args.model}_prediction.png")
        try:
            # 使用之前生成trading signals时已对齐的数据
            if signal_df is not None:
                # 使用信号数据框，它已经包含了对齐的数据和预测
                plot_prediction_results(signal_df, predictions, args.predict_days, args.code, save_path=pred_path, show=args.show_plots)
                logger.info(f"Prediction chart saved to: {pred_path}")
            else:
                # 如果没有信号数据框，尝试重新对齐
                logger.warning(f"Length mismatch for prediction plotting: df={len(df)}, predictions={len(predictions)}")
                
                # 使用dropna()获取有效数据点
                valid_df = df.dropna()
                
                if len(valid_df) != len(predictions):
                    # 使用最近的有效数据点
                    plot_df = df.iloc[-len(predictions):].copy()
                    logger.info(f"Using last {len(predictions)} data points for prediction plotting")
                else:
                    # 使用所有有效数据点
                    plot_df = valid_df.copy()
                    logger.info(f"Using {len(valid_df)} valid data points for prediction plotting")
                
                plot_prediction_results(plot_df, predictions, args.predict_days, args.code, save_path=pred_path, show=args.show_plots)
                logger.info(f"Prediction chart saved to: {pred_path}")
        except Exception as e:
            logger.error(f"Error plotting prediction results: {str(e)}")
            traceback.print_exc()
        
        # 显示总结
        elapsed_time = time.time() - start_time
        logger.info(f"Processing complete, elapsed time: {format_time(elapsed_time)}")
        logger.info(f"All results saved to: {result_dir}")
        
        # 如果需要显示图表
        if args.show_plots:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as e:
                logger.error(f"Failed to display plots: {e}")
                
        logger.info(f"Prediction completed. Model: {args.model}, Accuracy: {accuracy:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
