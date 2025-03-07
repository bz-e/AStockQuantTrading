#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting Module - Responsible for various visualization functions

This module implements various visualization functions, including stock data visualization,
prediction result visualization, feature importance visualization, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime, timedelta

# Set matplotlib basic configuration
try:
    # Use English fonts to ensure proper display
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Adjust global font size for better readability
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
except Exception as e:
    print(f"Error setting matplotlib configuration: {e}")
    
# Define global chart style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_stock_data(df, code, title=None, save_path=None, figsize=(14, 8), show=False):
    """
    Plot stock historical data chart
    
    Parameters:
        df (pandas.DataFrame): Stock data, must contain 'date', 'close', 'volume' columns
        code (str): Stock code
        title (str): Chart title, if not provided, a default title will be used
        save_path (str): Chart save path, if not provided, the chart will not be saved
        figsize (tuple): Chart size (width, height)
        show (bool): Whether to show the plot, default is False
    
    Returns:
        matplotlib.figure.Figure: Created chart object
    """
    try:
        if df is None or len(df) == 0:
            print("Error: Empty data provided")
            return None
            
        # Ensure date column is datetime type
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
        # Create two subplots, one for price and one for volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot close price in the top subplot
        ax1.plot(df['date'], df['close'], color='blue', linewidth=2, label='Close Price')
        
        # Add moving averages
        if len(df) >= 20:
            df['ma20'] = df['close'].rolling(window=20).mean()
            ax1.plot(df['date'], df['ma20'], color='red', linewidth=1, label='20-Day MA')
            
        if len(df) >= 60:
            df['ma60'] = df['close'].rolling(window=60).mean()
            ax1.plot(df['date'], df['ma60'], color='green', linewidth=1, label='60-Day MA')
            
        # Set title and labels
        if title is None:
            title = f"{code} Historical Price Chart"
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend(loc='best')
        
        # Plot volume in the bottom subplot
        if 'volume' in df.columns:
            ax2.bar(df['date'], df['volume'], color='gray', alpha=0.5, label='Volume')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            
            # Format volume display (in millions)
            def volume_formatter(x, pos):
                if x >= 1e8:
                    return f'{x/1e8:.1f}B'
                elif x >= 1e5:
                    return f'{x/1e5:.1f}M'
                else:
                    return f'{x:.0f}'
                    
            ax2.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
            
        # Set x-axis format
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)
        
        # If there are many data points, adjust x-axis label interval
        if len(df) > 30:
            interval = max(1, len(df) // 10)
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Stock chart saved to: {save_path}")
            
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting stock data: {e}")
        return None


def plot_prediction_results(df, predictions, future_days, code, title=None, save_path=None, figsize=(14, 10), show=False):
    """
    Plot stock prediction results chart
    
    Parameters:
        df (pandas.DataFrame): Original stock data, must contain 'date', 'close' columns
        predictions (numpy.ndarray): Prediction results array
        future_days (int): Number of future days to predict
        code (str): Stock code
        title (str): Chart title, if not provided, a default title will be used
        save_path (str): Chart save path, if not provided, the chart will not be saved
        figsize (tuple): Chart size (width, height)
        show (bool): Whether to show the plot, default is False
    
    Returns:
        matplotlib.figure.Figure: Created chart object
    """
    try:
        if df is None or len(df) == 0:
            print("Error: Empty data provided")
            return None
            
        if predictions is None or len(predictions) == 0:
            print("Error: Empty predictions")
            return None
            
        # Ensure date column is datetime type
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
        else:
            print("Warning: No date column in data, using index as date")
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df))
            
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual close price
        ax.plot(df['date'], df['close'], color='blue', linewidth=2, label='Actual Close Price')
        
        # Create prediction markers
        df['prediction'] = predictions
        
        # Mark up predictions
        up_pred = df[df['prediction'] == 1]
        if len(up_pred) > 0:
            ax.scatter(up_pred['date'], up_pred['close'], color='red', s=50, marker='^', label=f'Predicted Up ({len(up_pred)} days)')
            
        # Mark down predictions
        down_pred = df[df['prediction'] == 0]
        if len(down_pred) > 0:
            ax.scatter(down_pred['date'], down_pred['close'], color='green', s=50, marker='v', label=f'Predicted Down ({len(down_pred)} days)')
            
        # Add future prediction area
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        
        # Add future area shading
        ax.axvspan(last_date, future_dates[-1] if future_dates else last_date, alpha=0.2, color='yellow', label='Future Prediction Area')
        
        # Set chart title and labels
        if title is None:
            title = f"{code} {future_days}-Day Stock Price Prediction"
        plt.title(title, fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Set date format
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        if len(df) > 30:
            # If there are many data points, show a tick every certain number of days
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
        plt.xticks(rotation=45)
        
        # Add last prediction annotation
        last_pred = "Up" if predictions[-1] == 1 else "Down"
        last_price = df['close'].iloc[-1]
        plt.annotate(f'Latest Prediction: {last_pred}', 
                     xy=(df['date'].iloc[-1], last_price),
                     xytext=(df['date'].iloc[-1], last_price * 1.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center')
                     
        plt.tight_layout()
        
        # Save chart
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Prediction chart saved to: {save_path}")
            except Exception as e:
                print(f"Error saving prediction chart: {e}")
                
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting prediction results: {e}")
        return None


def plot_feature_importance(feature_importance, title=None, top_n=20, save_path=None, figsize=(10, 8), show=False):
    """
    Plot feature importance chart
    
    Parameters:
        feature_importance (list): Feature importance list, each element is (feature name, importance)
        title (str): Chart title, if not provided, a default title will be used
        top_n (int): Number of top features to show
        save_path (str): Chart save path, if not provided, the chart will not be saved
        figsize (tuple): Chart size (width, height)
        show (bool): Whether to show the plot, default is False
    
    Returns:
        matplotlib.figure.Figure: Created chart object
    """
    try:
        if feature_importance is None or len(feature_importance) == 0:
            print("Error: Empty feature importance data")
            return None
            
        # Limit feature number
        if len(feature_importance) > top_n:
            print(f"Showing top {top_n} features")
            feature_importance = feature_importance[:top_n]
            
        # Convert data format
        features, importances = zip(*feature_importance)
        
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x=list(importances), y=list(features), ax=ax, palette='viridis')
        
        # Set Y-axis labels (feature names)
        ax.set_yticklabels(features)
        
        # Set X-axis labels
        ax.set_xlabel('Importance (%)')
        
        # Set chart title
        if title is None:
            title = 'Feature Importance Ranking'
        ax.set_title(title, fontsize=14)
        
        # Set grid lines only on X-axis
        ax.grid(axis='x')
        
        # Format X-axis as percentage
        def percentage_formatter(x, pos):
            return f'{100 * x:.1f}%'
            
        ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        plt.tight_layout()
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Feature importance chart saved to: {save_path}")
            
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None, figsize=(8, 6), show=False):
    """
    Plot confusion matrix
    
    Parameters:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): Class names, default is ['Down', 'Up']
        title (str): Chart title
        save_path (str): Chart save path, if not provided, the chart will not be saved
        figsize (tuple): Chart size (width, height)
        show (bool): Whether to show the plot, default is False
    
    Returns:
        matplotlib.figure.Figure: Created chart object
    """
    try:
        if cm is None:
            print("Error: Empty confusion matrix")
            return None
            
        if class_names is None:
            class_names = ['Down', 'Up']
            
        # Calculate accuracy
        accuracy = np.trace(cm) / float(np.sum(cm))
        
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False,
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
                   
        # Set title and labels
        ax.set_title(f"{title}\nAccuracy: {accuracy:.2%}", fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
            
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return None


def plot_multi_model_comparison(model_results, metric='accuracy', title=None, save_path=None, figsize=(10, 6), show=False):
    """
    Plot multi-model performance comparison chart
    
    Parameters:
        model_results (dict): Model results dictionary, format is {model name: {metric name: metric value}}
        metric (str): Metric to compare, such as 'accuracy', 'f1', 'auc' etc.
        title (str): Chart title, default is "Model Performance Comparison"
        save_path (str): Chart save path, if not provided, the chart will not be saved
        figsize (tuple): Chart size (width, height)
        show (bool): Whether to show the plot, default is False
    
    Returns:
        matplotlib.figure.Figure: Created chart object
    """
    try:
        if not model_results:
            print("Error: Model results are empty")
            return None
            
        # Extract model names and metric values
        model_names = list(model_results.keys())
        metric_values = [results.get(metric, 0) for results in model_results.values()]
        
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, metric_values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
                   
        # Set Y-axis range and labels
        min_value = min(metric_values) * 0.9 if metric_values else 0
        max_value = max(metric_values) * 1.1 if metric_values else 1
        ax.set_ylim(min_value, max_value)
        
        # Set axis labels
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        
        # Set title
        if title is None:
            title = f'Model {metric} Performance Comparison'
        ax.set_title(title)
        
        # Set grid lines
        ax.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Model comparison chart saved to: {save_path}")
            except Exception as e:
                print(f"Error saving model comparison chart: {e}")
                
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting model comparison chart: {e}")
        return None
