#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文字体渲染测试脚本
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def test_chinese_display():
    """测试各种方式的中文显示效果"""
    
    # 方法1：使用rcParams全局配置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 
                                       'Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(15, 10))
    
    # 测试1：直接使用rcParams(全局字体设置)
    ax1 = fig.add_subplot(221)
    ax1.set_title('测试1: 全局字体设置', fontsize=14)
    ax1.set_xlabel('横轴 (X-Axis)')
    ax1.set_ylabel('纵轴 (Y-Axis)')
    
    x = np.linspace(0, 2*np.pi, 100)
    ax1.plot(x, np.sin(x), label='正弦曲线')
    ax1.plot(x, np.cos(x), label='余弦曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 测试2：使用FontProperties指定字体
    ax2 = fig.add_subplot(222)
    font_arial = FontProperties(family='Arial Unicode MS')
    ax2.set_title('测试2: 使用Arial Unicode MS', fontproperties=font_arial, fontsize=14)
    ax2.set_xlabel('横轴', fontproperties=font_arial)
    ax2.set_ylabel('纵轴', fontproperties=font_arial)
    
    labels = ['一月', '二月', '三月', '四月', '五月']
    values = [5, 7, 3, 4, 6]
    bars = ax2.bar(labels, values, color='blue', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontproperties=font_arial)
    
    # 测试3：使用Heiti TC字体
    ax3 = fig.add_subplot(223)
    font_heiti = FontProperties(family='Heiti TC')
    ax3.set_title('测试3: 使用Heiti TC', fontproperties=font_heiti, fontsize=14)
    ax3.set_xlabel('日期', fontproperties=font_heiti)
    ax3.set_ylabel('价格', fontproperties=font_heiti)
    
    dates = [f'2025-{i:02d}-01' for i in range(1, 13)]
    prices = np.cumsum(np.random.normal(0, 1, 12))
    ax3.plot(dates, prices, marker='o', label='股票价格走势')
    # 每隔两个点显示一个标签
    ax3.set_xticks(dates[::2])
    ax3.set_xticklabels(dates[::2], rotation=45, ha='right')
    ax3.legend(prop=font_heiti)
    
    # 测试4：直接在文本中使用中文
    ax4 = fig.add_subplot(224)
    # 不指定字体，使用全局配置
    ax4.set_title('测试4: 直接在文本中显示中文字符', fontsize=14)
    ax4.set_xlabel('X轴 (日期)')
    ax4.set_ylabel('Y轴 (数值)')
    
    x = np.arange(10)
    y = np.random.randint(1, 10, 10)
    ax4.scatter(x, y, s=100)
    
    # 直接添加带中文的文本
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax4.annotate(f'点{i+1}: ({xi},{yi})', (xi, yi), xytext=(5, 5), 
                     textcoords='offset points', fontsize=9)
    
    # 添加额外的文本标注
    ax4.annotate('这是一个重要的数据点', xy=(5, y[5]), xytext=(10, 30),
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout(pad=3)
    
    # 保存图片
    plt.savefig('chinese_font_test.png', dpi=120)
    print(f"测试图片已保存为: {os.path.abspath('chinese_font_test.png')}")
    
    # 显示图形
    plt.show()
    plt.close()

def list_all_fonts_with_chinese():
    """列出支持中文的字体"""
    print("\n系统中支持中文的字体:")
    
    all_fonts = mpl.font_manager.fontManager.ttflist
    chinese_chars = "中文字体测试"
    
    # 创建一个图形用于测试每种字体
    for font in all_fonts:
        try:
            # 尝试创建一个小图形来测试字体
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.text(0.5, 0.5, chinese_chars, fontproperties=FontProperties(fname=font.fname), fontsize=12)
            plt.close(fig)  # 仅测试，不显示
            
            # 如果没有引发异常，认为字体支持中文
            print(f" - {font.name} (文件: {font.fname})")
        except:
            continue
    
    print("\n推荐的中文字体:")
    for font_name in ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'STHeiti']:
        try:
            file_path = mpl.font_manager.findfont(FontProperties(family=font_name))
            print(f" - {font_name}: {file_path}")
        except Exception as e:
            print(f" - {font_name}: 未找到 ({e})")

def main():
    """主函数"""
    print("=" * 60)
    print("中文字体渲染测试脚本")
    print("=" * 60)
    
    # 列出当前matplotlib配置
    print(f"matplotlib版本: {mpl.__version__}")
    print(f"matplotlib配置目录: {mpl.get_configdir()}")
    print(f"matplotlib缓存目录: {mpl.get_cachedir()}")
    
    # 当前配置的字体
    print(f"\n当前配置的字体系列: {plt.rcParams['font.family']}")
    print(f"当前配置的Sans字体: {plt.rcParams['font.sans-serif']}")
    
    # 列出支持中文的字体
    list_all_fonts_with_chinese()
    
    # 测试中文显示
    print("\n开始测试中文显示...")
    test_chinese_display()
    
    print("\n测试完成！")
    print("如果图表中仍显示方框，您可以尝试以下方案:")
    print("1. 安装更多支持中文的字体，如Noto Sans CJK SC")
    print("2. 手动指定字体文件路径")
    print("3. 使用FontProperties对象并明确指定字体文件")

if __name__ == "__main__":
    main()
