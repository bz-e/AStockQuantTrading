#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文字体修复脚本 - 增强版
解决matplotlib中文显示为方框的问题
支持自动检测系统中的中文字体并配置matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager
import os
import subprocess
import tempfile
import sys
import shutil
from pathlib import Path

def clear_matplotlib_cache():
    """清除matplotlib字体缓存"""
    cache_dir = mpl.get_cachedir()
    cache_files = [
        "fontlist-v330.json",
        "fontlist-v320.json",
        "fontlist-v310.json", 
        "fontlist-v300.json",
        "tex.cache",
        "fonts.cache"
    ]
    
    cleared = False
    for cache_file in cache_files:
        file_path = os.path.join(cache_dir, cache_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已删除缓存文件: {file_path}")
                cleared = True
            except Exception as e:
                print(f"删除缓存文件失败 {file_path}: {e}")
    
    if cleared:
        print("已清除matplotlib字体缓存，下次运行时将重新扫描字体")
    else:
        print("未找到需要清除的matplotlib字体缓存文件")
    
    return cleared

def detect_chinese_fonts():
    """检测系统中安装的中文字体"""
    print("正在扫描系统中的中文字体...")
    
    # 常见的中文字体文件路径
    common_font_paths = [
        # macOS系统字体
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # Arial Unicode MS
        "/System/Library/Fonts/STHeiti Medium.ttc",  # Heiti TC
        "/System/Library/Fonts/PingFang.ttc",  # PingFang SC
        "/Library/Fonts/Arial Unicode.ttf", 
        "/System/Library/Fonts/Hiragino Sans GB.ttc",  # Hiragino Sans GB
        
        # 用户字体目录
        os.path.expanduser("~/Library/Fonts/Arial Unicode.ttf"),
        os.path.expanduser("~/Library/Fonts/NotoSansSC-Regular.otf"),
        
        # Microsoft字体
        "/Library/Fonts/Microsoft/Microsoft Sans Serif.ttf",
        "/Library/Fonts/Microsoft/SimSun.ttf",
        "/Library/Fonts/Microsoft/SimHei.ttf"
    ]
    
    # 检测直接可用的字体文件
    font_files_found = []
    for font_path in common_font_paths:
        if os.path.exists(font_path):
            font_files_found.append(font_path)
            print(f"找到字体文件: {font_path}")
    
    # 使用matplotlib的字体管理器获取中文字体
    chinese_font_families = []
    chinese_font_names = []
    
    for font in fontManager.ttflist:
        # 检查字体名称中是否包含中文相关关键词
        font_name = font.name.lower()
        if any(keyword in font_name for keyword in ['chinese', 'cjk', 'sc', 'tc', 'heiti', 'songti', 'kaiti', 'yahei', 'gothic', 'ming', 'noto', 'unicode']):
            chinese_font_families.append(font.name)
            print(f"找到中文字体: {font.name} (文件: {font.fname})")
            # 同时记录字体文件路径
            if font.fname not in font_files_found:
                font_files_found.append(font.fname)
    
    # 尝试运行fc-list命令获取系统字体列表（Unix/Linux/macOS）
    try:
        result = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            fc_fonts = result.stdout.strip().split('\n')
            for font_info in fc_fonts:
                if ':' in font_info:
                    font_path = font_info.split(':')[0].strip()
                    if font_path not in font_files_found:
                        font_files_found.append(font_path)
                        print(f"fc-list找到字体: {font_path}")
    except (FileNotFoundError, subprocess.SubprocessError):
        # fc-list命令可能不存在，跳过
        pass
    
    return {
        'font_files': font_files_found,
        'font_families': chinese_font_families
    }

def generate_matplotlibrc():
    """生成或更新matplotlibrc配置文件"""
    fonts = detect_chinese_fonts()
    
    if not fonts['font_families'] and not fonts['font_files']:
        print("未检测到任何中文字体，无法生成配置")
        return False
    
    # 使用字体家族或者第一个检测到的字体文件
    font_config = ""
    if fonts['font_families']:
        font_families = ", ".join(fonts['font_families'][:5])  # 限制数量，避免过长
        font_config = f"""
# 中文字体配置
font.family: sans-serif
font.sans-serif: {font_families}, sans-serif
axes.unicode_minus: False
"""
    else:
        # 尝试直接使用第一个字体文件
        font_path = fonts['font_files'][0]
        font_config = f"""
# 中文字体配置 - 直接指定字体文件
font.family: sans-serif
# 请在代码中使用以下字体文件路径:
# font_path = "{font_path}"
# font_prop = FontProperties(fname=font_path)
axes.unicode_minus: False
"""
    
    # 创建matplotlib配置目录
    mpl_config_dir = os.path.expanduser('~/.matplotlib')
    os.makedirs(mpl_config_dir, exist_ok=True)
    
    config_path = os.path.join(mpl_config_dir, 'matplotlibrc')
    
    # 备份现有配置
    if os.path.exists(config_path):
        backup_path = config_path + '.bak'
        try:
            shutil.copy2(config_path, backup_path)
            print(f"已备份原配置文件到: {backup_path}")
        except Exception as e:
            print(f"备份配置文件失败: {e}")
    
    # 写入新配置
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(font_config)
        print(f"已生成matplotlib配置文件: {config_path}")
        return True
    except Exception as e:
        print(f"写入配置文件失败: {e}")
        return False

def test_chinese_font():
    """测试中文字体显示"""
    fonts = detect_chinese_fonts()
    
    if not fonts['font_files']:
        print("未找到可用的中文字体文件，测试失败")
        return
    
    # 创建测试图形
    plt.figure(figsize=(12, len(fonts['font_files']) * 0.8))
    plt.title("中文字体测试")
    plt.axis('off')
    
    # 测试所有检测到的字体文件
    for i, font_path in enumerate(fonts['font_files']):
        try:
            font_prop = FontProperties(fname=font_path)
            plt.text(0.1, 0.9 - i*0.08, f"测试中文字体 - {os.path.basename(font_path)}", 
                    fontproperties=font_prop, fontsize=12)
        except Exception as e:
            print(f"测试字体 {font_path} 失败: {e}")
    
    # 保存测试图形
    test_path = 'chinese_font_test.png'
    plt.savefig(test_path)
    print(f"中文字体测试结果已保存至: {test_path}")
    
    # 尝试展示图形
    try:
        plt.show()
    except Exception:
        pass
    
    return True

def fix_enhanced_ml_prediction():
    """修复enhanced_ml_prediction.py中的字体问题"""
    fonts = detect_chinese_fonts()
    
    if not fonts['font_files']:
        print("未检测到中文字体，无法进行修复")
        return False
    
    # 找到enhanced_ml_prediction.py文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, 'enhanced_ml_prediction.py')
    
    if not os.path.exists(target_file):
        print(f"未找到目标文件: {target_file}")
        return False
    
    # 读取文件内容
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False
    
    # 更新全局字体设置
    font_families = ", ".join([f"'{family}'" for family in fonts['font_families'][:5]]) if fonts['font_families'] else "'Arial Unicode MS', 'Heiti TC', 'STHeiti'"
    global_font_config = f"plt.rcParams['font.sans-serif'] = [{font_families}]"
    
    # 修改全局字体设置
    if "plt.rcParams['font.sans-serif']" in content:
        content = content.replace("plt.rcParams['font.sans-serif'] = ['SimHei']", global_font_config)
    
    # 生成font_paths代码
    font_paths_code = "font_paths = [\n"
    for path in fonts['font_files'][:5]:  # 限制数量
        font_paths_code += f'    "{path}",  # {os.path.basename(path)}\n'
    font_paths_code += "]"
    
    # 更新文件内容
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已更新文件: {target_file}")
        
        print("\n推荐在visualize_prediction函数中使用以下代码：")
        print(font_paths_code)
        print("""
# 查找第一个存在的字体文件
font_path = None
for path in font_paths:
    if os.path.exists(path):
        font_path = path
        break
        
if font_path:
    font_prop = FontProperties(fname=font_path)
    use_english = False
    print(f"使用字体文件: {font_path}")
else:
    font_prop = None
    use_english = True
    print("警告: 未找到合适的中文字体文件，将使用英文标签")
""")
        return True
    except Exception as e:
        print(f"更新文件失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("中文字体问题修复工具 - 增强版")
    print("=" * 60)
    print("此工具可以帮助解决matplotlib中文显示为方框的问题")
    print()
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'clear-cache':
            clear_matplotlib_cache()
            return
        elif sys.argv[1] == 'test':
            test_chinese_font()
            return
        elif sys.argv[1] == 'config':
            generate_matplotlibrc()
            return
        elif sys.argv[1] == 'fix':
            fix_enhanced_ml_prediction()
            return
        elif sys.argv[1] == 'detect':
            detect_chinese_fonts()
            return
    
    # 默认操作：全面检测和修复
    print("正在检测系统字体...")
    fonts = detect_chinese_fonts()
    
    if not fonts['font_files'] and not fonts['font_families']:
        print("\n警告: 未检测到任何中文字体！")
        print("请考虑安装中文字体如：Arial Unicode MS、SimHei、PingFang SC或Noto Sans SC")
        return
    
    # 清除缓存
    clear_matplotlib_cache()
    
    # 生成配置
    generate_matplotlibrc()
    
    # 测试字体
    test_chinese_font()
    
    print("\n=" * 60)
    print("修复完成！请尝试重新运行您的程序")
    print("如果问题仍然存在，请尝试运行: python fix_chinese_font.py fix")
    print("=" * 60)

if __name__ == "__main__":
    main()
