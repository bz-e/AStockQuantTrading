"""
字体配置安装脚本
"""
import os
import sys
import shutil
import platform
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import urllib.request

def get_system_info():
    """获取系统信息"""
    print(f"Python版本: {sys.version}")
    print(f"matplotlib版本: {mpl.__version__}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['sw_vers'], capture_output=True, text=True)
            print(f"详细macOS信息:\n{result.stdout}")
    except Exception as e:
        print(f"获取详细系统信息失败: {e}")

def list_current_fonts():
    """列出当前系统中的中文字体"""
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [
        f for f in all_fonts if any(name in f for name in 
        ['Noto Sans CJK', 'Noto Sans SC', 'SimHei', 'SimSun', 'Microsoft YaHei', 
         'PingFang', 'Heiti', 'Arial Unicode MS', 'STHeiti', 'STSong', 'STFangsong'])
    ]
    
    print("\n当前系统中可能支持中文的字体:")
    for font in sorted(set(chinese_fonts)):  # 使用set去重
        print(f" - {font}")
    
    return chinese_fonts

def download_noto_sans_sc():
    """下载谷歌Noto Sans SC字体"""
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"
    font_path = os.path.expanduser("~/Library/Fonts/NotoSansSC-Regular.otf")
    
    if os.path.exists(font_path):
        print(f"字体文件已存在: {font_path}")
        return font_path
    
    print(f"正在从 {font_url} 下载Noto Sans SC字体...")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            urllib.request.urlretrieve(font_url, tmp_file.name)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(font_path), exist_ok=True)
            
            # 复制到字体目录
            shutil.copy2(tmp_file.name, font_path)
            os.unlink(tmp_file.name)
            
        print(f"字体已成功下载并安装到 {font_path}")
        return font_path
    except Exception as e:
        print(f"下载字体时出错: {e}")
        return None

def create_matplotlibrc():
    """创建或更新matplotlibrc配置文件"""
    config_dir = mpl.get_configdir()
    rc_path = os.path.join(config_dir, 'matplotlibrc')
    
    print(f"\nmatplotlib配置目录: {config_dir}")
    
    # 将要添加的配置行
    new_config_lines = [
        "# 中文字体配置",
        "font.family: sans-serif",
        "font.sans-serif: Arial Unicode MS, Noto Sans SC, PingFang SC, Heiti TC, Microsoft YaHei, SimHei, sans-serif",
        "axes.unicode_minus: False"
    ]
    
    # 如果文件已存在，保留原来的配置，只更新字体相关配置
    if os.path.exists(rc_path):
        with open(rc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 过滤掉已有的字体配置行
        filtered_lines = [line for line in lines if not any(
            line.strip().startswith(x) for x in ["font.family", "font.sans-serif", "axes.unicode_minus"])]
        
        # 合并配置
        all_lines = filtered_lines + ["\n"] + new_config_lines
        
        with open(rc_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_lines))
        
        print(f"已更新matplotlib配置文件: {rc_path}")
    else:
        # 创建新的配置文件
        os.makedirs(config_dir, exist_ok=True)
        with open(rc_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(new_config_lines))
        
        print(f"已创建新的matplotlib配置文件: {rc_path}")

def test_font_rendering(font_name):
    """测试特定字体的渲染效果"""
    try:
        # 尝试使用这个字体渲染中文
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f'测试 "{font_name}" 字体渲染中文文本', 
                fontfamily=font_name, fontsize=12, ha='center')
        ax.axis('off')
        
        # 保存到临时文件以检查渲染效果
        temp_file = os.path.join(tempfile.gettempdir(), f"font_test_{font_name.replace(' ', '_')}.png")
        plt.savefig(temp_file, dpi=100)
        plt.close(fig)
        
        print(f"已生成字体 '{font_name}' 的测试图像: {temp_file}")
        return True
    except Exception as e:
        print(f"测试字体 '{font_name}' 时出错: {e}")
        return False

def test_chinese_display():
    """测试中文显示效果"""
    plt.figure(figsize=(10, 6))
    
    # 基本中文文本
    plt.subplot(211)
    plt.title('中文显示测试', fontsize=14)
    plt.xlabel('横轴 (X-Axis)')
    plt.ylabel('纵轴 (Y-Axis)')
    
    x = np.linspace(0, 2*np.pi, 100)
    plt.plot(x, np.sin(x), label='正弦曲线')
    plt.plot(x, np.cos(x), label='余弦曲线')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 更多中文文本和符号
    plt.subplot(212)
    labels = ['股票', '基金', '债券', '期货', '期权']
    values = [25, 20, 15, 30, 10]
    
    plt.bar(labels, values, color=['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de'])
    plt.title('投资产品分布', fontsize=14)
    plt.xlabel('产品类型')
    plt.ylabel('占比 (%)')
    
    # 添加负号测试
    plt.figtext(0.5, 0.01, '负数显示测试: -123.45', ha='center', fontsize=12)
    
    plt.tight_layout(pad=3)
    plt.subplots_adjust(bottom=0.15)
    
    # 保存图片
    plt.savefig('font_test.png', dpi=100)
    print(f"测试图片已保存为: {os.path.abspath('font_test.png')}")
    
    # 显示图形
    plt.show()

def clear_matplotlib_cache():
    """清除matplotlib的字体缓存"""
    cache_dir = mpl.get_cachedir()
    print(f"matplotlib缓存目录: {cache_dir}")
    
    # 需要清除的缓存文件
    cache_files = [
        "fontlist-v330.json",
        "fontlist-v310.json", 
        "fontlist-v300.json",
        "tex.cache",
        "fonts.cache"
    ]
    
    for cache_file in cache_files:
        file_path = os.path.join(cache_dir, cache_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已删除缓存文件: {file_path}")
            except Exception as e:
                print(f"删除缓存文件失败 {file_path}: {e}")
    
    print("matplotlib字体缓存已清除，需要重新启动Python来重建字体缓存。")

def test_all_available_fonts():
    """测试所有可能的中文字体"""
    chinese_fonts = [
        'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Microsoft YaHei', 
        'SimHei', 'STHeiti', 'SimSun', 'STFangsong', 'Noto Sans SC'
    ]
    
    print("\n正在测试所有可能的中文字体渲染...")
    
    fonts_tested = 0
    for font_name in chinese_fonts:
        # 检查字体是否存在于系统中
        if any(font_name.lower() in font.name.lower() for font in fm.fontManager.ttflist):
            if test_font_rendering(font_name):
                fonts_tested += 1
    
    if fonts_tested == 0:
        print("警告: 未能找到任何可用的中文字体!")
        # 尝试下载Noto Sans SC
        download_noto_sans_sc()
    else:
        print(f"成功测试了 {fonts_tested} 种中文字体")

def main():
    """主函数"""
    print("=" * 50)
    print("matplotlib中文字体配置工具")
    print("=" * 50)
    
    # 获取系统信息
    get_system_info()
    
    # 列出当前字体
    current_chinese_fonts = list_current_fonts()
    
    # 如果没有中文字体，尝试安装
    if not current_chinese_fonts:
        print("\n未检测到中文字体，将尝试下载安装Google Noto Sans SC字体...")
        download_noto_sans_sc()
    
    # 创建或更新matplotlib配置
    create_matplotlibrc()
    
    # 清除字体缓存
    clear_matplotlib_cache()
    
    # 测试所有可能的字体
    test_all_available_fonts()
    
    # 重新列出字体
    print("\n配置后可用的中文字体:")
    list_current_fonts()
    
    # 测试中文显示
    print("\n现在将测试中文显示效果...")
    test_chinese_display()
    
    print("\n配置完成！请重启Python或Jupyter Notebook以使配置生效。")
    print("如果图表中仍显示方框，请尝试以下步骤:")
    print("1. 确保系统安装了中文字体")
    print("2. 重启Python或Jupyter Notebook环境")
    print("3. 尝试修改matplotlib的字体配置")

if __name__ == "__main__":
    main()
