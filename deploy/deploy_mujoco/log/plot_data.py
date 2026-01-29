import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Optional
import os


def plot_csv_data(csv_file: str, data_columns: List[str], output_file: Optional[str] = None):
    """
    从CSV文件中读取指定列的数据并绘图，自动分成两列显示
    
    Args:
        csv_file: CSV文件路径
        data_columns: 要绘制的数据列名称列表
        output_file: 输出图像文件路径，如果为None则直接显示图像
    """
    # 读取CSV数据
    timestamps = []
    data_dict = {col: [] for col in data_columns}
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.append(row['timestamp'])
            for col in data_columns:
                if col in row:
                    try:
                        data_dict[col].append(float(row[col]))
                    except ValueError:
                        # 如果数据不是数字，跳过
                        data_dict[col].append(0.0)
                else:
                    # 如果列不存在，添加0.0
                    data_dict[col].append(0.0)
    
    # 将时间戳转换为数字索引以便绘图
    time_indices = list(range(len(timestamps)))
    
    # 计算子图布局 (分两列)
    n_columns = 2
    n_rows = int(np.ceil(len(data_columns) / n_columns))
    
    if n_rows == 0:
        print("没有指定要绘制的数据列")
        return
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 5 * n_rows))
    
    # 如果只有一个子图，将其转换为数组格式
    if len(data_columns) == 1:
        axes = [axes]
    elif n_rows == 1 and len(data_columns) < 2:
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else list(axes)
    elif len(data_columns) == 1:
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else [axes] if np.isscalar(axes) else [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    # 绘制每个数据列
    for i, col_name in enumerate(data_columns):
        ax = axes[i]
        ax.plot(time_indices, data_dict[col_name], label=col_name, linewidth=1.0)
        ax.set_title(f'{col_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(col_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='可视化CSV日志数据')
    parser.add_argument('csv_file', type=str, help='CSV文件路径')
    parser.add_argument('--columns', '-c', type=str, nargs='+', required=True, 
                        help='要绘制的数据列名称列表，用空格分隔')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='输出图像文件路径 (可选)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件不存在 - {args.csv_file}")
        return
    
    plot_csv_data(args.csv_file, args.columns, args.output)


if __name__ == "__main__":
    main()