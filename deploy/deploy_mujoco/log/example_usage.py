"""
日志记录和可视化示例程序
演示如何使用DataLogger记录数据和使用plot_data.py可视化数据
"""

import sys
import os
import time
import numpy as np

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_logger import DataLogger
from plot_data import plot_csv_data


def example_usage():
    """
    示例：如何使用DataLogger记录数据并使用plot_data可视化
    """
    print("开始演示日志记录和可视化...")
    
    # 1. 创建数据记录器
    logger = DataLogger(log_dir="./log", file_name="example_log.csv")
    print(f"创建日志文件: {logger.get_log_path()}")
    
    # 2. 模拟记录一些数据
    print("开始记录模拟数据...")
    for i in range(100):
        # 模拟一些数据，例如机器人位置、速度等
        data = {
            "position_x": np.sin(i * 0.1) * 1.0,
            "position_y": np.cos(i * 0.1) * 1.0,
            "velocity_x": np.cos(i * 0.1) * 0.1,
            "velocity_y": -np.sin(i * 0.1) * 0.1,
            "angle": i * 0.05,
            "angular_velocity": 0.05
        }
        
        logger.log_data(data)
        
        if i % 20 == 0:
            print(f"已记录 {i+1}/100 条数据")
        
        time.sleep(0.01)  # 模拟时间间隔
    
    print(f"数据记录完成，文件保存在: {logger.get_log_path()}")
    
    # 3. 可视化数据
    csv_file = logger.get_log_path()
    columns_to_plot = ["position_x", "position_y", "velocity_x", "velocity_y", "angle"]
    
    print(f"\n开始可视化数据...")
    print(f"CSV文件: {csv_file}")
    print(f"要绘制的列: {columns_to_plot}")
    
    # 使用plot_data函数绘制数据
    plot_csv_data(csv_file, columns_to_plot)
    
    print("示例演示完成！")


def command_line_example():
    """
    演示如何从命令行使用plot_data脚本
    """
    example_usage()


def advanced_example():
    """
    高级示例：展示更多功能
    """
    print("\n高级示例：批量记录数据")
    
    # 创建新的记录器
    logger = DataLogger(log_dir="./log", file_name="advanced_example.csv")
    
    # 批量记录数据
    data_batch = []
    for i in range(50):
        data = {
            "torque_1": np.sin(i * 0.2) * 10,
            "torque_2": np.cos(i * 0.2) * 10,
            "torque_3": np.sin(i * 0.3) * 5,
            "current_1": np.abs(np.sin(i * 0.2) * 2),
            "current_2": np.abs(np.cos(i * 0.2) * 2)
        }
        data_batch.append(data)
    
    logger.log_multiple(data_batch)
    print(f"批量记录了 {len(data_batch)} 条数据到: {logger.get_log_path()}")
    
    # 可视化批量记录的数据
    print("可视化批量记录的数据...")
    plot_csv_data(logger.get_log_path(), ["torque_1", "torque_2", "torque_3", "current_1", "current_2"])


def batch_list_example():
    """
    批量记录列表数据示例：演示如何记录tau[12]这样的列表
    """
    print("\n批量记录列表数据示例：tau[12]")
    
    # 创建新的记录器
    logger = DataLogger(log_dir="./log", file_name="tau_list_example.csv")
    
    # 模拟记录tau[12]数据
    for step in range(30):
        # 生成模拟的tau数据 (12个关节的力矩)
        tau_values = [np.sin(step * 0.1 + i * 0.2) * (10 - i * 0.5) for i in range(12)]
        
        # 将列表数据转换为字典格式，方便记录
        tau_dict = {}
        for i, value in enumerate(tau_values):
            tau_dict[f"tau_{i}"] = value  # tau_0, tau_1, ..., tau_11
        
        # 还可以记录其他相关数据
        tau_dict["step"] = step
        tau_dict["total_torque"] = sum(abs(v) for v in tau_values)
        
        logger.log_data(tau_dict)
        
        if step % 10 == 0:
            print(f"已记录tau数据第 {step+1} 步")
    
    print(f"tau[12]列表数据记录完成，文件保存在: {logger.get_log_path()}")
    
    # 可视化tau数据的几个示例
    print("可视化tau数据...")
    tau_columns = [f"tau_{i}" for i in range(6)]  # 只显示前6个tau值，避免图形过于拥挤
    plot_csv_data(logger.get_log_path(), tau_columns + ["total_torque"])


def batch_list_by_function_example():
    """
    使用辅助函数批量记录列表数据示例
    """
    print("\n使用辅助函数批量记录列表数据示例")
    
    # 创建新的记录器
    logger = DataLogger(log_dir="./log", file_name="tau_function_example.csv")
    
    def log_array_data(logger_instance, array_data, prefix="data", step_info=None):
        """
        辅助函数：记录数组/列表数据
        
        Args:
            logger_instance: DataLogger实例
            array_data: 要记录的数组或列表
            prefix: 数据名称前缀
            step_info: 额外的步进信息（可选）
        """
        data_dict = {}
        
        # 将数组数据转换为字典格式
        for i, value in enumerate(array_data):
            data_dict[f"{prefix}_{i}"] = value
        
        # 添加额外的步进信息
        if step_info:
            data_dict.update(step_info)
        
        logger_instance.log_data(data_dict)
    
    # 模拟记录数据
    for step in range(20):
        # 生成模拟的关节角度数据 (12个关节)
        joint_angles = [np.cos(step * 0.15 + i * 0.1) * (1.0 - i * 0.05) for i in range(12)]
        
        # 使用辅助函数记录数据
        step_info = {
            "step": step,
            "timestamp_info": f"step_{step}"
        }
        log_array_data(logger, joint_angles, prefix="joint_angle", step_info=step_info)
        
        if step % 10 == 0:
            print(f"已记录关节角度数据第 {step+1} 步")
    
    print(f"关节角度数组数据记录完成，文件保存在: {logger.get_log_path()}")
    
    # 可视化关节角度数据
    print("可视化关节角度数据...")
    angle_columns = [f"joint_angle_{i}" for i in range(8)]  # 显示前8个关节角度
    plot_csv_data(logger.get_log_path(), angle_columns)


if __name__ == "__main__":
    example_usage()
    advanced_example()
    batch_list_example()
    batch_list_by_function_example()