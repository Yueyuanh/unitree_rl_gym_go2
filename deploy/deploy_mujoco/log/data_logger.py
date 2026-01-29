import os
import csv
import time
from datetime import datetime
from typing import Dict, List, Any


class DataLogger:
    """
    数据记录器，支持将自定义数据名称和数据记录为CSV格式
    """
    def __init__(self, log_dir: str = "./log", file_name: str = None):
        """
        初始化数据记录器

        Args:
            log_dir: 日志保存目录
            file_name: 日志文件名，如果不提供则自动生成带时间戳的文件名
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_name = f"log_{timestamp}.csv"
        else:
            self.file_name = file_name
            
        self.file_path = os.path.join(log_dir, self.file_name)
        self.fieldnames = ["timestamp"]
        self.data_buffer = []
        self.initialized = False

    def add_fieldnames(self, fieldnames: List[str]):
        """
        添加字段名称
        
        Args:
            fieldnames: 字段名称列表
        """
        for field in fieldnames:
            if field not in self.fieldnames:
                self.fieldnames.append(field)
    
    def log_data(self, data: Dict[str, Any]):
        """
        记录数据
        
        Args:
            data: 包含数据的字典，键为数据名称，值为数据值
        """
        if not self.initialized:
            # 首次记录时初始化CSV文件
            current_time = time.time()
            timestamp_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')
            log_entry = {"timestamp": timestamp_str}
            log_entry.update(data)
            
            # 更新字段名称
            for key in data.keys():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
            
            # 创建CSV文件并写入表头
            with open(self.file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(log_entry)
            
            self.initialized = True
        else:
            # 非首次记录，直接追加数据
            current_time = time.time()
            timestamp_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')
            log_entry = {"timestamp": timestamp_str}
            log_entry.update(data)
            
            # 添加新字段（如果有的话）
            for key in data.keys():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
            
            with open(self.file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(log_entry)

    def log_multiple(self, data_list: List[Dict[str, Any]]):
        """
        批量记录数据
        
        Args:
            data_list: 包含多个数据字典的列表
        """
        if not data_list:
            return
            
        # 检查所有字段名
        all_fieldnames = set(self.fieldnames)
        for data in data_list:
            all_fieldnames.update(data.keys())
        
        # 更新字段名
        for field in all_fieldnames:
            if field not in self.fieldnames:
                self.fieldnames.append(field)
        
        # 写入数据
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            
            for data in data_list:
                current_time = time.time()
                timestamp_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')
                log_entry = {"timestamp": timestamp_str}
                log_entry.update(data)
                writer.writerow(log_entry)
        
        if not self.initialized:
            self.initialized = True

    def flush(self):
        """
        强制刷新缓冲区，确保数据写入文件
        """
        pass  # 目前直接写入文件，不需要额外的缓冲区操作

    def get_log_path(self):
        """
        返回当前日志文件路径
        """
        return self.file_path


if __name__ == "__main__":
    # 示例使用
    logger = DataLogger()
    
    # 记录一些示例数据
    for i in range(10):
        data = {
            "position_x": i * 0.1,
            "position_y": i * 0.2,
            "velocity": i * 0.05,
            "angle": i * 0.3
        }
        logger.log_data(data)
        time.sleep(0.1)  # 模拟时间间隔
    
    print(f"数据已记录到: {logger.get_log_path()}")