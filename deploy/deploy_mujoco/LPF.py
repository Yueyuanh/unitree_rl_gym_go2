import numpy as np

class LPF:
    """最简单的低通滤波器"""
    def __init__(self, alpha=0.2, n_dof=12):
        """
        参数:
            alpha: 滤波系数 (0-1)，越小滤波越强，越大响应越快
            n_dof: 关节数量
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.filtered_value = np.zeros(n_dof)
        
    def update(self, raw_value):
        """
        更新滤波器：y = alpha * x + (1-alpha) * y_prev
        """
        # 一阶低通滤波
        self.filtered_value = self.alpha * raw_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
    def reset(self, value=None):
        """重置滤波器状态"""
        if value is not None:
            self.filtered_value = value.copy()
        else:
            self.filtered_value = np.zeros_like(self.filtered_value)