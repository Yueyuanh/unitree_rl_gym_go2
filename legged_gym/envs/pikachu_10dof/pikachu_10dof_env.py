from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Pikachu10Robot(LeggedRobot):
    """皮卡丘机器人环境类,继承自LeggedRobot基类"""
    
    def _get_noise_scale_vec(self, cfg):
        """设置用于缩放观测噪声的向量。
           [注意]: 当更改观测结构时必须适配此方法

        参数:
            cfg (Dict): 环境配置文件

        返回值:
            [torch.Tensor]: 用于乘以[-1, 1]均匀分布的比例向量
        """
        # 创建与观测缓冲区第一个元素形状相同的零向量
        noise_vec = torch.zeros_like(self.obs_buf[0])
        
        # 从配置中获取噪声相关参数
        self.add_noise = self.cfg.noise.add_noise  # 是否添加噪声的标志
        noise_scales = self.cfg.noise.noise_scales  # 不同观测量的噪声比例
        noise_level = self.cfg.noise.noise_level    # 噪声等级
        
        # 设置角速度噪声（前3个维度）
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        
        # 设置重力向量噪声（第4-6个维度）
        noise_vec[3:6] = noise_scales.gravity * noise_level
        
        # 设置命令噪声（第7-9个维度），这里设为0表示不添加噪声
        noise_vec[6:9] = 0.
        
        # 设置关节位置噪声
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        
        # 设置关节速度噪声
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        
        # 设置前一步动作噪声，这里设为0
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.
        
        # 设置相位（正弦/余弦）噪声，这里设为0
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0.
        
        return noise_vec

    def _init_foot(self):
        """初始化足部相关状态和缓冲区"""
        # 获取足部数量（根据足部索引列表长度）
        self.feet_num = len(self.feet_indices)
        
        # 获取刚体状态张量
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # 将张量包装为torch张量
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        
        # 重塑为[num_envs, 刚体数, 13]的形状（13个刚体状态参数）
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        
        # 提取足部状态（特定索引的刚体）
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        
        # 提取足部位置（前3个维度：x, y, z）
        self.feet_pos = self.feet_state[:, :, :3]
        
        # 提取足部线速度（第8-10个维度：vx, vy, vz）
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        """初始化所有缓冲区，重写父类方法"""
        # 调用父类的缓冲区初始化方法
        super()._init_buffers()
        
        # 额外初始化足部相关缓冲区
        self._init_foot()

    def update_feet_state(self):
        """更新足部状态信息（位置、速度等）"""
        # 刷新刚体状态张量以获取最新仿真数据
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 更新足部状态视图（由于形状已定义，只需重新赋值）
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        
        # 更新足部位置
        self.feet_pos = self.feet_state[:, :, :3]
        
        # 更新足部速度
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        """物理仿真步骤后的回调函数，用于更新状态和计算奖励"""
        # 更新足部状态
        self.update_feet_state()

        # 设置步态周期参数
        period = 0.8      # 步态周期（秒）
        offset = 0.5      # 左右腿相位偏移（50%周期）
        
        # 计算当前相位（时间取模周期并归一化）
        self.phase = (self.episode_length_buf * self.dt) % period / period
        
        # 左腿相位（直接使用当前相位）
        self.phase_left = self.phase
        
        # 右腿相位（添加偏移后取模）
        self.phase_right = (self.phase + offset) % 1
        
        # 合并左右腿相位为形状[num_envs, 2]的张量
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        # 调用父类的回调函数
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """计算观测向量，包含机器人状态和命令信息"""
        # 计算相位的正弦和余弦值（用于周期性步态）
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)   # 正弦相位
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)   # 余弦相位
        
        # 构建普通观测向量（用于策略网络）
        self.obs_buf = torch.cat((
                                    self.base_ang_vel * self.obs_scales.ang_vel,           # 基座角速度（缩放后）
                                    self.projected_gravity,                                # 投影重力向量
                                    self.commands[:, :3] * self.commands_scale,            # 命令信号（缩放后）
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差（缩放后）
                                    self.dof_vel * self.obs_scales.dof_vel,                # 关节速度（缩放后）
                                    self.actions,                                          # 当前动作
                                    sin_phase,                                             # 正弦相位
                                    cos_phase                                              # 余弦相位
                                    ), dim=-1)
        
        # 构建特权观测向量（用于价值网络，包含更多信息）
        self.privileged_obs_buf = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel,           # 基座线速度（缩放后）
                                    self.base_ang_vel * self.obs_scales.ang_vel,           # 基座角速度（缩放后）
                                    self.projected_gravity,                                # 投影重力向量
                                    self.commands[:, :3] * self.commands_scale,            # 命令信号（缩放后）
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差（缩放后）
                                    self.dof_vel * self.obs_scales.dof_vel,                # 关节速度（缩放后）
                                    self.actions,                                          # 当前动作
                                    sin_phase,                                             # 正弦相位
                                    cos_phase                                              # 余弦相位
                                    ), dim=-1)
        
        # 如果需要添加噪声
        if self.add_noise:
            # 生成[-1, 1]均匀分布的噪声，按比例缩放后添加到观测
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _reward_contact(self):
        """接触奖励：奖励符合步态相位的足部接触"""
        # 初始化奖励为零向量
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 遍历每个足部
        for i in range(self.feet_num):
            # 判断是否处于支撑相（相位<0.55）
            is_stance = self.leg_phase[:, i] < 0.55
            
            # 判断足部是否接触地面（z方向接触力>1N）
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            
            # XOR逻辑：支撑相应接触，摆动相应不接触，符合条件则奖励
            res += ~(contact ^ is_stance)
        
        return res
    
    def _reward_feet_swing_height(self):
        """足部摆动高度奖励：奖励摆动相足部达到目标高度"""
        # 判断足部是否接触地面（接触力向量的模>1N）
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
        # 计算足部高度与目标高度（0.08米）的平方误差，仅对非接触状态计算
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        
        # 对每个环境的所有足部误差求和
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        """生存奖励：奖励机器人保持存活状态"""
        # 返回常数奖励值
        return 1.0
    
    def _reward_contact_no_vel(self):
        """无速度接触惩罚：惩罚接触但速度不为零的足部"""
        # 判断足部是否接触地面
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
        # 仅计算接触状态下足部的速度
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        
        # 计算速度平方作为惩罚项
        penalize = torch.square(contact_feet_vel[:, :, :3])
        
        # 对每个环境的所有足部、所有方向的速度惩罚求和
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        """髋关节位置奖励：惩罚髋关节偏离初始位置"""
        # 提取髋关节索引（[3,4,9,10]，计算位置平方和作为惩罚
        return torch.sum(torch.square(self.dof_pos[:, [1,2,6,7]]), dim=1)