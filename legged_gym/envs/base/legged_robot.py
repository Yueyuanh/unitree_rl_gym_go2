from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

# 继承训练任务 四足机器人
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ 解析提供的配置文件，
            调用create_sim()（创建仿真和环境），
            初始化训练期间使用的PyTorch缓冲区

        参数:
            cfg (Dict): 环境配置文件
            sim_params (gymapi.SimParams): 仿真参数
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX（必须是PhysX）
            device_type (string): 'cuda' 或 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): 如果为True则无渲染运行
        """
        self.cfg = cfg  # 配置文件
        self.sim_params = sim_params  # 仿真参数
        # 地形高度采样点
        self.height_samples = None
        # TODO:debug 可视化
        self.debug_viz = False  # 调试可视化标志
        self.init_done = False  # 初始化完成标志
        
        # 解析参数
        self._parse_cfg(self.cfg)
        # 父类初始化（BaseTask）
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 如果不是无头模式，设置相机位置
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        # 初始化缓存空间
        self._init_buffers()
        # 准备奖励函数，规则配置，计算奖励总和
        self._prepare_reward_function()
        self.init_done = True  # 标记初始化完成

    # 四足机器人仿真步进
    def step(self, actions):
        """ 应用动作，进行仿真，调用self.post_physics_step()

        参数:
            actions (torch.Tensor): 形状为(num_envs, num_actions_per_env)的张量

        返回值:
            (obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras): 观测、特权观测、奖励、重置标志、额外信息
        """
        
        # 动作空间归一化参数(100)
        clip_actions = self.cfg.normalization.clip_actions
        # 归一化并转为tensor向量，归一化范围[-100,100]
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # 进行物理仿真并渲染每一帧
        self.render()
        
        # 策略步，根据控制解算率（decimation）决定执行次数
        for _ in range(self.cfg.control.decimation):
            # 计算力矩输出 <- PD控制器
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 发送力矩到仿真
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)  # 执行物理仿真
            
            # 如果是测试模式，进行时间同步
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)  # 已过时间
                sim_time = self.gym.get_sim_time(self.sim)  # 仿真时间
                # 延时补偿，等待仿真渲染
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            
            # 如果是CPU模式，获取仿真结果
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            
            # 刷新关节状态张量
            self.gym.refresh_dof_state_tensor(self.sim)

        # 执行物理仿真后的处理
        self.post_physics_step()

        # 返回裁剪后的观测、裁剪后的状态（None）、奖励、终止标志和额外信息
        # 归一化观测器
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        # 如果有特权观测，也进行裁剪
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ 检查终止条件，计算观测和奖励
            调用self._post_physics_step_callback()进行通用计算
            如果需要，调用self._draw_debug_vis()绘制调试可视化
        """
        # 刷新演员根状态张量和接触力张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 更新步数计数器
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 准备状态量
        self.base_pos[:] = self.root_states[:, 0:3]  # 基座位置
        self.base_quat[:] = self.root_states[:, 3:7]  # 基座四元数
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])  # 转换为欧拉角（滚转、俯仰、偏航）
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 基座线速度（本体坐标系）
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 基座角速度（本体坐标系）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 投影重力（本体坐标系）

        # 调用物理仿真后的回调函数
        self._post_physics_step_callback()

        # 计算观测、奖励、重置等...
        self.check_termination()  # 检查终止条件
        self.compute_reward()  # 计算奖励
        
        # 获取需要重置的环境ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)  # 重置指定环境
        
        # 如果配置了随机推动机器人
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        # 计算观测（在某些情况下可能需要仿真步骤来刷新某些观测，例如身体位置）
        self.compute_observations()

        # 保存当前动作用于下一帧计算
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ 检查环境是否需要重置 """
        # 1. 检查终止接触点的接触力是否超过阈值
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        
        # 2. 检查俯仰角（pitch）是否超过1.0弧度（约57度）或滚转角（roll）是否超过0.8弧度（约46度）
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1]) > 1.0, torch.abs(self.rpy[:,0]) > 0.8)
        
        # 3. 检查是否超时（达到最大回合长度）
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf  # 超时也会触发重置

    def reset_idx(self, env_ids):
        """ 重置某些环境
            调用self._reset_dofs(env_ids), self._reset_root_states(env_ids), 和self._resample_commands(env_ids)
            [可选] 调用self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids)
            记录回合信息
            重置某些缓冲区

        参数:
            env_ids (list[int]): 需要重置的环境ID列表
        """
        if len(env_ids) == 0:
            return
        
        # 重置机器人状态
        self._reset_dofs(env_ids)  # 重置关节状态
        self._reset_root_states(env_ids)  # 重置根状态

        self._resample_commands(env_ids)  # 重新采样命令

        # 重置缓冲区
        self.actions[env_ids] = 0.  # 动作归零
        self.last_actions[env_ids] = 0.  # 上次动作归零
        self.last_dof_vel[env_ids] = 0.  # 上次关节速度归零
        self.feet_air_time[env_ids] = 0.  # 足部空中时间归零
        self.episode_length_buf[env_ids] = 0  # 回合长度计数器归零
        self.reset_buf[env_ids] = 1  # 设置重置标志
        
        # 填充额外信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # 计算平均奖励并存储到额外信息中
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.  # 重置回合奖励总和
        
        # 如果使用命令课程学习，记录最大命令范围
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        
        # 如果配置了发送超时信息给算法，添加超时信息
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ 计算奖励
            调用每个具有非零缩放因子的奖励函数（在self._prepare_reward_function()中处理）
            将每个项添加到回合总和和总奖励中
        """
        self.rew_buf[:] = 0.  # 重置奖励缓冲区
        
        # 遍历所有奖励函数
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]  # 奖励函数名称
            # 计算奖励值并乘以缩放因子
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew  # 累加到总奖励
            self.episode_sums[name] += rew  # 累加到回合奖励总和
        
        # 如果只允许正奖励，裁剪负奖励
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # 裁剪后添加终止奖励
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ 计算观测向量 """
        self.obs_buf = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel,  # 基座线速度（缩放后）
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # 基座角速度（缩放后）
                                    self.projected_gravity,  # 投影重力
                                    self.commands[:, :3] * self.commands_scale,  # 命令（缩放后）
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差（缩放后）
                                    self.dof_vel * self.obs_scales.dof_vel,  # 关节速度（缩放后）
                                    self.actions  # 当前动作
                                    ), dim=-1)
        
        # 如果不是盲模式，添加感知输入
        # 如果需要，添加噪声
        if self.add_noise:
            # 生成[-1, 1]的均匀分布噪声并缩放
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ 创建仿真、地形和环境 """
        self.up_axis_idx = 2  # 2表示z轴向上，1表示y轴向上 -> 相应地调整重力
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()  # 创建地平面
        self._create_envs()  # 创建环境

    def set_camera(self, position, lookat):
        """ 设置相机位置和方向
        
        参数:
            position: 相机位置 [x, y, z]
            lookat: 相机看向的点 [x, y, z]
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- 回调函数 --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ 回调：允许存储/更改/随机化每个环境的刚体形状属性
            在环境创建期间调用
            基本行为：随机化每个环境的摩擦系数

        参数:
            props (List[gymapi.RigidShapeProperties]): 资产每个形状的属性
            env_id (int): 环境ID

        返回值:
            [List[gymapi.RigidShapeProperties]]: 修改后的刚体形状属性
        """
        # 如果配置了随机化摩擦系数
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # 准备摩擦系数随机化
                friction_range = self.cfg.domain_rand.friction_range  # 摩擦系数范围
                num_buckets = 64  # 分桶数量
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))  # 为每个环境随机分配桶ID
                # 为每个桶生成随机摩擦系数
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]  # 存储摩擦系数

            # 为每个形状设置摩擦系数
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ 回调：允许存储/更改/随机化每个环境的DOF属性
            在环境创建期间调用
            基本行为：存储URDF中定义的位置、速度和力矩限制

        参数:
            props (numpy.array): 资产每个DOF的属性
            env_id (int): 环境ID

        返回值:
            [numpy.array]: 修改后的DOF属性
        """
        if env_id == 0:
            # 初始化关节位置、速度和力矩限制张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            
            # 遍历所有关节属性
            for i in range(len(props)):
                # 从URDF中读取限制值
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 下限
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 上限
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 速度限制
                self.torque_limits[i] = props["effort"][i].item()  # 力矩限制
                # 应用软限制（soft limits）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 中间值
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 范围
                # 根据配置的收缩系数调整限制范围
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        return props

    def _process_rigid_body_props(self, props, env_id):
        """ 回调：处理刚体属性，用于随机化基座质量等
        
        参数:
            props: 刚体属性列表
            env_id: 环境ID
            
        返回值:
            修改后的刚体属性
        """
        # 如果配置了随机化基座质量
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range  # 质量添加范围
            # 在基座质量上添加随机值
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ 回调：在计算终止条件、奖励和观测之前调用
            默认行为：基于目标和航向计算角速度命令，计算测量的地形高度，随机推动机器人
        """
        # 在命令重新采样时间点重新采样命令
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        # 如果使用航向命令，计算角速度命令
        if self.cfg.commands.heading_command:
            # 计算机器人前进方向
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算当前航向角
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 计算目标航向与当前航向的差值，并限制在[-1, 1]范围内
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        """ 随机选择某些环境的命令

        参数:
            env_ids (List[int]): 需要新命令的环境ID列表
        """
        # 重新采样x方向线速度命令
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 重新采样y方向线速度命令
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 根据是否使用航向命令选择不同的采样方式
        if self.cfg.commands.heading_command:
            # 采样航向角命令
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # 采样偏航角速度命令
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 将小命令设为零（避免机器人因微小命令而抖动）
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ 从动作计算力矩
            动作可以解释为给PD控制器的位置或速度目标，或直接作为缩放后的力矩
            [注意]：力矩必须与DOF数量具有相同的维度，即使某些DOF未被驱动

        参数:
            actions (torch.Tensor): 动作

        返回值:
            [torch.Tensor]: 发送到仿真的力矩
        """
        # PD控制器
        actions_scaled = actions * self.cfg.control.action_scale  # 缩放动作
        control_type = self.cfg.control.control_type  # 控制类型
        
        # 根据控制类型计算力矩
        if control_type == "P":
            # 位置控制：P增益*(目标位置-当前位置) - D增益*速度
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            # 速度控制：P增益*(目标速度-当前速度) - D增益*加速度
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            # 力矩控制：直接使用动作作为力矩
            torques = actions_scaled
        else:
            raise NameError(f"未知控制器类型: {control_type}")
        
        # 将力矩限制在允许范围内
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ 重置选定环境的DOF位置和速度
            位置在0.5:1.5 x默认位置范围内随机选择
            速度设为零

        参数:
            env_ids (List[int]): 环境ID列表
        """
        # 在默认位置附近随机生成关节位置
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # 关节速度设为零
        self.dof_vel[env_ids] = 0.

        # 将环境ID转换为int32类型
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 设置选定环境的关节状态
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ 重置选定环境的ROOT状态位置和速度
            根据课程设置基座位置
            在-0.5:0.5 [m/s, rad/s]范围内选择随机化的基座速度

        参数:
            env_ids (List[int]): 环境ID列表
        """
        # 基座位置
        if self.custom_origins:
            # 如果有自定义原点，使用自定义原点
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # 在中心1米范围内随机添加xy位置偏移
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            # 使用标准原点
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # 基座速度：线速度和角速度在[-0.5, 0.5]范围内随机
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        
        # 将环境ID转换为int32类型
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 设置选定环境的根状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ 随机推动机器人。通过设置随机化的基座速度来模拟冲量 """
        env_ids = torch.arange(self.num_envs, device=self.device)
        # 根据推动间隔选择需要推动的环境
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        
        if len(push_env_ids) == 0:
            return
        
        # 设置xy方向的线速度在[-max_vel, max_vel]范围内随机
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        
        # 更新选定环境的根状态
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_command_curriculum(self, env_ids):
        """ 实现命令增加的课程

        参数:
            env_ids (List[int]): 正在重置的环境ID
        """
        # 如果跟踪奖励超过最大值的80%，增加命令范围
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # 减小下限，增加上限，但限制在最大课程范围内
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self, cfg):
        """ 设置用于缩放添加到观测中的噪声的向量
            [注意]：更改观测结构时必须适配此方法

        参数:
            cfg (Dict): 环境配置文件

        返回值:
            [torch.Tensor]: 用于乘以[-1, 1]均匀分布的比例向量
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])  # 创建与观测缓冲区第一个元素形状相同的零向量
        self.add_noise = self.cfg.noise.add_noise  # 是否添加噪声的标志
        noise_scales = self.cfg.noise.noise_scales  # 噪声缩放因子
        noise_level = self.cfg.noise.noise_level  # 噪声水平
        
        # 设置不同观测量的噪声缩放
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # 线速度噪声
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 角速度噪声
        noise_vec[6:9] = noise_scales.gravity * noise_level  # 重力噪声
        noise_vec[9:12] = 0.  # 命令噪声（设为0，不添加噪声）
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 关节位置噪声
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 关节速度噪声
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0.  # 先前动作噪声（设为0）

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ 初始化将包含仿真状态和处理量的torch张量 """
        # 获取gym GPU状态张量
        # 建立底层连接（acquire 获取指针）
        # 获取所有机器人根节点的状态，并返回一个GPU张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取所有关节的状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取所有刚体的接触力张量
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # 同步更新数据
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 为不同的切片创建包装张量
        # 装饰为tensors，将底层的仿真数据指针包装成PyTorch的tensor
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 根状态
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 关节状态
        
        # 切片赋值
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 关节位置
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 关节速度
        self.base_quat = self.root_states[:, 3:7]  # 基座四元数
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)  # 基座欧拉角
        self.base_pos = self.root_states[:self.num_envs, 0:3]  # 基座位置
        # 接触力：形状为(num_envs, num_bodies, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # 初始化稍后使用的数据
        self.common_step_counter = 0  # 通用步数计数器
        self.extras = {}  # 额外信息字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # 噪声缩放向量
        
        # 初始化向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        # 初始化参数张量
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # 命令张量：x速度, y速度, 偏航速度, 航向
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        # 命令缩放因子
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        # 计算基座状态（本体坐标系）
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      
        # 关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]  # 关节名称
            angle = self.cfg.init_state.default_joint_angles[name]  # 默认关节角度
            self.default_dof_pos[i] = angle  # 设置默认位置
            
            found = False
            # 从配置中查找关节的PD增益
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # P增益
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # D增益
                    found = True
                    break
            
            # 如果未找到增益配置，设为0并打印警告
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"关节 {name} 的PD增益未定义，将其设为零")
        
        # 将默认关节位置扩展为适合批量处理的形状
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ 准备奖励函数列表，将调用这些函数来计算总奖励
            查找self._reward_<REWARD_NAME>，其中<REWARD_NAME>是cfg中所有非零奖励缩放的名称
        """
        # 移除零缩放 + 将非零缩放乘以dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            # 去除0奖励缩放
            if scale == 0:
                self.reward_scales.pop(key) 
            else:
                # 将奖励缩放乘以时间步长
                self.reward_scales[key] *= self.dt
        
        # 准备函数列表
        self.reward_functions = []  # 奖励函数列表
        self.reward_names = []  # 奖励名称列表
        
        for name, scale in self.reward_scales.items():
            if name == "termination":  # 跳过终止奖励
                continue
            
            self.reward_names.append(name)
            name = '_reward_' + name  # 构建函数名
            # 获取叫这个名字的函数，并加入到列表中，方便后面统一调用
            self.reward_functions.append(getattr(self, name))

        # 初始化每集奖励字典：{_reward_name (key) : torch(num_envs) (value)}
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ 向仿真中添加地平面，根据cfg设置摩擦和恢复系数 """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 法向量（z轴向上）
        plane_params.static_friction = self.cfg.terrain.static_friction  # 静摩擦系数
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 动摩擦系数
        plane_params.restitution = self.cfg.terrain.restitution  # 恢复系数
        self.gym.add_ground(self.sim, plane_params)  # 添加地平面

    def _create_envs(self):
        """ 创建环境：
             1. 加载机器人URDF/MJCF资产
             2. 对于每个环境
                2.1 创建环境
                2.2 调用DOF和刚体形状属性回调
                2.3 使用这些属性创建演员并添加到环境中
             3. 存储机器人不同身体的索引
        """
        # 构建资产路径
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)  # 资产根目录
        asset_file = os.path.basename(asset_path)  # 资产文件名

        # 设置资产选项
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode  # 默认DOF驱动模式
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints  # 折叠固定关节
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  # 用胶囊体替换圆柱体
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # 翻转视觉附件
        asset_options.fix_base_link = self.cfg.asset.fix_base_link  # 固定基座连杆
        asset_options.density = self.cfg.asset.density  # 密度
        asset_options.angular_damping = self.cfg.asset.angular_damping  # 角阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping  # 线阻尼
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity  # 最大角速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity  # 最大线速度
        asset_options.armature = self.cfg.asset.armature  # 电枢
        asset_options.thickness = self.cfg.asset.thickness  # 厚度
        asset_options.disable_gravity = self.cfg.asset.disable_gravity  # 禁用重力

        # 加载机器人资产
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 获取DOF数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 获取刚体数量
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)  # 获取DOF属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)  # 获取刚体形状属性

        # 从资产中保存身体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)  # 刚体名称
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)  # DOF名称
        self.num_bodies = len(body_names)  # 刚体数量
        self.num_dofs = len(self.dof_names)  # DOF数量

        # 查找足部名称
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        
        # 查找惩罚接触名称
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        
        # 查找终止接触名称
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始化基座状态
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])  # 设置初始位置

        # 获取环境原点
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []  # 演员句柄列表
        self.envs = []  # 环境句柄列表
        
        # 为每个环境创建实例
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            

            # TODO:random init pose quat
            # 设置初始位置（添加随机偏移）
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            # 处理刚体形状属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            
            # 创建演员（机器人实例）
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            # 处理DOF属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            
            # 处理刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            
            # 保存环境句柄和演员句柄
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        print("=========刚体顺序=========")
        print(body_names)
        print("=========DOF顺序=========")
        print(f"DOF数量:{self.num_dof}\n ")
        print("(is_driven, lower_limit, upper_limit, drive_mode, stiffness, damping, max_force, friction, armature, velocity)")
        print(f"DOF属性:{dof_props}\n")
        print(self.dof_names) # 动作输出顺序
        print("=========================")

        # 获取足部索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # 获取惩罚接触索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        # 获取终止接触索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ 设置环境原点。在崎岖地形上，原点由地形平台定义。否则创建网格 """
        self.custom_origins = False  # 默认不使用自定义原点
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        # 创建机器人网格
        num_cols = np.floor(np.sqrt(self.num_envs))  # 列数
        num_rows = np.ceil(self.num_envs / num_cols)  # 行数
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing  # 环境间距
        
        # 设置环境原点坐标
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]  # x坐标
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]  # y坐标
        self.env_origins[:, 2] = 0.  # z坐标

    def _parse_cfg(self, cfg):
        """ 解析配置参数 """
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 控制时间步长
        self.obs_scales = self.cfg.normalization.obs_scales  # 观测缩放因子
        # 将类转为字典
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 奖励缩放因子
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 命令范围

        self.max_episode_length_s = self.cfg.env.episode_length_s  # 最大回合长度（秒）
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 最大回合长度（步数）

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 推动间隔（步数）

    #------------ 奖励函数 ----------------
    def _reward_lin_vel_z(self):
        """ 惩罚z轴基座线速度 """
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        """ 惩罚xy轴基座角速度 """
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        """ 惩罚非平坦的基座方向 """
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """ 惩罚基座高度偏离目标 """
        base_height = self.root_states[:, 2]  # 基座高度
        return torch.square(base_height - self.cfg.rewards.base_height_target)  # 与目标高度的平方差
    
    def _reward_torques(self):
        """ 惩罚力矩 """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """ 惩罚关节速度 """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """ 惩罚关节加速度 """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        """ 惩罚动作变化率 """
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        """ 惩罚选定身体的碰撞 """
        # 检查接触力是否大于0.1N
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        """ 终止奖励/惩罚 """
        return self.reset_buf * ~self.time_out_buf  # 只在非超时重置时给予惩罚
    
    def _reward_dof_pos_limits(self):
        """ 惩罚关节位置接近限制 """
        # 计算低于下限的超出量（负值）
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        # 加上高于上限的超出量（正值）
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """ 惩罚关节速度接近限制 """
        # 裁剪到最大误差 = 每个关节1 rad/s以避免巨大惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        """ 惩罚力矩接近限制 """
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        """ 线速度命令跟踪（xy轴） """
        # 计算命令与基座线速度的平方误差
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 使用指数衰减函数计算奖励
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        """ 角速度命令跟踪（偏航） """
        # 计算命令与基座角速度的平方误差
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # 使用指数衰减函数计算奖励
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        """ 奖励长步态 """
        # 需要过滤接触，因为PhysX在网格上的接触报告不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # z方向接触力大于1N
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 当前或上次接触
        self.last_contacts = contact  # 保存当前接触状态
        
        # 首次接触标志：空中时间>0且现在有接触
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt  # 增加空中时间
        
        # 只在首次接触地面时给予奖励（空中时间-0.5）
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        # 无命令时不给予奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        # 重置接触的足部空中时间
        self.feet_air_time *= ~contact_filt
        
        return rew_airTime
    
    def _reward_feet_stumble(self):
        """ 绊倒惩罚：惩罚足部撞击垂直表面 """
        # 检查xy方向接触力是否大于5倍z方向接触力
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        """ 惩罚零命令时的运动 """
        # 当命令很小时，惩罚关节位置偏离默认位置
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        """ 惩罚高接触力 """
        # 计算超过最大接触力的部分
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)