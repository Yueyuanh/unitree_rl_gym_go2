
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
# 学习任务基础类
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        # 获取参数
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        # 是否开启渲染
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        # 优化 禁用profiling 模式 减少运行开销
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers 分配内存
        # 各个环境中的观测值
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        # reward 奖励信息
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 存储是否需要重置的标志 全1
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # 每个环境中当前的episode的长度（步数）
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 每个环境是否超时的
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 特权信息观测器
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        # 创建仿真环境
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        # 渲染模式创建可视化窗口 
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            # 按键事件
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "GO")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "Lift")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "Back")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "Right")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_Q, "Yaw+")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_E, "Yaw-")
    
    # 获取观测器信息接口
    def get_observations(self):
        return self.obs_buf
    
    # 获取特权观测器
    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
    # 重置某个环境
    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError
    
    # 重置所有环境
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    # 仿真任务 弱定义 具体根据机器人不同而不同,传入参数为action
    # 返回参数 观测值 特权观测值 _ _ _
    def step(self, actions):
        raise NotImplementedError

    # 渲染
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            # 处理键盘事件
            for evt in self.gym.query_viewer_action_events(self.viewer):

                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync



            # fetch results 获取结果
            # 确保GPU的时间能被同步到CPU上
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                # 同步渲染
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                # 停止渲染
                self.gym.poll_viewer_events(self.viewer)