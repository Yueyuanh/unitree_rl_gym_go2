import time
import sys
import os

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

from deploy_joystick import DeployJoystick

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

from LPF import LPF

TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
LEGGED_GYM_ROOT_DIR = "/mnt/data/Projects/unitree_rl_gym"

sys.path.append(os.path.join(os.path.dirname(__file__)))

from log.data_logger import DataLogger
from log.plot_data import plot_csv_data

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    # 重力方向
    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_to_mat(quat):
    """将四元数 (w, x, y, z) 转换为旋转矩阵"""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])
    return R

# PD 控制器  目标位置  当前位置  Kp 目标速度  当前速度  Kd
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def WirelessControllerHandler(msg:WirelessController_):
    global joymsg
    joymsg=msg

if __name__ == "__main__":
    # get config file name from command line
    import argparse
    # 获取传入参数
    parser = argparse.ArgumentParser()
    # 设定参数名称和类型 config_file  str  提示
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()     # 获取参数
    config_file = args.config_file # 传入参数
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        # 加载yaml参数
        config = yaml.load(f, Loader=yaml.FullLoader)
        # 策略路径      replace 用具体参数替代yaml中的位置 
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        # 加载mujoco模型 
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # 仿真时间
        simulation_duration = config["simulation_duration"]
        # 仿真时间步长
        simulation_dt = config["simulation_dt"]
        # 控制周期倍率  control freq = comtrol_decimation * dt
        control_decimation = config["control_decimation"]

        # 加载PD参数 并设置为np.array格式
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        # 加载初始化默认关节角度
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        # lin_vel_scale = config["lin_vel_scale"]
        # yaw角速度缩放系数?
        ang_vel_scale = config["ang_vel_scale"]
        # 关节角度缩放系数
        dof_pos_scale = config["dof_pos_scale"]
        # 关节速度缩放系数
        dof_vel_scale = config["dof_vel_scale"]
        # 动作缩放系数
        action_scale = config["action_scale"]
        # 控制指令缩放系数
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # 动作空间
        num_actions = config["num_actions"]
        # 观测器空间
        num_obs = config["num_obs"]
        # 初始化指令
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # joy
        use_joystick=config["joystick"]

    # define context variables
    # 初始化动作 目标位置 观测器 
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_pos_new= default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    logger = DataLogger(log_dir="./logs", file_name="1_logs.csv")

    # Joystick初始化
    if use_joystick:
        ChannelFactoryInitialize(1)
        DeployJoy=DeployJoystick()
        DeployJoy.SetupJoystick()

        wirelesscontroller = ChannelSubscriber(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        wirelesscontroller.Init(WirelessControllerHandler, 10)


    counter = 0

    # Load robot model
    # 加载机器人Mujoco模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt


    # 初始化输出机器人关节信息 
    print("<<------------- Link ------------->> ")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_BODY, i)
        if name:
            print("link_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Joint ------------->> ")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
        if name:
            print("joint_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Actuator ------------->>")
    for i in range(m.nu):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
        )
        if name:
            print("actuator_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Sensor ------------->>")
    index = 0
    for i in range(m.nsensor):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
        )
        if name:
            print(
                "sensor_index:",
                index,
                ", name:",
                name,
                ", dim:",
                m.sensor_dim[i],
            )
        index = index + m.sensor_dim[i]
    print(" ")



    # load policy
    policy = torch.jit.load(policy_path)


    # LPF
    filter = LPF(alpha=0.5, n_dof=13)
    tau_filter = LPF(alpha=0.99, n_dof=13)

    tau_limit=[3,27,27,12.5,12.5,12.5,12.5, 27,27,12.5,12.5,12.5,12.5 ]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # 仿真总时间 duration
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time() # 单步时间记录
            soft_limit=0.95
            limit_angles_rl=[[-3.14,-0.77],[-0.7,2.7],[-0.12,0.14],[-0.26,0.26],[0.04,2.44],[-1.64,1.64]]

            if   target_dof_pos[0]<-1.05*soft_limit: target_dof_pos[0]= -1.05*soft_limit
            elif target_dof_pos[0]> 1.05*soft_limit: target_dof_pos[0]=  1.05*soft_limit
                
            for i in range(6):
                if   target_dof_pos[i+1]<limit_angles_rl[i][0]*soft_limit: target_dof_pos[i+1]= limit_angles_rl[i][0]*soft_limit
                elif target_dof_pos[i+1]>limit_angles_rl[i][1]*soft_limit: target_dof_pos[i+1]= limit_angles_rl[i][1]*soft_limit
                
                if   target_dof_pos[i+7]<limit_angles_rl[i][0]*soft_limit: target_dof_pos[i+7]= limit_angles_rl[i][0]*soft_limit
                elif target_dof_pos[i+7]>limit_angles_rl[i][1]*soft_limit: target_dof_pos[i+7]= limit_angles_rl[i][1]*soft_limit


  
            # target_dof_pos=default_angles.copy()
            # PD控制器计算输出力矩 初始化力矩 目标位置 | 位置反馈| Kp | 目标速度为0 阻尼  |关节速度反馈| Kd 
            # target_dof_pos=filter.update(target_dof_pos)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)

            # print(target_dof_pos[7])

            tau=np.nan_to_num(tau,nan=0.0,posinf=0.0,neginf=0.0)
            # print(tau)

            # tau cilp
            if tau[0]>3:    tau[0]=3
            elif tau[0]<-3: tau[0]=-3

            for i in range(12):
                if   tau[i+1]>tau_limit[i+1]:  tau[i+1]=tau_limit[i+1]
                elif tau[i+1]<-tau_limit[i+1]: tau[i+1]=-tau_limit[i+1]

            # 手柄控制
            if use_joystick:
                cmd[0]=joymsg.rx
                cmd[1]=joymsg.ry
                cmd[2]=-joymsg.lx

                joykeys_event=DeployJoy.JoykeysPress()

                if 'select' in joykeys_event["press"]:
                    mujoco.mj_resetData(m, d)
                    mujoco.mj_forward(m, d)   # 更新一次，保证观测量正确
                    print("Environment reset!")
            

            # print(target_dof_pos[0],d.qpos[7:][0],tau[0])
            # 发送控制指令


            d.ctrl[:] = tau_filter.update(tau)

            d.ctrl[:]=tau
            # d.ctrl[1]=0
            # d.ctrl[7]=0

            data={}
            for i, value in enumerate(tau):
                data[f"tau_{i}"] = value  # tau_0, tau_1, ..., tau_11
    
            # logger.log_data(data)


            mujoco.mj_step(m, d) 

            counter += 1
            # 10


            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                # 观测器
                # d.qpos存储所有自由度的位置，前7个元素是根节点的位置（3d）+四元数姿态（4d），
                #       [7:]是后续关节的位置
                qj = d.qpos[7:]     # 关节位置
                # d.qvel中存储所有自由度的速度，前六个元素是根节点的线速度（3d）+角速度（3d）,
                #       [6:]是后续关节的速度
                dqj = d.qvel[6:]    # 关节速度
                quat = d.qpos[3:7]  # 四元数
                omega = d.qvel[3:6] # 角速度

                # print(omega.round(3))
                # 根节点线速度
                # base_vel_world = d.qvel[:3]

                # 现实世界很难获得，需要通过卡尔曼观测器估计机体三维线速度信息
                # 应该是机体坐标系下的速度而不是世界坐标系
                # base_vel_body=np.zeros(3)
                R=quat_to_mat(quat)
                # base_vel_body=R.T @ base_vel_world
                # omega=R.T @ omega
                # base_vel=base_vel_body*lin_vel_scale

                # print(base_vel_world,base_vel_body)

                # 关节位置 = ( 关节位置 - 初始化角度 )*关节位置缩放系数（1.0）
                qj = (qj - default_angles) * dof_pos_scale
                # 关节速度 = 关节速度 * 速度缩放系数（0.05）
                dqj = dqj * dof_vel_scale
                # 重力分量(方向)
                gravity_orientation = get_gravity_orientation(quat)
                # yaw角速度*缩放系数（0.25）
                omega = omega * ang_vel_scale
                # print(gravity_orientation.round(4))

                # print(cmd)

                # 周期
                period = 0.8
                count = counter * simulation_dt
                # # 相位
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # print(sin_phase,cos_phase)
                # print(base_vel,cmd* cmd_scale)

                # print(gravity_orientation.round(4))
                # print(omega)
                # 打包观测器 打包成一维

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                #obs[9,22]
                obs[9 : 9 + num_actions] = qj
                # obs[22:35]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                # obs[35,48] 上一时刻的动作
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                # obs[48:50] sin cos 信号 相位信息
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                # obs=[-8.0540e-02, -1.6194e-01,  4.1675e-02, -1.4745e-01, -7.3020e-02,   
                # -9.8637e-01,  1.0305e+00, -1.0089e+00, -2.2490e-01,  5.3667e-01,
                # -1.5164e+00, -4.0075e-01,  1.4100e-01, -1.2256e-02,  1.6431e+00,
                # -1.0473e+00, -1.8177e+00, -3.5059e-01,  1.4144e-01,  5.7338e-02,
                # 1.9748e+00, -1.2052e+00,  3.7655e-01,  4.2059e-03,  1.6979e-01,
                # -1.4942e-02,  3.5155e-01, -8.5052e-02, -8.9427e-04,  4.6827e-02,
                # -5.8924e-02, -7.1224e-02,  2.6418e-01,  1.4206e-02, -5.8199e-03,
                # 2.0944e+00, -6.0467e+00, -1.3720e+00,  6.7695e+00,  1.4935e+00,
                # 4.3352e+00, -4.1948e+00, -7.1790e+00, -1.5414e+00,  2.0149e+00,
                # 6.4584e-01,  7.8638e+00, -4.8295e+00,  6.2783e-07, -1.0000e+00]

                # tensor([[ 2.0944, -6.0467, -1.3720,  6.7695,  1.4935,  4.3352, 
                #           -4.1948, -7.1790,-1.5414,  2.0149,  0.6458,  7.8638,
                #           -4.8295]]

                # [ 2.0068092 -6.9474497 -0.8435412  6.433929   1.2357185  5.315982
                #  -4.1622024 -6.8173785 -1.1801616  3.4977179  1.1306261  5.199345
                #  -5.2906165]

                # obs=[7.1055e-02,  3.3005e-04,  3.1044e-02, 
                #      -1.8151e-01, -1.3374e-02,-9.8330e-01,
                #      -0.0000e+00, -0.0000e+00,  8.7392e-02,
                #      5.0813e-02, -7.7008e-01, -4.6794e-01,  1.4055e-01, -3.0442e-02,  1.9716e+00,
                #     -1.3050e+00, -7.7546e-01, -4.7927e-01, -1.2128e-01,  1.7992e-01,
                #      2.0425e+00, -1.3691e+00,
                
                #     -2.4589e-01, -9.2737e-04, -3.6394e-03,
                #     -1.4477e-02, -5.7018e-03,  2.3264e-02, -2.3863e-02, -6.5263e-03,
                #     2.3988e-03,  1.8767e-02,  2.6773e-03,  3.0690e-02, -3.0007e-02,

                #     3.1760e-01, -9.7260e-01, -1.9446e+00, -7.4880e+00,  1.9570e-01,
                #     5.4554e+00, -5.0065e+00, -2.1622e-01, -1.7229e+00,  2.7710e+01,
                #     -4.4098e-02,  4.0404e+00, -5.4060e+00, 

                #     7.0711e-01, -7.0710e-01]

                obs = np.array(obs, dtype=np.float32)

                # 将numpy数组转化为Pytorch张量（共享内存）添加批次维度 [obs_dim]->[1,obs_dim]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                # 策略推理      前向传播      |切断梯度计算|切回numpy|去除批次纬度[1,action_dim]->[action_dim] 
                action = policy(obs_tensor).detach().numpy().squeeze()

                # print(action)
                # transform action to target_dof_pos
                # 返回最终的关节位置=动作缩放后+初始位置
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = default_angles

                # print(obs[:3])
                print(target_dof_pos)
                # print(sin_phase,cos_phase)
                # # 输出观测向量各切片信息
                # print("=== 观测向量切片信息 ===")
                # print(f"[0:3]   角速度 omega: {obs[:3]}")
                # print(f"[3:6]   重力方向 gravity_orientation: {obs[3:6]}")
                # print(f"[6:9]   缩放命令 cmd*scale: {obs[6:9]}")
                # print(f"[9:22]  关节位置 qj: {obs[9:22]}")
                # print(f"[22:35] 关节速度 dqj: {obs[22:35]}")
                # print(f"[35:48] 上一动作 action: {obs[35:48]}")
                # print(f"[48:50] 相位 [sin,cos]: {obs[48:50]}")

                # tensor([[-0.3153,  0.5659, -1.8623, -7.6752,  0.3769,  4.4714, 
                #          -4.6902, 0.5220,-1.6662, 22.2204, -0.1231,  1.8568, 
                #          -5.4031]], device='cuda:0')



            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # 更新GUI 
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # tau_columns = [f"tau_{i}" for i in range(13)]  # 只显示前6个tau值，避免图形过于拥挤
    # plot_csv_data(logger.get_log_path(), tau_columns)