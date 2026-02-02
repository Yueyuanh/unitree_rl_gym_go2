from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class PikachuRough10Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.2] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0

        #    'back_tail_joint' : 0,   
        #    'left_arm_joint' : -1,               
        #    'right_arm_joint' : -1,   

           'left_hip_pitch_joint' : 0.56,   
           'left_hip_roll_joint' : 0,               
           'left_hip_yaw_joint' : 0,         
           'left_knee_joint' : 1.53,       
           'left_ankle_joint' : 0.9,     

           'right_hip_pitch_joint' : 0.56, 
           'right_hip_roll_joint' : 0,               
           'right_hip_yaw_joint' : 0,         
           'right_knee_joint' : 1.53,       
           'right_ankle_joint' : 0.9
        }
    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 10 + 10 + 10 + 2 = 41
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_rane = [-1., 3.]
        push_robots = True
        push_interval_s = 5 #推动间隔
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = { 
                    'hip_yaw': 50,
                    'hip_roll': 100,
                    'hip_pitch': 80,
                    'knee': 80,
                    'ankle': 80
                    }  # [N*m/rad] -
        
        damping = {   
                    'hip_yaw': 0.5,
                    'hip_roll': 2,
                    'hip_pitch': 0.1,
                    'knee': 0.1,
                    'ankle': 0.1
                    }  # [N*m*s/rad] - 
        
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放系数，网络输出动作是1/0.25=4倍，actionScale越小，action输出范围就越大，动作就越精细
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # 每4步发送更新一次控制量，
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Pikachu_V01/urdf/Pikachu_V01_10dof.urdf'
        name = "Pikachu_10DOF"
        foot_name = "ankle"
        penalize_contacts_on = ["hip","knee"] # 双足模式惩罚手臂触地，
        terminate_after_contacts_on = ["base_link","tail","hip","knee"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25 # 0.2855
        
        class scales( LeggedRobotCfg.rewards.scales ):# 大倍数提升时reward会呈倍数突变，
            tracking_lin_vel = 2 # 1.0
            tracking_ang_vel = 0.5 # 0.5
            lin_vel_z = -2.0   # Z轴上下惩罚（放置跳跃 摔倒）-2.0
            ang_vel_xy = -0.05 # Roll轴角速度惩罚   -0.05
            orientation = -1.0 # 机体旋转角度惩罚 -1
            torques = -0.00001  # 力矩惩罚（能量）-0.00001 
            dof_acc = -2.5e-7   # 关节加速度
            dof_vel = -0.001     # 关节速度
            base_height = -10.0 # 基础身高 -10
            feet_air_time = 0.0 # 足部悬空时间
            collision = -0.0     # 碰撞(penalize_contacts_on)
            action_rate = -0.2 # 两次动作变化率惩罚
            stand_still = -0.02  # 静止站立惩罚

            dof_pos_limits = -5.0 # 关节角度限制
            alive = 0.15          # 生存奖励  
            hip_pos = -1.0        # hip_roll hip_yaw 保持原位
            contact_no_vel = -0.5 # 足端撞地惩罚
            feet_swing_height = -10.0 # 足端摆动高度
            contact = 0.18            # 接触奖励（符合步态相位）


class PikachuRough10CfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8 #0.8
        # # Actor网络隐藏层维度
        actor_hidden_dims = [512, 256, 128]
        # Critic网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]

        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 #策略熵系数（探索度） 0.01  Converged to a local minimum(收敛到局部最小) -> Higher value Does not converge fast enough（收敛不够快） -> Lower value
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        # num_actions_per_env=10
        max_iterations = 100000
        run_name = ''
        experiment_name = 'Pikachu_V01'

        # load and resume
        resume = False
        # load_run = -1 # -1 = last run
        # checkpoint = -1 # -1 = last saved model

        # load_run = "Jan29_23-17-16_"
        # checkpoint = 32000