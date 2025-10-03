from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        # 特权信息维度
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        # 反弹系数
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        # 课程学习 先从简单命令开始 逐渐增大难度
        curriculum = False
        #  课程学习的最大难度系数
        max_curriculum = 1.
        # 控制纬度
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 命令重新生成的时间间隔 随机变换指令周期
        resampling_time = 10. # time before command are changed[s]
        # 朝向命令模式 就是yaw_set
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        # 位置控制
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        # 摩擦力域随机化
        randomize_friction = True 
        friction_range = [0.5, 1.25]
        # 质量域随机化
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        # 外界干扰力
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        # 总奖励 = SUM( 奖励分量 * 权重系数 )
        """
            # 第一优先级：安全稳定
            稳定性奖励 = lin_vel_z + ang_vel_xy + orientation + collision

            # 第二优先级：任务执行  
            任务奖励 = tracking_lin_vel + tracking_ang_vel

            # 第三优先级：运动质量(平滑型)
            质量奖励 = feet_air_time - action_rate - torques
        """
        class scales:
            termination = -0.0 
            tracking_lin_vel = 1.0  # 跟踪线性速度奖励系数
            tracking_ang_vel = 0.5  # 跟踪角速度 
            lin_vel_z = -2.0        # 垂直方向速度 放置跳跃下沉
            ang_vel_xy = -0.05      # 滚转/俯仰 角速度
            orientation = -0.       # 旋转 身体朝向偏差
            torques = -0.00001      # 关节力矩 节能 减少损耗
            dof_vel = -0.           # 关节速度
            dof_acc = -2.5e-7       # 关节加速度 减少冲击
            base_height = -0.5      # 身体重心高度 0
            feet_air_time =  1.0    # 足端悬空时间
            collision = -1.         # 碰撞
            feet_stumble = -0.5     # 脚部绊倒
            action_rate = -0.01     # 动作变化率 不能变的频繁
            stand_still = -0.       # 静止站立惩罚

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1. # 根据urdf的限制范围 1～100%
        soft_torque_limit = 1.
        base_height_target = 1. # 目标基础身体高度
        max_contact_force = 100. # forces above this value are penalized
        # 最大接触力限制

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [12, 0, 6]  # [m]
        lookat = [0, 0, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1    # 随机种子
    runner_class_name = 'OnPolicyRunner'
    class policy:
        # 初始化策略标准差
        init_noise_std = 1.0
        # Actor网络隐藏层维度
        actor_hidden_dims = [512, 256, 128]
        # Critic网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]
        # 激活函数 指数线性单元 避免梯度消失 训练更稳定
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        # 价值函数损失权重
        value_loss_coef = 1.0
        # 对价值函数也使用裁剪
        use_clipped_value_loss = True
        # PPO 裁剪参数 限制策略更新幅度
        clip_param = 0.2
        # 熵正则化系数 鼓励探索
        entropy_coef = 0.01

        # 每个数据批次重复训练5次
        num_learning_epochs = 5
        # 将数据分成4个小批次
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # 学习率
        learning_rate = 1.e-3 #5.e-4
        # 自适应学习率调度
        schedule = 'adaptive' # could be adaptive, fixed
        # 折扣因子
        gamma = 0.99
        # GAE 广义优势估计 参数
        lam = 0.95
        # 目标KL散度 用于自适应学习率
        desired_kl = 0.01
        # 梯度裁剪阈值
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        # 每个环境收集24步的数据
        num_steps_per_env = 24 # per iteration
        # 最大训练迭代次数
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''

        # load and resume
        # 是否从检查点恢复
        resume = False
        # 加载最后一次运行
        load_run = -1 # -1 = last run
        # 加载最后保存的模型
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt