import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils import ExperimentLogger
from legged_gym.utils.helpers import launch_tensorboard
from legged_gym.utils.helpers import class_to_dict
import torch

def train(args):

    logdir = ExperimentLogger.generate_logdir(args.task)
    exp_msg = {}
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    exp_msg = ExperimentLogger.commit_experiment(logdir, args)  # force to commit expriment message

    if args.launch_tensorboard:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task)
        launch_tensorboard(log_root)

    env, env_cfg = task_registry.make_env(name=args.task, args=args,env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=logdir
    )
    exp_msg["env_cfg"] = class_to_dict(env_cfg)
    exp_msg["train_cfg"] = class_to_dict(train_cfg)
    ExperimentLogger.save_hyper_params(logdir, env_cfg, train_cfg)


    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True,experiment_log=exp_msg)

if __name__ == '__main__':
    args = get_args()
    train(args)
