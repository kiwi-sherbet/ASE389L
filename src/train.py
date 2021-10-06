#! /usr/bin/python3

import os
import pprint
import yaml
import datetime

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

from playground import envTest
from logger import WandbLogger
from constants import *


def train():

    experiment_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["experiment"]), 'r'))
    ppo_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["ppo"]), 'r'))
    reward_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["reward"]), 'r'))

    device = experiment_config["Device"] if torch.cuda.is_available() else "cpu"

    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    save_name = f'{experiment_config["Save"]}_Sigma_{ppo_config["Initial Sigma"]}_Learning_{ppo_config["Learning Rate"]}_{t0}'
    save_path = os.path.join(PATH_SAVE, experiment_config["Task"].replace(" ", "_"), save_name)
    policy_path = os.path.join(save_path, 'policy')
    config_path = os.path.join(save_path, 'config')
    save_training_video_path = "{}/video/train".format(save_path)
    save_test_video_path = "{}/video/test".format(save_path)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(policy_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)
    os.makedirs(save_training_video_path, exist_ok=True)
    os.makedirs(save_test_video_path, exist_ok=True)

    env = envTest(training=True, recording=save_test_video_path, reward=reward_config)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    train_envs = SubprocVectorEnv(
        [lambda: envTest(training=True, reward=reward_config) for _ in range(ppo_config["Training Envs"])],
        norm_obs=False)

    test_envs = SubprocVectorEnv(
        [lambda: envTest(training=True, reward=reward_config) for _ in range(ppo_config["Test Envs"]-1)]
        + [lambda: envTest(training=True, recording=save_test_video_path, reward=reward_config)],
        norm_obs=False, obs_rms=train_envs.obs_rms, update_obs_rms=False)

    # seed
    seed = experiment_config["Seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    # train_envs.seed()
    # test_envs.seed()

    activation = [nn.ReLU, nn.ReLU]

    net_a = Net(state_shape, hidden_sizes=(256, 256), device=device)
    net_c = Net(state_shape, hidden_sizes=(256, 256), device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=True, device=device).to(device)
    critic = Critic(net_c, device=device).to(device)
    
    # # orthogonal initialization
    # for m in list(actor.modules()) + list(critic.modules()):
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(m.weight)
    #         torch.nn.init.zeros_(m.bias)

    torch.nn.init.constant_(actor.sigma_param, ppo_config["Initial Sigma"])
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo_config["Learning Rate"])

    lr_scheduler = None
    if ppo_config["Learning Rate Decay"]:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            ppo_config["Step/Epoch"] / ppo_config["Step/Collect"]) * ppo_config["Epoch"]

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        try:
            return Independent(Normal(*logits), 1)
        except ValueError as e:
            print(logits)
            print(logits[0].shape, logits[1].shape)
            raise ValueError from e

    print("\r\nExperiment Info")
    pprint.pprint(experiment_config)
    print("\r\nPPO CONFIG")
    pprint.pprint(ppo_config)
    print("\r\nReward CONFIG")
    pprint.pprint(reward_config)

    policy = PPOPolicy( actor, critic, optim, dist, 
                        discount_factor=ppo_config["Gamma"],
                        gae_lambda=ppo_config["GAE Lambda"],
                        max_grad_norm=ppo_config["Max Grad Norm"],
                        vf_coef=ppo_config["Value Coefficient"], 
                        ent_coef=ppo_config["Entropy Coefficient"],
                        reward_normalization=ppo_config["Reward Nomalization"], 
                        action_scaling=True,
                        action_bound_method="clip",
                        lr_scheduler=lr_scheduler, 
                        action_space=env.action_space,
                        eps_clip=ppo_config["Epsilon Clip"],
                        value_clip=ppo_config["Value Clip"],
                        dual_clip=ppo_config["Dual Clip"], 
                        advantage_normalization=ppo_config["Advantage Normalization"],
                        recompute_advantage=ppo_config["Recompute Advantage"])

    # collector
    if ppo_config["Training Envs"] > 1:
        buffer = VectorReplayBuffer(ppo_config["Buffer Size"], len(train_envs))
    else:
        buffer = ReplayBuffer(ppo_config["Buffer Size"])
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    with open(os.path.join(config_path, 'experiment.yaml'), 'w') as yaml_file:
        yaml.dump(experiment_config, yaml_file, default_flow_style=False)

    with open(os.path.join(config_path, 'ppo.yaml'), 'w') as yaml_file:
        yaml.dump(ppo_config, yaml_file, default_flow_style=False)

    with open(os.path.join(config_path, 'reward.yaml'), 'w') as yaml_file:
        yaml.dump(reward_config, yaml_file, default_flow_style=False)

    # log
    logger = WandbLogger(project = experiment_config["Project"], task = experiment_config["Task"], 
                        path=save_path, update_interval=100, train_interval=100, 
                        reward_config=reward_config, ppo_config=ppo_config, experiment_config=experiment_config,
                        actor=net_a, critic=net_c)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(policy_path, 'policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'model': policy.state_dict(),
            'optim': optim.state_dict(),
        }, os.path.join(policy_path, 'checkpoint_{}.pth'.format(epoch)))

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, ppo_config["Epoch"], ppo_config["Step/Epoch"],
        ppo_config["Repeat/Collect"], ppo_config["Test Envs"], ppo_config["Batch Size"],
        step_per_collect=ppo_config["Step/Collect"], save_fn=save_fn, save_checkpoint_fn=save_checkpoint_fn, logger=logger,
        test_in_train=False)

    pprint.pprint(result)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()
