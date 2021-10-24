#! /usr/bin/python3

import os
import yaml

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal


from tianshou.policy import PPOPolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector

from playground import envTest
from constants import *


def eval(path, epoch=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo_config = yaml.load(open("{}/config/{}".format(path, SUBPATH_CONFIG["ppo"]), 'r'))
    reward_config = yaml.load(open("{}/config/{}".format(path, SUBPATH_CONFIG["reward"]), 'r'))

    if epoch is not None:
        video_path = "{}/video/checkpoint_{}".format(path, epoch)
    else:
        video_path = "{}/video/policy".format(path)
    os.makedirs(video_path, exist_ok=True)

    env = envTest(training=True, recording=save_test_video_path, reward=reward_config)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net_a = Net(state_shape, hidden_sizes=(256, 256), device=device)
    net_c = Net(state_shape, hidden_sizes=(256, 256), device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=True, device=device).to(device)
    critic = Critic(net_c, device=device).to(device)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo_config["Learning Rate"])

    lr_scheduler = None
    if ppo_config["Learning Rate Decay"]:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            ppo_config["Step/Epoch"] / ppo_config["Step/Collect"]) * ppo_config["Epoch"]

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

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

    if epoch:
        filepath = "{}/policy/checkpoint_{}.pth".format(path, epoch)
        checkpoint = torch.load(filepath, map_location=device)
        policy.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        print("\rLoaded agent from: ", filepath)
    else:
        filepath = "{}/policy/policy.pth".format(path)
        checkpoint = torch.load(filepath, map_location=device)
        policy.load_state_dict(checkpoint)
        print("\rLoaded agent from: ", filepath)


    test_collector = Collector(policy, env)

    policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    result = test_collector.collect(n_episode=1)
    print(f'\rFinal reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':

    task = "Blind Locomotion".replace(" ", "_")
    exp = "seed_0_0804_152302_ppo"

    # eval("{}/{}/{}".format(PATH_SAVE, task, exp), )

    for idx in range(12):
        epoch = (idx+1) * 10 
        eval("{}/{}/{}".format(PATH_SAVE, task, exp), epoch)
