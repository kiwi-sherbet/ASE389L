import yaml
import numpy as np
import torch
import os
import cv2

from constants import *
from playground import envTest

experiment_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["experiment"]), 'r'))
ppo_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["ppo"]), 'r'))
reward_config = yaml.load(open("{}/{}".format(PATH_CONFIG, SUBPATH_CONFIG["reward"]), 'r'))
device = experiment_config["Device"] if torch.cuda.is_available() else "cpu"

save_path = os.path.join(PATH_SAVE, "Test")
save_test_video_path = "{}/video".format(save_path)
os.makedirs(save_test_video_path, exist_ok=True)

env = envTest(training=False, recording=save_test_video_path, reward=reward_config)
env.reset()

done = False
cnt = 0

while not done:
    # action = np.random.normal(0, 0.25, size=(2))
    # action = np.array([0.3, 0.3])
    action = np.cos(0.05*cnt) * np.array([1.0, 1.0])
    ob, rw, done, _ = env.step(action)

    rgb, depth = env.robot.getImgRGBD()
    
    cnt+=1

env.close()
