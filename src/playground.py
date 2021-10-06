#! /usr/bin/python3

## Import basic modules ##
import sys

## Import utility modules ##
import pybullet as p
import pybullet_utils.bullet_client as c
import numpy as np
import datetime
import time

# import user-defined modules ##
import robot
import control
from constants import *

# import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym import spaces, Env
import cv2




## Set up physical models
def setupWorld(client):

    p = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    modelTerrain  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(modelTerrain, -1, lateralFriction=1.0)

    ## Set up robot
    modelRobot = robot.modelMobile(p, [0., 0., 0.5], [0, 0, 0, 1])

    return modelRobot, modelTerrain


## Terrain affordance module
class envTest(Env):

    def __init__(self,  training=True, recording=None, reward=COEFF_REWARD_DEFAULT):

        if training:
            self.client = c.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = c.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.dicRewardCoeff = reward

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(30,), dtype=np.float32)

        self.action_space = spaces.Box( low=-1,
                                        high=1,
                                        shape=(NUM_ACTIONS,), dtype=np.float32)

        if recording is not None:
            self.flagRecord = True
            self._camera_dist = 1.0
            self._camera_yaw = 0
            self._camera_pitch = -30
            self._render_width = 480
            self._render_height = 360
            self._video_format = cv2.VideoWriter_fourcc(*'MP42')
            self._video_save_path = recording
        else:
            self.flagRecord = False


    def seed(self):

        pass


    def reset(self):

        # Generate a new episode
        self._setup()

        # Observation initiation
        ob = self._get_obs()

        return ob


    def _setup(self):

        ## Initiate simulation
        self.client.resetSimulation()

        self.client.setAdditionalSearchPath(PATH_DATA)

        ## Set up simulation
        self.client.setTimeStep(TIME_STEP)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
 
        ## Set up playground
        self.client.setGravity(0, 0, -9.8)

        # creating environment
        self.modelRobot, self.modelTerrain = setupWorld(self.client)

        if self.flagRecord:
            path = "{}/{}.avi".format(self._video_save_path, datetime.datetime.now().strftime("%m%d_%H%M%S"))
            print(path)
            self.recorder = cv2.VideoWriter(path,
                                            self._video_format, 30, (self._render_width, self._render_height))

        self.control = control.ctlrRobot(self.modelRobot)
        self.dicCmdParam = {"Offset": np.zeros(NUM_COMMANDS), 
                            "Scale":  np.array([1] * NUM_COMMANDS)}
        self.dicActParam = {"Offset": np.zeros(NUM_ACTIONS), 
                            "Scale":  np.array([1] * NUM_ACTIONS)}
        self.target = self._get_targets()

        for _ in range(REPEAT_INIT):
            self.control.holdRobot()
            self.client.stepSimulation()
            self._get_obs()
        self._evaluate()


    def step(self, action):

        self._run_sim(action)
        ob = self._get_obs()
        reward, done, dicLog = self._evaluate()

        if self.flagRecord:
            img = self.render()
            self.recorder.write(img)

        if done:
            self.close()

        return ob, reward, done, dicLog


    def close(self):

        del self.modelRobot, self.modelTerrain
        del self.target
        if self.flagRecord:
            self.recorder.release()
            del self.recorder


    def _run_sim(self, action):

        while not self.control.updateAction(self.dicActParam["Scale"] * action + self.dicActParam["Offset"]):
            self.control.step()
            self.client.stepSimulation()


    def _evaluate(self):

        done = False
        dicLog = {}
        reward = 0

        target = self._get_targets()
        obsBodyPos, obsBodyAtt = self.modelRobot.getCurrentBodyPose()

        ### YOUR ENVIRONMENT CONSTRAINTS HERE ###
        rewPos = self.dicRewardCoeff["position"] * np.sum((target[0:3] - obsBodyPos)**2)
        rewYaw = self.dicRewardCoeff["yaw"] * (target[3] - obsBodyAtt[2])**2
        reward = rewPos + rewYaw

        dicLog["Test"] = 0

        if done:
            dicLog["Done"] = 1

        return reward, done, dicLog


    def _get_obs(self):

        obsBodyPos, obsBodyAtt = self.modelRobot.getCurrentBodyPose()
        obsWheel = self.modelRobot.getCurrentWheelJoints(VELOCITY)

        return np.concatenate((obsBodyPos, obsBodyAtt, obsWheel), axis=None)


    def _get_targets(self):

        return np.array([1, 0, 0, 0.2])


    def render(self):

        if not self.flagRecord:
            return
            
        try:
            raBodyXYZ, raBodyAng = self.modelRobot.getCurrentBodyPose()
        except Exception as e: 
            sys.stdout.write("\r{}\r\n".format(e))
            raBodyXYZ = np.zeros(3)

        view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=raBodyXYZ,
            distance=self._camera_dist,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self.client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            renderer=self.client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        image = rgb_array[:, :, :3]

        return image
