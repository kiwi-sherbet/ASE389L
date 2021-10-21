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


def setupObstacles(client, number):
    p = client
    idColCylinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0.5)
    idVisualShape = -1
    raRandPos = FIELD_RANGE*np.random.randn(number, 2)

    mass = 10
    baseOrientation = [0, 0, 0, 1]

    raObstacles = []

    for idx in range(number):
        basePosition = [raRandPos[idx,0], raRandPos[idx,1], 0.25]
        raObstacles.append(p.createMultiBody(mass, idColCylinder, idVisualShape, basePosition, baseOrientation))

    return raObstacles


def setupGoal(client, coord):
    p = client
    idVisualShape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.05)
    idCollisionShape = None

    basePosition = [coord[0], coord[1], 0.025]
    modelGoal = p.createMultiBody(baseVisualShapeIndex=idVisualShape, basePosition=basePosition)

    return modelGoal


## Set up physical models
def setupWorld(client):

    p = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    modelTerrain  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(modelTerrain, -1, lateralFriction=1.0)

    raObstacles = setupObstacles(p, 20)
    modelGoal = setupGoal(p, FIELD_RANGE*np.random.randn(2))
    ## Set up robot
    modelRobot = robot.modelMobile(p, [0., 0., 0.5], [0, 0, 0, 1])

    raModels = {'robot': modelRobot, 'terrain': modelTerrain, 'goal': modelGoal, 'obstacles': raObstacles}

    return raModels



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
                                            shape=(15,), dtype=np.float32)

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
            self.recorder = None
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
        self.models = setupWorld(self.client)
        self.robot = self.models['robot']
        self.goal = self.models['goal']

        self.control = control.ctlrRobot(self.robot)
        self.dicCmdParam = {"Offset": np.zeros(NUM_COMMANDS), 
                            "Scale":  np.array([1] * NUM_COMMANDS)}
        self.dicActParam = {"Offset": np.zeros(NUM_ACTIONS), 
                            "Scale":  np.array([1] * NUM_ACTIONS)}
        self.target = self._get_targets()

        self.cnt = 0

        for _ in range(REPEAT_INIT):
            self.control.holdRobot()
            self.client.stepSimulation()
            self._get_obs()
        self._evaluate()


    def step(self, action):

        self._run_sim(5*action)
        ob = self._get_obs()
        reward, done, dicLog = self._evaluate()

        if self.flagRecord:
            if self.recorder == None:
                self._start_recorder()

            self._write_recorder()

            if done:
                self._close_recorder()


        return ob, reward, done, dicLog


    def close(self):

        pass

    
    def _write_recorder(self):
        
        img = self.render()
        self.recorder.write(img)


    def _start_recorder(self):

        path = "{}/{}.avi".format(self._video_save_path, datetime.datetime.now().strftime("%m%d_%H%M%S"))
        self.recorder = cv2.VideoWriter(path, self._video_format, 30, (self._render_width, self._render_height))


    def _close_recorder(self):

        self.recorder.release()
        self.recorder = None


    def _run_sim(self, action):

        while not self.control.updateAction(self.dicActParam["Scale"] * action + self.dicActParam["Offset"]):
            self.control.step()
            self.client.stepSimulation()


    def _evaluate(self):

        done = False
        dicLog = {}
        dicRew = {}
        reward = 0

        target = np.array(self._get_targets())
        raWheelTrq = np.array(self.robot.getCurrentWheelJoints(TORQUE))
        raWheelVel = np.array(self.robot.getCurrentWheelJoints(VELOCITY))
        raBodyPos, raBodyAtt = self.robot.getCurrentBodyPose()
        raBodyPos = np.array(raBodyPos)
        raBodyAtt = np.array(raBodyAtt)

        ### YOUR ENVIRONMENT CONSTRAINTS HERE ###
        
        sqrErr = np.sum((target - raBodyPos)**2)
        valEng = np.sum(np.absolute(raWheelTrq.flatten() @ raWheelVel.flatten()))

        dicRew["Position"] = self.dicRewardCoeff["Position"] * sqrErr
        dicRew["Energy"] = self.dicRewardCoeff["Energy"] * valEng

        if sqrErr < 0.04:
            done = True
            dicRew["Terminal"] = self.dicRewardCoeff["Terminal"]
        elif self.cnt > 1000 or self.robot.checkFlipped():
            done = True
        else:
            self.cnt+=1

        for rew in dicRew.values():
            reward += rew

        if done:
            dicLog["Done"] = 1
        dicLog["Reward"] = dicRew

        return reward, done, dicLog


    def _get_obs(self):

        obsBodyPos, obsBodyAtt = self.robot.getCurrentBodyPose()
        obsWheel = self.robot.getCurrentWheelJoints(VELOCITY)
        obsVel = self.robot.getBaseVelocity()[0:2]
        obsYawRate = self.robot.getBaseRollPitchYawRate()[2]
        obsTarget = self.robot.getTargetToLocalFrame(self._get_targets())

        return np.concatenate((obsBodyPos, obsBodyAtt, obsWheel, obsVel, obsYawRate, obsTarget), axis=None)


    def _get_targets(self):

        raXYZ, _ = self.client.getBasePositionAndOrientation(self.goal)

        return raXYZ


    def render(self):

        if not self.flagRecord:
            return
            
        try:
            raBodyXYZ, raBodyAng = self.robot.getCurrentBodyPose()
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
