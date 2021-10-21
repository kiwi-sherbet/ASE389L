#! /usr/bin/python3

import numpy as np

from constants import *


class ctlrRobot(object):
        
    def __init__(self, hdlrRobot):

        self.raDefaultJointAction = np.array(DEFAULT_ACTION)
        self.raMinJointPos = np.array([-2] * NUM_ACTIONS)
        self.raMaxJointPos = np.array([2] * NUM_ACTIONS)

        self.hdlrRobot = hdlrRobot

        self._flagAction = True

        self.reset()
        self.rcvObservation()

        return


    def reset(self):

        ## Initiate internal variables

        self._cntStep = 0
        self._raAction = np.zeros(2)

        self.rcvObservation()


        ## Reset the quadruped
        self.resetRobot()

        return


    def resetRobot(self):
        """
        Reset joint positions.

        Args:
            None

        Returns:
            None
        """

        self.hdlrRobot.initCurrentBody(self.hdlrRobot.raPosXYZInit, self.hdlrRobot.raPosQInit)

        self.hdlrRobot.initCurrentWheeilJoints(self.raDefaultJointAction)

        return


    def holdRobot(self):

        self.rcvObservation()
        self.applyAction(self._raAction)


    def rcvObservation(self):

        self._raJointPos = self.hdlrRobot.getCurrentWheelJoints(POSITION)
        self._raJointVel = self.hdlrRobot.getCurrentWheelJoints(VELOCITY)

        return



    def applyAction(self, raAction):

        self.hdlrRobot.setTargetWheelJoints(POSITION, raAction)

        return


    def step(self):

        self.rcvObservation()
        self.applyAction(self._raAction)
        self._cntStep += 1

        return


    def getTrueMotorAngles(self):

        return self._raJointPos


    def getTrueMotorVelocities(self):

        return self._raJointVel


    def getTimeSinceReset(self):

        return self._cntStep * TIME_STEP


    def updateAction(self, raAction):

        if self._flagAction:
            self._flagAction = False
            for idxJoint in range(NUM_ACTIONS):
                self._raAction[idxJoint] = raAction[idxJoint]
                pass
            return True
        else:
            if self._cntStep % REPEAT_ACTION == 0:
                self._flagAction = True
            return False
