#! /usr/bin/python3

import numpy as np

from constants import *
import pid


class ctlrRobot(object):
        
    def __init__(self, hdlrRobot):

        self.raDefaultJointAction = np.array(DEFAULT_ACTION)
        self.raMinJointPos = np.array([-2] * NUM_ACTIONS)
        self.raMaxJointPos = np.array([2] * NUM_ACTIONS)

        self.hdlrRobot = hdlrRobot

        self.raPIDGainP = GAIN_P
        self.raPIDGainD = GAIN_D

        self._flagAction = True

        self.reset()
        self.rcvObservation()

        return


    def reset(self):

        ## Initiate internal variables

        self._cntStep = 0
        self._dicAction = {}

        for idxJoint in range(0, NUM_ACTIONS):
            self._dicAction[idxJoint] = 0.4 #self.raDefaultJointAction[idxJoint]

        self.raMotorCtlr = [pid.hdlrPID(  self.raPIDGainP[idxJoint], self.raPIDGainD[idxJoint], 0, 1.0, 
                                            int(1.0/TIME_STEP), 0.3, 40, False) 
                            for idxJoint in range(0, NUM_ACTIONS)]

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
        self.applyAction(self._dicAction)


    def rcvObservation(self):

        self._raJointPos = self.hdlrRobot.getCurrentWheelJoints(POSITION)
        self._raJointVel = self.hdlrRobot.getCurrentWheelJoints(VELOCITY)

        return



    def applyAction(self, dicAction):

        q = self.getTrueMotorVelocities()

        for idxJoint in range(0, NUM_ACTIONS):

            fdbVal = dicAction[idxJoint] - q[idxJoint]
            fwdVal = 0

            if fwdVal != 0:
                fdbVal = 0

            self.raMotorCtlr[idxJoint].update(fdbVal, fwdVal)

        raJointTorq = [self.raMotorCtlr[idxJoint].output() for idxJoint in range(NUM_ACTIONS)]

        self.hdlrRobot.setTargetWheelJoints(TORQUE, raJointTorq)

        return


    def step(self):

        self.rcvObservation()
        self.applyAction(self._dicAction)
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
                self._dicAction[idxJoint] = raAction[idxJoint]
                pass
            return True
        else:
            if self._cntStep % REPEAT_ACTION == 0:
                self._flagAction = True
            return False
