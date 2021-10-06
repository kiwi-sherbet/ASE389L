#! /usr/bin/python3

from constants import *


# Define objects for the legged manipulator ##
## TODO: torque stuffs
class modelMobile():

	def __init__(self, clientBullet, xyzInit, qInit, flagFix=0):

		self.sim = clientBullet

		self.raPosXYZInit = xyzInit
		self.raPosQInit = qInit

		self.idRobot = self.sim.loadURDF(	SUBPATH_ROBOT["test"], 
											basePosition = self.raPosXYZInit, baseOrientation = self.raPosQInit, 
											useFixedBase=0)

		numJoints = self.sim.getNumJoints(self.idRobot)
		jointNameToID = {}

		for idJoint in range(numJoints):

			jointInfo = self.sim.getJointInfo(self.idRobot, idJoint)
			jointNameToID[jointInfo[1].decode('UTF-8')] = jointInfo[0]

			self.sim.setJointMotorControl2(self.idRobot, idJoint, self.sim.VELOCITY_CONTROL, targetVelocity = 0, force= 0)
			self.sim.enableJointForceTorqueSensor(self.idRobot, idJoint, enableSensor = 1)

		self.jntActiveRight = jointNameToID['jointActiveRight']
		self.jntActiveLeft = jointNameToID['jointActiveLeft']

		return


	def initCurrentBody(self, raPosXYZ, raPosQ):

		self.sim.resetBasePositionAndOrientation(self.idRobot, raPosXYZ, raPosQ)
		

	def initCurrentWheeilJoints(self, raVal):

		self.sim.resetJointState(self.idRobot, self.jntActiveRight, targetValue= raVal[0], targetVelocity=0)
		self.sim.resetJointState(self.idRobot, self.jntActiveLeft, targetValue= raVal[1], targetVelocity=0)
		

	def getCurrentBodyPose(self):

		## Mode: True for joint position and False for joint velocity
		
		return self.sim.getBasePositionAndOrientation(self.idRobot)


	def setTargetWheelJoints(self, mode, raVal):

		## Mode: True for joint position and False for joint velocity

		if mode == POSITION:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveRight, self.sim.POSITION_CONTROL, targetPosition=raVal[0], force=100)
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveLeft, self.sim.POSITION_CONTROL, targetPosition=raVal[0], force=100)

		elif mode == VELOCITY:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveRight, self.sim.VELOCITY_CONTROL, targetVelocity=raVal[0], force=100)
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveLeft, self.sim.VELOCITY_CONTROL, targetVelocity=raVal[0], force=100)

		else:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveRight, self.sim.TORQUE_CONTROL, force=raVal[0])
			self.sim.setJointMotorControl2(self.idRobot, self.jntActiveLeft, self.sim.TORQUE_CONTROL, force=raVal[0])


	def getCurrentWheelJoints(self, mode):

		## Mode: True for joint position and False for joint velocity

		if mode == POSITION:
			return (self.sim.getJointState(self.idRobot, self.jntActiveRight)[0],
					self.sim.getJointState(self.idRobot, self.jntActiveLeft)[0])

		elif mode == VELOCITY:
			return (self.sim.getJointState(self.idRobot, self.jntActiveRight)[1],
					self.sim.getJointState(self.idRobot, self.jntActiveLeft)[1])

		else:
			return (self.sim.getJointState(self.idRobot, self.jntActiveRight)[2][5],
					self.sim.getJointState(self.idRobot, self.jntActiveLeft)[2][5])
