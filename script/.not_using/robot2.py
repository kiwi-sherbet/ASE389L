#! /usr/bin/python3

from constants import *


# Define objects for the legged manipulator ##
class modelRobot():

	def __init__(self, clientBullet, xyzInit, qInit):

		self.sim = clientBullet

		self.raPosXYZInit = xyzInit
		self.raPosQInit = qInit

		self.idRobot = self.sim.loadURDF(	SUBPATH_ROBOT["test"], 
											basePosition = self.raPosXYZInit, baseOrientation = self.raPosQInit, 
											useFixedBase=1)

		self.setupRobot()


	def setupRobot(self):

		dicJointNameToID = {}

		for idJoint in range(self.sim.getNumJoints(self.idRobot)):

			self.sim.setJointMotorControl2(self.idRobot, idJoint, self.sim.VELOCITY_CONTROL, targetVelocity = 0, force= 0)
			self.sim.enableJointForceTorqueSensor(self.idRobot, idJoint, enableSensor = 1)

			jointInfo = self.sim.getJointInfo(self.idRobot, idJoint)
			dicJointNameToID[jointInfo[1].decode('UTF-8')] = jointInfo[0]


		self.jntActive1 = dicJointNameToID["joint1"]
		self.jntActive2 = dicJointNameToID["joint2"]
		self.jntToeFR = dicJointNameToID["jointToe"] 

		self.raGainP = (0, 0, 0)
		self.raGainD = (0, 0, 0)

		return


	def getCurrentBodyPose(self):

		raXYZ, raQ = self.sim.getBasePositionAndOrientation(self.idRobot)
		raAng = self.sim.getEulerFromQuaternion(raQ)

		return raXYZ, raAng


	def getCurrentBodyVelocity(self):

		raXYZ, raAng = self.sim.getBaseVelocity(self.idRobot)

		return raXYZ, raAng

		
	def getCurrentBodyRate(self):

		raVelXYZ, raVelAng = self.getCurrentBodyVelocity()
		raPosAng = self.getCurrentBodyPose()[1]

		qInvRot = self.sim.invertTransform([0, 0, 0], self.sim.getQuaternionFromEuler(raPosAng))[1]

		raVelXYZ = self.sim.multiplyTransforms(	[0, 0, 0], qInvRot, raVelXYZ, self.sim.getQuaternionFromEuler([0, 0, 0]))[0]
		raVelAng = self.sim.multiplyTransforms(	[0, 0, 0], qInvRot, raVelAng, self.sim.getQuaternionFromEuler([0, 0, 0]))[0]

		return raVelXYZ, raVelAng


	def initCurrentBody(self, raPosXYZ, raPosQ):

		self.sim.resetBasePositionAndOrientation(self.idRobot, raPosXYZ, raPosQ)


	def initCurrentJoints(self, raVal):

		self.sim.resetJointState(self.idRobot, self.jntActive2, targetValue= raVal[0], targetVelocity=0)
		self.sim.resetJointState(self.idRobot, self.jntActive2, targetValue= raVal[1], targetVelocity=0)


	def getCurrentJoints(self, mode):

		## Mode: True for joint position and False for joint velocity

		raVal = []

		if mode == POSITION:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive1)[0])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive2)[0])

		elif mode == VELOCITY:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive1)[1])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive2)[1])

		else:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive1)[2][5])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActive2)[2][3])

		return raVal


	def setTargetJoints(self, mode, raVal):

		## Mode: True for joint position and False for joint velocity

		if mode == POSITION:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive1, self.sim.POSITION_CONTROL, targetPosition= raVal[0], force=30)
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive2, self.sim.POSITION_CONTROL, targetPosition= raVal[1], force=30)

		elif mode == VELOCITY:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive1, self.sim.VELOCITY_CONTROL, targetVelocity= raVal[0], force=30)
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive2, self.sim.VELOCITY_CONTROL, targetVelocity= raVal[1], force=30)

		else:
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive1, self.sim.TORQUE_CONTROL, force= raVal[0])
			self.sim.setJointMotorControl2(self.idRobot, self.jntActive2, self.sim.TORQUE_CONTROL, force= raVal[1])

		return


	def getFootContacts(self):

		raPoint = self.sim.getContactPoints(bodyA=self.idRobot)
		raLinkToe = [self.jntToe]
		raContact = [False]

		for point in raPoint:
			# Ignore self contacts
			if point[2] == self.idRobot:
				continue
			try:
				idxContact = raLinkToe.index(point[3])
				raContact[idxContact] = True
			except ValueError:
				continue
		
		return raContact