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


	def getBaseRollPitchYaw(self):
		"""Get minitaur's base orientation in euler angle in the world frame.

		Returns:
		  A tuple (roll, pitch, yaw) of the base in world frame.
		"""
		_, orientation = self.getCurrentBodyPose()
		roll_pitch_yaw = self.sim.getEulerFromQuaternion(orientation)
		return roll_pitch_yaw


	def getBaseVelocity(self):
		linear_velocity = self.sim.getBaseVelocity(self.idRobot)[0]
		_, orientation = self.getCurrentBodyPose()
		_, orientation_inversed = self.sim.invertTransform(
			[0, 0, 0], orientation)
		relative_velocity, _ = self.sim.multiplyTransforms(
			[0, 0, 0], orientation_inversed, linear_velocity,
			self.sim.getQuaternionFromEuler([0, 0, 0]))
		return relative_velocity


	def getBaseRollPitchYawRate(self):
		angular_velocity = self.sim.getBaseVelocity(self.idRobot)[1]
		_, orientation = self.getCurrentBodyPose()
		_, orientation_inversed = self.sim.invertTransform(
			[0, 0, 0], orientation)
		relative_angular_velocity, _ = self.sim.multiplyTransforms(
			[0, 0, 0], orientation_inversed, angular_velocity,
			self.sim.getQuaternionFromEuler([0, 0, 0]))
		return relative_angular_velocity


	def getTargetToLocalFrame(self, position):
		position_base, orientation_base = self.getCurrentBodyPose()
		position_base_inversed, orientation_base_inversed = self.sim.invertTransform(
			position_base, orientation_base)
		relative_position, _ = self.sim.multiplyTransforms(
			position_base_inversed, orientation_base_inversed, position, self.sim.getQuaternionFromEuler([0, 0, 0]))
		return relative_position


	def checkFlipped(self):
		position_base, orientation_base = self.getCurrentBodyPose()
		rot_mat = self.sim.getMatrixFromQuaternion(orientation_base)
		return rot_mat[-1] < 0.5