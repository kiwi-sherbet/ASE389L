#! /usr/bin/python3

TORQUE = 0
POSITION = 1
VELOCITY = 2

# Define objects for the legged manipulator ##
## TODO: torque stuffs
class modelMobile():

	def __init__(self, clientBullet, xyzInit, qInit, flagFix=0):

		self.sim = clientBullet
		self.idRobot = self.sim.loadURDF("mobile.urdf", basePosition = xyzInit, baseOrientation = qInit, useFixedBase=flagFix)

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


	def getCurrentBodyPose(self):

		## Mode: True for joint position and False for joint velocity

		raVal = []

		state = self.sim.getBasePositionAndOrientation(self.idRobot)
		
		raVal.append(state[0])
		raVal.append(state[1])

		return raVal


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

		raVal = []

		if mode == POSITION:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveRight)[0])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveLeft)[0])

		elif mode == VELOCITY:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveRight)[1])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveLeft)[1])

		else:
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveRight)[2][5])
			raVal.append(self.sim.getJointState(self.idRobot, self.jntActiveLeft)[2][5])

		return raVal

