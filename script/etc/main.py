#! /usr/bin/python3

## Import basic modules ##
import time

## Import utility modules ##
import pybullet as p
import threading as t
from numpy import pi as PI
import numpy as np

import robot

## Import my modules ##

## System configuration ##
PATH_DATA="../data"

## Set up physical models
def setupWorld():

    ## Start simulation ##
    p.connect(p.GUI)
    p.setAdditionalSearchPath(PATH_DATA)
    # p.setRealTimeSimulation(1)

    num_bullet_solver_iterations = 100

    p.setPhysicsEngineParameter(
            numSolverIterations=num_bullet_solver_iterations)
    p.setTimeStep(0.01)
    p.setPhysicsEngineParameter(enableConeFriction=0)

    ## Set up playground
    p.setGravity(0, 0, -10)

    ## Set up terrains
    idTerrain = p.loadURDF("samurai/samurai.urdf", [0.0, 0.0, 0.0], useFixedBase=1)

    ## Set up robots
    modelRobot = robot.modelMobile(p, [0, 0, 0.5], [0, 0, 0, 1], flagFix=0)

    return modelRobot


def main():
    
    modelMotor = setupWorld()

    trj = [0.5 * np.sin(PI * idx/300) for idx in range (0, 600)]
    idx = 0

    while True:

        if idx < 600 -1:
            idx += 1
            if idx == 300:
                print("changed")
        else:
            idx = 0
            print("changed")

        modelMotor.setTargetWheelJoints(robot.TORQUE, [trj[idx], trj[idx]])
        p.stepSimulation()
        t.Event().wait(0.01)

    return

if __name__ == "__main__":
    main()

