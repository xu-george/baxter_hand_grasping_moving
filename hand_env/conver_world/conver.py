import pybullet as p
import pybullet_data
import math
import numpy as np
from hand_env.conver_world.trajectory_generator import line_generator, sinousoid_generator, circle_generator, oval_generator
import time

# get current directory and add to path
import sys
sys.path.append("..")

# get current path and parent path
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.dirname(cur_path))
conveyor_path = os.path.join(parent_path, "models/objects/block/covery.urdf")

# define a converyor class
class Conveyor:    
    def __init__(self, conveyor_id, velocity, init_pos=[0, 0, 1], end_pos="None", init_orn=[0, 0, 0, 1], 
                 traj_type="line", frequency=0.1, start_angle="None", centre_point="None", 
                 radius="None", radius_x="None", radius_y="None") -> None:
        """
        :param conveyor_id: the id of the conveyor model
        :param init_pos: the initial position of the conveyor
        :param init_orn: the initial orientation of the conveyor
        """        
        self.conveyorId = conveyor_id    
        self.ini_pos = init_pos
        self.end_pos = end_pos
        self.ini_orn = init_orn  
        self.traj_type = traj_type
        self.velocity = velocity
                
        # periodic trajectory parameters
        self.start_angle = start_angle
        self.frequency = frequency
        self.centre_point = centre_point
        self.radius = radius
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.convery_width = 0.2

        # define the trajectory generator
        if self.traj_type == "line":
           self.traj = line_generator(start_point=self.ini_pos, end_point=self.end_pos, velocity=self.velocity, line_color=[1, 0, 0])
        elif self.traj_type == "sinousoid":
            self.traj = sinousoid_generator(start_point=self.ini_pos, velocity=self.velocity, line_color=[1, 0, 0], 
                                            plot_points=50, amplitude=self.radius, frequency=self.frequency)
        elif self.traj_type == "circle":                     
            self.traj = circle_generator(start_angle=self.start_angle, velocity=self.velocity, centre_point=self.centre_point, line_color=[1, 0, 0], 
                                         radius=self.radius, lineWidth=3)
        elif self.traj_type == "oval":
            start_angle, centre_point, velocity, radius_x, radius_y,
            self.traj = oval_generator(start_angle=self.start_angle, velocity=self.velocity, centre_point=self.centre_point, 
                                       radius_x=self.radius_x, radius_y=self.radius_y,line_color=[1, 0, 0])      

        self.reset()

    # reset the conveyor
    def reset(self):
        [pos, orn] = self.traj.reset()
        # self.conv_constrain = p.createConstraint(self.conveyorId, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1], orn)
        p.resetBasePositionAndOrientation(self.conveyorId, pos, orn)       

    def step(self):
        [pos, orn] = self.traj.step()
        #p.changeConstraint(self.conv_constrain, pos, orn, maxForce=100) 
        p.resetBasePositionAndOrientation(self.conveyorId, pos, orn)


# create a main function to test all the trajectory
if __name__ == "__main__":
    # setup pybullet simulation
    # Connect to the physics server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set up the simulation parameters
    p.setGravity(0, 0, -10)
    timeStep = 1.0 / 240.0
    p.setTimeStep(timeStep)

    # Load the plane
    planeId = p.loadURDF("plane.urdf")


    StartPos = [0, 0, 1]
    end_point = [1, 0, 1]
    StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    convId = p.loadURDF(conveyor_path, StartPos, StartOrientation)

    # # -----------------------test the line trajectory ------------------------------------------
    # conver = Conveyor(conveyor_id=convId, velocity=np.array([0.05/(240*60), 0, 0]), init_pos=StartPos, end_pos=[1, 0, 1], 
    #                   init_orn=StartOrientation, traj_type="line")
    # conver.traj.draw()
    
    # conver.reset()
    # for i in range(1000):
    #     conver.step()
    #     p.stepSimulation()
    #     time.sleep(timeStep)

    # # ----------------------- test the sinousoid trajectory -------------------------------------    
    conver2 = Conveyor(conveyor_id=convId, velocity=np.array([0.001, 0, 0]), init_pos=StartPos, frequency=2*math.pi, 
                       radius=0.5, traj_type="sinousoid")
    conver2.traj.draw()
    conver2.reset()
    for i in range(1000):
        conver2.step()
        p.stepSimulation()
        time.sleep(timeStep)

    # ----------------------- test the circle trajectory -------------------------------------
    # start_angle = 0
    # centre_point = [0, 0, 1]
    # velocity=np.pi/500
    # radius = 0.5
    # conver3 = Conveyor(conveyor_id=convId, velocity=velocity, start_angle=start_angle, centre_point=centre_point,
    #                    radius=radius, traj_type="circle")
    # conver3.traj.draw()
    # conver3.reset()
    # for i in range(1000):
    #     conver3.step()
    #     p.stepSimulation()
    #     time.sleep(timeStep)

    # # ----------------------- test the oval trajectory -------------------------------------
    # start_angle = 0
    # centre_point = [0, 0, 1]
    # velocity=np.pi/500
    # radius_x = 0.4
    # radius_y = 0.2
    # conver3 = Conveyor(conveyor_id=convId, velocity=velocity, start_angle=start_angle, centre_point=centre_point,
    #                    radius_x=radius_x, radius_y=radius_y, traj_type="oval")
    # conver3.traj.draw()
    # conver3.reset()
    # for i in range(1000):
    #     conver3.step()
    #     p.stepSimulation()
    #     time.sleep(timeStep)