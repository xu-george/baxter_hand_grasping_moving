import os 
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data
import sys
import random
import time
import operator

# add the path to os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hand_bullet import BaxterHand
from baxter_hand_gym import HandGymEnv
from conver_world.conver import Conveyor

# A global variable to segment one step of the simulation
update_freq = 240

"""
In this code, we use fake trajectory prediction
"""

# define some basic trajectories
def oval_traj(convId, speed=1):    
    start_angle = -np.pi/2
    centre_point = [0, 0, 0.745]
    velocity=speed*2.5e-2/update_freq
    radius_x = 0.15
    radius_y = 0.3
    convery = Conveyor(conveyor_id=convId, velocity=velocity, start_angle=start_angle, centre_point=centre_point,
                       radius_x=radius_x, radius_y=radius_y, traj_type="oval")
    return convery

def circle_traj(convId):
    start_angle = -np.pi/2
    centre_point = [0, 0, 0.745]
    velocity=0.8*2.5e-2/update_freq  # the change of angle per step
    radius = 0.2
    convery = Conveyor(conveyor_id=convId, velocity=velocity, start_angle=start_angle, centre_point=centre_point,
                       radius=radius, traj_type="circle")
    return convery

def sin_traj(convId, speed=1):
    StartPos = [0, -0.4, 0.745]    
    frequency = 2.5*np.pi
    radius = 0.2
    velocity = np.array([0, speed * 5e-3/update_freq, 0])   # the velocity across the y axis
    conver = Conveyor(conveyor_id=convId, velocity=velocity, init_pos=StartPos, frequency=frequency, 
                       radius=radius, traj_type="sinousoid")
    return conver

def line_traj(convId, speed=1):
    StartPos = [0, -0.4, 0.745]
    end_point = [0, 0.4, 0.745]
    StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    velocity = np.array([0, speed * 2*5e-3/update_freq, 0])   # the velocity across the y axis
    conver = Conveyor(conveyor_id=convId, velocity=velocity, init_pos=StartPos, end_pos=end_point, 
                      init_orn=StartOrientation, traj_type="line")
    return conver  

# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
pb_path = pybullet_data.getDataPath()
hand_path = os.path.join(current_path, "models/baxter_hand/left_wide.urdf")
table_path = os.path.join(current_path, "models/objects/table/table.urdf")
retangle_path = os.path.join(current_path, "models/objects/block/retangle.urdf")
cube_path = os.path.join(current_path, "models/objects/block/model.urdf")
conveyor_path = os.path.join(current_path, "models/objects/block/covery.urdf")

# inherit HandGymEnv
class DyGrasping(HandGymEnv):
    def __init__(self, max_episode_steps=100,real_time=False, renders=True, reward_type="sparse", control_model="p_o", traj="line", predict=False, speed=1) -> None:
        """
        :param max_episode_steps: the maximum number of steps in one episode
        :param real_time: whether to run the simulation in real time
        :param renders: whether to render the simulation
        :param reward_type: the type of reward, "sparse" or "dense"
        :param control_model: the control model, "p_o" or "v_o"
        :param dyna_obj: whether to use dynamic objects
        :param traj: the trajectory of the conveyor
        """
        self.traj = traj
        self.real_time = real_time
        self.speed = speed
        self.predict = predict
        super(DyGrasping, self).__init__(max_episode_steps=max_episode_steps, renders=renders, reward_type=reward_type, control_model=control_model)         


    def reset(self):
        self.terminated = 0
        self._envStepCounter = 0
        self._grasped_time = 0 

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        # ---------------load models---------------------#
        self.planeId = p.loadURDF(os.path.join(pb_path, "plane.urdf"), [0, 0, 0])
        self.table_id = p.loadURDF(table_path, [0, 0, 0], [0.0, 0.0, 0, 1])

        # load robot and control
        self.handId = p.loadURDF(hand_path, useFixedBase=False)  
        self.hand = BaxterHand(self.handId, init_pos=self.gripper_pos, 
                               init_orn=self.gripper_orn, work_limit=[self.os_low, self.os_high])
        self.base_contraint = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, 
                                                 [0, 0, 0], [0, 0, 0], self.gripper_pos, [0, 0, 0, 1], self.gripper_orn)
        
        # ----------- load conveyor ----------------
        self.convId = p.loadURDF(conveyor_path, [0, -0.4, 0.745], [0, 0, 0, 1])
        if self.traj == "circle":
            self.convey = circle_traj(self.convId)
        elif self.traj == "oval":
            self.convey = oval_traj(self.convId, speed=self.speed)
        elif self.traj == "line":
            self.convey = line_traj(self.convId, speed=self.speed)
        elif self.traj == "sin":
            self.convey = sin_traj(self.convId, speed=self.speed)
        else:
            print("No such trajectory")
        self.convey.traj.draw() 
        self.convey.reset()

        # load object
        obj_pose, _ = p.getBasePositionAndOrientation(self.convId)          
        ang = np.pi * random.random() / 4        
        obj_orn = p.getQuaternionFromEuler([0, 0, ang])

        flags = p.URDF_USE_INERTIA_FROM_FILE
        if self._control_model == "p":
            self.obj_id = p.loadURDF(cube_path, obj_pose, obj_orn, flags=flags)
        else:
            self.obj_id = p.loadURDF(retangle_path, obj_pose, obj_orn, flags=flags)

        # ------------------------set the dynamic of the obejcts------------------------#
        p.changeDynamics(self.obj_id, -1, lateralFriction=1, spinningFriction=0.001, rollingFriction=0.001, restitution=0)       
        p.changeDynamics(self.table_id, -1, lateralFriction=1, restitution=1)
        p.changeDynamics(self.handId, self.hand.finger_tips_a_id, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1, restitution=0.01)
        p.changeDynamics(self.handId, self.hand.finger_tips_b_id, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1, restitution=0.01)

        for _ in range(int(self.control_time/self._timeStep)*10):
            p.stepSimulation()     

        # get object inital position
        cube_pose, _ = p.getBasePositionAndOrientation(self.obj_id)
        self.cube_init_z = cube_pose[2]
        return self.get_physic_state() 
    
    def get_physic_state(self):
        # get physic state
        """
        The physic state contain:
        1. joint cos ().  3.end_effector pose
        4. normalized gripper_state. 5.gripper to cube.
        """        
        motor_id = self.hand.motorIndices
        joint_states = p.getJointStates(self.handId, motor_id) 

        cube_pose, cube_orn = p.getBasePositionAndOrientation(self.obj_id)
        # plot the cube frame
        self.draw_frame(cube_pose, cube_orn, 0.5)
        # get euler angle from quaternion
        cube_orn = list(p.getEulerFromQuaternion(cube_orn))

        if self.predict and (not self._touched()):
            pre_pose, pre_theta = self.predict_object()
            cube_pose = pre_pose            
            cube_orn[2] = pre_theta            
            
            # update the dot position      
            self.draw_frame(cube_pose, p.getQuaternionFromEuler([0, 0, cube_orn[2]]), 1, 0.05)

        # get end effector state
        end_effector_p, tran_orn = self.get_end_state()

        # get joint state -- normalized to [-1, 1]
        gripper_state = 2 * (- joint_states[-1][0] / self.hand.gripper_open) - 1
        dist = np.linalg.norm(np.array(end_effector_p) - np.array(cube_pose))

        if self._control_model == "p_o":
            physic_state = np.concatenate((end_effector_p, cube_pose, [gripper_state], [dist], [tran_orn[2]], [cube_orn[2]]), axis=0)
        elif self._control_model == "p":
            physic_state = np.concatenate((end_effector_p, cube_pose, [gripper_state], [dist]), axis=0)
        elif self._control_model == "o":
            physic_state = np.array([tran_orn[2], cube_orn[2]])      
        return physic_state 

    def step(self, action):
        # take the action
        self._envStepCounter += 1

        if self._control_model == "p_o":
            d_p = action[0:3] * self.v_end
            d_r = action[3] * self.r_end
            d_gripper = action[4] * self.v_gripper
        elif self._control_model == "p":   
            d_p = action[0:3] * self.v_end
            d_r = 0
            d_gripper = action[3] * self.v_gripper
        
        # TODO: segment the action into several steps 
        self.hand.gripper_control(d_gripper)
        #self.hand.osc([d_p, d_r], self.base_contraint)
        
        update_fre = 240
        d_p, d_r = d_p / update_fre, d_r / update_fre
        
        for _ in range(update_fre):
            self.convey.step()
            self.hand.osc([d_p, d_r], self.base_contraint)            
            p.stepSimulation()          

        # update obs
        obs = self.get_physic_state() 

        # update terminated and truncated        
        truncated = (self._envStepCounter >= self.max_episode_steps)

        # judge if fall down from convex
        if self.predict:
            on_convex = (p.getContactPoints(self.convId, self.obj_id) != ())
            if (not on_convex) and (not self._touched()):
                truncated = True

        # success or out of the workspace
        terminated = False #self._success()       

        # update reward
        reward = self._reward()
        # update info
        info = {}
        if self._success():
            info["success"] = 1
        else:
            info["success"] = 0 

        return obs, reward, terminated, truncated, info
    
    def predict_object(self):
        """
        predict the object position and orientation based on the distance between gripper and object
        """
        end_effector_p, tran_orn = self.get_end_state()
        cube_pose, cube_orn = p.getBasePositionAndOrientation(self.obj_id)
        dist = np.linalg.norm(np.array(end_effector_p) - np.array(cube_pose))
        step = int(32 * 240 * (1/(1+np.exp(-4.5*dist)) - 0.5))

        d_p, d_o = self.convey.traj.predict(step)

        # update the object position and orientation
        cube_pose = np.array(cube_pose) + d_p
        cube_orn = p.getEulerFromQuaternion(cube_orn)

        theta = cube_orn[2] + d_o        
        return cube_pose, theta
    
    def draw_frame(self, position, orientation, color=1, size=0.05):
        # Create the arrows for the frame
        arrow_length = size
        arrow_colors = color * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X: Red, Y: Green, Z: Blue
        orn = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        for i in range(3):
            # Calculate the direction of the arrow based on the orientation
            arrow_direction = orn[:, i]

            # Calculate the tip position based on the base position and orientation
            tip_position = position + arrow_length * arrow_direction

            # Draw the arrow
            arrow_id = p.addUserDebugLine(position, tip_position, lineColorRGB=arrow_colors[i], lineWidth=3,
                                          parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)        
