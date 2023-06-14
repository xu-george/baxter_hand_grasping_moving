"""
a gym environment for baxter hand to grasping objects
"""
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

# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
pb_path = pybullet_data.getDataPath()
hand_path = os.path.join(current_path, "models/baxter_hand/left_wide.urdf")
table_path = os.path.join(current_path, "models/objects/table/table.urdf")
retangle_path = os.path.join(current_path, "models/objects/block/retangle.urdf")
cube_path = os.path.join(current_path, "models/objects/block/model.urdf")


class HandGymEnv(gym.Env):
    """
    A gym environment for baxter hand to grasping objects
    """
    def __init__(self, actionRepeat=1, max_episode_steps=100, predicts=False,
                 renders=True, reward_type="sparse", control_model="p_o") -> None:       
            
        """        
        actionRepeat: the number of repeat for each action        
        max_episode_steps: the maximum number of steps for each episode
        renders: whether render the environment
        reward_type: the type of reward, "sparse" or "dense"  
        control: the type of control, "p_o" or "o" or "p" : the position or the orientation or both
        traj: oval, circle, sinousoid, line
        """
        super(HandGymEnv, self).__init__()   
            
        self._renders = renders
        self._reward_types = reward_type

        self._control_model = control_model
        self._timeStep = 1./240.
        self.control_time = 1/20
        self._actionRepeat = actionRepeat
        self.urdfRootPath = pybullet_data.getDataPath()        
        self._max_episode_steps = max_episode_steps

        self._observation = []
        self.observation_space = {}        

        self.v_end = 0.05   # the velocity of end-effector
        self.r_end = 0.5   # the rotation of end-effector
        self.v_gripper = 2  # need at least 1 steps to close    

        # set the initial position and orientation of the gripper
        self.gripper_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        self.gripper_pos = [0, 0, 1]
        # define the workspace limit of the end effector
        self.os_low = np.array([-0.3, -0.4, 0.905])        
        self.os_high = np.array([0.3, 0.4, 1.2])

        self._envStepCounter = 0
        self.max_episode_steps = max_episode_steps

        self.terminated = 0
        self._p = p

        # if show GUI
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, 0.2, np.pi / 4.])
        else:
            p.connect(p.DIRECT)
        self.viewer = None

        self.seed()

        obs = self.reset()
        # get action space and observation space
        if self._control_model == "p_o":
            self.action_dim = 5
        elif self._control_model == "p":
            self.action_dim = 4
        elif self._control_model == "o":
            self.action_dim = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs.shape, dtype=np.float32)

    def action_spec(self):
        low = -1 * np.ones(self.action_dim)
        high = 1 * np.ones(self.action_dim)
        return low, high
    
    def  get_action_space(self):
        if self._control_model != "p_o":
            return self.action_space
        else:
            p_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            o_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            return p_space, o_space
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        
        # --------------------------- load models ---------------------------------
        self.planeId = p.loadURDF(os.path.join(pb_path, "plane.urdf"), [0, 0, 0])

        # ----------------- load table     
        self.table_id = p.loadURDF(table_path, [0, 0, 0], [0.0, 0.0, 0, 1])

        # -------------------------- load robot -----------------------
        self.handId = p.loadURDF(hand_path, useFixedBase=False)  
        
        self.hand = BaxterHand(self.handId, init_pos=self.gripper_pos, 
                               init_orn=self.gripper_orn, work_limit=[self.os_low, self.os_high])  
               
        # add constraint  
        self.base_contraint = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, 
                                                 [0, 0, 0], [0, 0, 0], self.gripper_pos, [0, 0, 0, 1], self.gripper_orn)      
        
        # --------------- load object ---------------
        if self._control_model == "o":
            obj_pose =  np.array([0, 0, 0.74])
        elif self._control_model != "o":
            obj_pose =  np.array([random.random() * 0.4 - 0.2, random.random() * 0.6 - 0.3, 0.74])            

        ang = 2 * np.pi * random.random() - np.pi
        obj_orn = list(p.getQuaternionFromEuler([0, 0, ang]))
        flags = p.URDF_USE_INERTIA_FROM_FILE

        if self._control_model == "p":
            self.obj_id = p.loadURDF(cube_path, obj_pose, obj_orn, flags=flags)
        else:
            self.obj_id = p.loadURDF(retangle_path, obj_pose, obj_orn, flags=flags)

        # -------------------set the dynamic of objects--------------------
        p.changeDynamics(self.obj_id, -1, lateralFriction=1, spinningFriction=0.001, rollingFriction=0.001, restitution=0)       
        p.changeDynamics(self.table_id, -1, lateralFriction=1, restitution=0)
        p.changeDynamics(self.handId, self.hand.finger_tips_a_id, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1, restitution=0)
        p.changeDynamics(self.handId, self.hand.finger_tips_b_id, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1, restitution=0)
        
        for _ in range(int(self.control_time/self._timeStep)*10):
            p.stepSimulation()           

        cube_pose, _ = p.getBasePositionAndOrientation(self.obj_id)
        self.cube_init_z = cube_pose[2]

        return self.get_physic_state()  
    
    def get_end_state(self):
        end_effector_p = p.getLinkState(self.hand.handId, self.hand.endEffector_id)[4]
        end_effector_o = p.getLinkState(self.hand.handId, self.hand.endEffector_id)[5] 

        # get the orientation from start to currernt                
        rotate = p.getDifferenceQuaternion(self.gripper_orn, end_effector_o)
        tran_orn = p.getEulerFromQuaternion(rotate)
        return end_effector_p, tran_orn

    
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
        # get euler angle from quaternion
        cube_orn = p.getEulerFromQuaternion(cube_orn)

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
    
    # ------------------------------------------------------------------------------
    # check whether the end effector is inside the workspace
    def _inside_workspace(self, pos):
        x, y, z = pos
        inside = (x > self.os_low[0]) and (x < self.os_high[0]) and \
                 (y > self.os_low[1]) and (y < self.os_high[1]) and \
                 (z > self.os_high[2]) and (z < self.os_high[2])
        return inside   
    
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
        elif self._control_model == "o":
            d_p = np.zeros(3)
            d_r = action[0] * self.r_end
            d_gripper = -1    

        update_fre = int(self.control_time/self._timeStep)
        self.hand.gripper_control(d_gripper)
        self.hand.osc([d_p, d_r], self.base_contraint)    
        
        for _ in range(update_fre*2):
            p.stepSimulation()    

        # update obs
        obs = self.get_physic_state() 

        # update terminated and truncated        
        truncated = (self._envStepCounter >= self.max_episode_steps) # and (not self._success())

        # success or out of the workspace
        terminated = False #self._success()       

        # update reward
        reward = self._reward()
        # update info
        info = {}
        if self._success():
            info["success"] = "True"
        else:
            info["success"] = "False"    

        return obs, reward, terminated, truncated, info
    
    def _success(self):
        cube_pose, _ = p.getBasePositionAndOrientation(self.obj_id)
        return (cube_pose[2] - self.cube_init_z) > 0.03 and self._grasped()        
    
    def _reward(self):
        reward = 0
        orn_reward = 0
        # lift reward
        cube_pose, cube_orn = p.getBasePositionAndOrientation(self.obj_id)
        cube_orn = p.getEulerFromQuaternion(cube_orn)
        end_pose, end_orn = self.get_end_state()

        if self._success():
           reward += 2.25

        if self._reward_types == "dense":

            # reaching reward    
            dist = np.linalg.norm(np.array(end_pose) - np.array(cube_pose))
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward       

            # grasped reward
            if self._grasped():
                reward += 0.25

        # orentation reward
        if self._reward_types == "dense" and (self._control_model == "o" or "p_o"):
            orn_reward = 1 - np.tanh(abs(cube_orn[2] - end_orn[2]))
        
        if self._control_model == "o":
            return orn_reward
        elif self._control_model == "p_o":
            return (reward + orn_reward)/4.5
        else:           
            return reward / 3.5
    
    def _grasped(self):
        """
        check the whether block is grasped
        :return:
        """
        left_contact = p.getContactPoints(self.handId, self.obj_id, self.hand.finger_tips_a_id)
        right_contact = p.getContactPoints(self.handId, self.obj_id, self.hand.finger_tips_b_id)
        return left_contact != () and right_contact != ()
    
    def _touched(self):
        """
        check the whether block is grasped
        :return:
        """
        left_contact = p.getContactPoints(self.handId, self.obj_id, self.hand.finger_tips_a_id)
        right_contact = p.getContactPoints(self.handId, self.obj_id, self.hand.finger_tips_b_id)
        return left_contact != () or right_contact != ()

    def __del__(self):
        p.disconnect()


if __name__ == "__main__":
    env = HandGymEnv(renders=True, max_episode_steps=100, control_model="p", reward_type="dense")

    # random action
    low, high = env.action_spec()

    # check the step function
    obs = env.reset()    
    action = np.array([1, 0, 0, 0])
    new_obs, reward, terminated, truncated, info = env.step(action)
    print("obs change: ", new_obs - obs)
    obs = new_obs

    # check the gripper 
    obs = env.reset()    
    action = np.array([0, 0, 0, 1])
    new_obs, reward, terminated, truncated, info = env.step(action)
    print("obs change: ", new_obs - obs)
    obs = new_obs

    # # check the rotation
    # action = np.array([0, 0, 0, 1, 0])
    # new_obs, reward, terminated, truncated, info = env.step(action)
    # print("obs change: ", new_obs - obs)
    # obs = new_obs
   
    # do visualization
    for i in range(10000):
        # get a random action         
        # action = np.random.uniform(low, high) 
        action = np.array([1])
        print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs: ", obs)  
        # delay 0.01s
        time.sleep(0.01)              
        if (i+1) % 100 == 0 or truncated:
            env.reset()