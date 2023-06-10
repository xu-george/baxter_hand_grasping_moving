# a simple code to test load urdf model on pybullet
import numpy as np
import pybullet
import pybullet as p
import pybullet_data
import os
import time

# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
pb_path = pybullet_data.getDataPath()
hand_path = os.path.join(current_path, "models/baxter_hand/left_wide.urdf")


class BaxterHand:
    """
    load baxter hand model and control it    
    """
    def __init__(self, hand_id, velocity=0.35, init_pos=[0, 0, 1], init_orn=[0, 1, 0, 1], work_limit="None") -> None:
        """
        :param hand_id: the id of the hand model
        :param velocity: the max velocity of the hand
        :param os_min: the min position of the end effector
        :param os_max: the max position of the end effector
        """
        # hand id in pybullet(loading urdf model)      
        self.handId = hand_id         
        self.maxVelocity = velocity  

        # define the end effector initial position, gripper state，
        self.ini_pos = init_pos
        self.ini_orn = init_orn

        #create control dict
        self.numJoints = p.getNumJoints(self.handId)
        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.handId, i)
            #print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)
        
        # finger information
        self.base_id = 0
        self.fingerForce = 10
        self.left_gripper_base = 0
        self.endEffector_id = 1        
        self.finger_tips_a_id = 3
        self.finger_tips_b_id = 5        
        self.finger_a_id = 2
        self.finger_b_id = 4

        # state value of gripper
        self.gripper_open = 0.020833
        self.gripper_close = 0    

        # set the work space limit
        self.os_min = work_limit[0]
        self.os_max = work_limit[1]   

        self.reset()    

    def reset(self):
        """
        reset the robot state with desired end_effector position and angel, gripper state
        :param pose_angle:
        :return:
        """
        # reset base position
        p.resetBasePositionAndOrientation(self.handId, self.ini_pos, self.ini_orn)

        # set gripper state --  close state
        p.resetJointState(self.handId, self.finger_a_id, self.gripper_close)
        p.resetJointState(self.handId, self.finger_b_id, self.gripper_close)
    
    # operation space control
    def osc(self, motorCommands, constrainId, space_limit=True):
        """
        :param motorCommands: the velocity and the angel of the end effector
        :return:
        """
        # ------------- position control version --------------
        # get current position and angel
        position, orientation = p.getBasePositionAndOrientation(self.handId)    

        # update position and angel
        position=np.array(position) + np.array(motorCommands[0])

        if space_limit:
            position = np.clip(position, self.os_min, self.os_max)

        # ------------------------------ update angel ------------------------------
        # get the rotation from initial to current
        rotation2 = p.getQuaternionFromEuler([0, 0, motorCommands[1]])
        new_orientation = p.multiplyTransforms([0, 0, 0], rotation2, [0, 0, 0], orientation)[1] 
        p.changeConstraint(constrainId, position, new_orientation, maxForce=100)        

    def gripper_control(self, gripper_state):
        """
        the code to control gripper
        :param gripper_state: [-1, 1]
        0 to 0.020833
        """
        # the state of the gripper
        gripper_state = (gripper_state + 1) * 0.5
        gripper_state_a = np.clip(gripper_state * self.gripper_open, self.gripper_close, self.gripper_open)

        gripper_state_b = np.clip(-gripper_state * self.gripper_open, -self.gripper_open, self.gripper_close)

        p.setJointMotorControl2(bodyUniqueId=self.handId,
                                jointIndex=self.finger_a_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_state_a,
                                force=self.fingerForce)
        p.setJointMotorControl2(bodyUniqueId=self.handId,
                                jointIndex=self.finger_b_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_state_b,
                                force=self.fingerForce)
        
    def gripper_close_open(self, gripper_command):
        """
        use command to close or open the gripper

        """
        if gripper_command == "close":
            target_position = self.gripper_close
        elif gripper_command == "open":
            target_position = self.gripper_open
        else:
            target_position = p.getJointState(self.handId, self.finger_a_id)[0]

        p.setJointMotorControl2(bodyUniqueId=self.handId,
                                jointIndex=self.finger_a_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position,
                                force=self.fingerForce)
        p.setJointMotorControl2(bodyUniqueId=self.handId,
                                jointIndex=self.finger_b_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=-target_position,
                                force=self.fingerForce)