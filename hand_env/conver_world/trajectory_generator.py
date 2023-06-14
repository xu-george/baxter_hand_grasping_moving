import numpy as np
import pybullet as p
import pybullet_data
import math
import time 

"""
declaim: In this simple version, we just use fake prediction (The ground truth of trajectory is known), 
while for the real robot, we use lstm to predict the trajectory
"""

# line generator
class line_generator:
    def __init__(self, start_point, end_point, velocity, line_color, lineWidth=3) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.line_color = line_color
        self.lineWidth = lineWidth
        self.velocity = velocity
        self.orn = [0, 0, 0, 1]
        self.reset()

    def draw(self):
        p.addUserDebugLine(self.start_point, self.end_point, self.line_color, lineWidth=self.lineWidth)

    def reset(self):        
        self.step_time = 0
        self.position = np.array(self.start_point)
        self.orn = [0, 0, 0, 1]
        return [self.position, self.orn]       

    def step(self):
        self.step_time += 1
        self.position += self.velocity
        return [self.position, self.orn]
    
    def predict(self, steps):
        """
        steps: the number of steps to predict
        return:predict the change of position and orientation
        """        
        pre_position = self.velocity * steps      
        return [pre_position, 0]
    

# sinousoid generator
class sinousoid_generator:
    def __init__(self, start_point, velocity, line_color, frequency, plot_points,
                 amplitude, lineWidth=3) -> None:
        """
        :param start_point: the start point of the sinousoid
        :param velocity: the velocity of the sinousoid
        :param line_color: the color of the sinousoid
        :param frequency: the frequency of the sinousoid
        :param plot_points: the number of points to plot
        :param amplitude: the amplitude of the sinousoid
        :param lineWidth: the width of the line        
        """
        self.start_point = start_point     
        self.frequency = frequency
        self.velocity = velocity
        self.orn = [0, 0, 0, 1]

        self.line_color = line_color
        self.lineWidth = lineWidth
        self.plot_points = plot_points
        self.amplitude = amplitude

        self.reset()

    def draw(self):
        curve_points = []
        for i in range(self.plot_points):
            y = 840 * i * self.velocity[1] + self.start_point[1] 
            x = self.amplitude * math.cos(self.frequency * y) + self.start_point[0]                       
            z = self.start_point[2]
            curve_points.append([x, y, z])

        for i in range(self.plot_points - 1):
            p.addUserDebugLine(curve_points[i], curve_points[i+1], self.line_color, lineWidth=3)

    def reset(self):
        # set the initial position = [x, y, z]
        self.position = self.start_point
        self.time_step = 0
        # set the initial orientation
        tangent_vector = np.array([1, self.amplitude * self.frequency * math.cos(self.frequency * self.start_point[0]), 0])
        tangent_vector /= np.linalg.norm(tangent_vector) 
        self.angle = math.atan2(tangent_vector[1], tangent_vector[0])      
        self.orn = p.getQuaternionFromEuler([0, 0,self.angle])
        return [self.position, self.orn]  

    def step(self):
        self.time_step += 1
        # updata the position   
        y = self.start_point[1] + self.time_step* self.velocity[1]     
        x = self.amplitude * math.sin(self.frequency * y) + self.start_point[0]        
        z = self.start_point[2]
        self.position = np.array([x, y, z])

        # updata the orientation
        tangent_vector = np.array([1, self.amplitude * self.frequency * math.cos(self.frequency * self.position[0]), 0])
        tangent_vector /= np.linalg.norm(tangent_vector)
        self.angle = math.atan2(tangent_vector[1], tangent_vector[0])
        self.orn = p.getQuaternionFromEuler([0, 0, self.angle])   

        return [self.position, self.orn]
    
    def predict(self, steps):
        """
        give the preiction step, return the change of position and orientation
        """

        y = self.start_point[1] + (self.time_step+steps)* self.velocity[1]
        x = self.amplitude * math.sin(self.frequency * y) + self.start_point[0]
        z = self.start_point[2]
        d_p = np.array([x, y, z]) - self.position
        tangent_vector = np.array([1, self.amplitude * self.frequency * math.cos(self.frequency * x), 0])
        new_theta = math.atan2(tangent_vector[1], tangent_vector[0])        
        return [d_p, new_theta- self.angle]


# circle generator
class circle_generator:
    def __init__(self, start_angle, centre_point, velocity, radius, line_color, lineWidth=3) -> None:
        """
        :param start_anglele: the start anglele of the circle
        :param velocity: the velocity of the circle
        :param amplitude: the amplitude of the circle
        :radius: the radius of the circle
        :param line_color: the color of the circle
        :param lineWidth: the width of the line        
        """
        self.start_angle = start_angle 
        self.centre_point = centre_point
        self.velocity = velocity
        self.radius = radius        

        self.line_color = line_color
        self.lineWidth = lineWidth

        self.reset()

    def draw(self):
        curve_points = []
        for i in range(60):
            angle = 6 * i * math.pi / 180
            x = self.radius * math.cos(angle) + self.centre_point[0]
            y = self.radius * math.sin(angle) + self.centre_point[1]
            z = self.centre_point[2]
            curve_points.append([x, y, z])

        for i in range(60 - 1):
            p.addUserDebugLine(curve_points[i], curve_points[i+1], self.line_color, lineWidth=3)      

    def reset(self):
        self.time_step = 0
        self.angle = self.start_angle
        self.position = [self.radius * math.cos(self.angle) + self.centre_point[0],
                         self.radius * math.sin(self.angle) + self.centre_point[1],
                         self.centre_point[2]]
        
        
        tangent_vector = [math.cos(self.angle + math.pi / 2), math.sin(self.angle + math.pi / 2), 0]
        self.theta = math.atan2(tangent_vector[1], tangent_vector[0])
        self.orn = p.getQuaternionFromEuler([0, 0, self.theta])
        return [self.position, self.orn]
    
    def step(self):
        self.time_step += 1
        self.angle = self.start_angle + self.time_step * self.velocity
        self.position = [self.radius * math.cos(self.angle) + self.centre_point[0],
                         self.radius * math.sin(self.angle) + self.centre_point[1],
                         self.centre_point[2]]
        self.position = np.array(self.position)
        
        tangent_vector = [math.cos(self.angle + math.pi / 2), math.sin(self.angle + math.pi / 2), 0]
        self.theta = math.atan2(tangent_vector[1], tangent_vector[0])  # the rotation angle with z axis
        self.orn = np.array(p.getQuaternionFromEuler([0, 0, self.theta]))        
        return [self.position, self.orn]
    
    def predict(self, steps):
        """
        give the preiction step, return the change of position and orientation
        """
        angle = self.angle + steps * self.velocity
        position = [self.radius * math.cos(angle) + self.centre_point[0],
                    self.radius * math.sin(angle) + self.centre_point[1],
                    self.centre_point[2]]
        position = np.array(position)
        d_p = position - self.position

        tangent_vector = [math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2), 0]
        theta =math.atan2(tangent_vector[1], tangent_vector[0])        
        return [d_p, theta- self.theta]

# oval generator
class oval_generator:
    """
    generate the oval shape trajectory     
    """
    def __init__(self, start_angle, centre_point, velocity, radius_x, radius_y, line_color, lineWidth=3) -> None:
        """
        :param start_anglele: the start anglele of the oval
        :param velocity: the velocity of the oval
        :param amplitude: the amplitude of the oval
        :radius: the radius of the oval
        """
        self.start_anglele = start_angle
        self.centre_point = centre_point
        self.velocity = velocity
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.line_color = line_color
        self.lineWidth = lineWidth

        self.reset()
    
    def draw(self):
        circle_points = []
        for i in range(60):
            angle = 6 * i * math.pi / 180            
            x = self.centre_point[0] + self.radius_x * math.cos(angle)
            y = self.centre_point[1] + self.radius_y * math.sin(angle)
            z = self.centre_point[2]
            circle_points.append([x, y, z])

        for i in range(60 - 1):
            p.addUserDebugLine(circle_points[i], circle_points[i+1], self.line_color, lineWidth=3)

    def reset(self):
        self.time_step = 0
        self.angle = self.start_anglele
        self.position = [self.centre_point[0] + self.radius_x * math.cos(self.angle),
                         self.centre_point[1] + self.radius_y * math.sin(self.angle),
                         self.centre_point[2]]
        
        tangent_vector = [1, -self.radius_x * math.sin(self.start_anglele) / self.radius_y, 0]
        tangent_vector /= np.linalg.norm(tangent_vector)
        self.theta = math.atan2(tangent_vector[1], tangent_vector[0])
        self.orn = p.getQuaternionFromEuler([0, 0, self.theta])       

        return [self.position, self.orn]
    
    def step(self):
        self.time_step += 1
        self.angle = self.start_anglele + self.time_step * self.velocity
        self.position = [self.centre_point[0] + self.radius_x * math.cos(self.angle),
                         self.centre_point[1] + self.radius_y * math.sin(self.angle),
                         self.centre_point[2]]
        
        tangent_vector = [1, -self.radius_x * math.sin(self.angle) / self.radius_y, 0]
        tangent_vector /= np.linalg.norm(tangent_vector)
        self.theta = math.atan2(tangent_vector[1], tangent_vector[0])        
        self.orn = p.getQuaternionFromEuler([0, 0, self.theta])       

        return [self.position, self.orn]      

    def predict(self, steps):
        """
        give the preiction step, return the change of position and orientation
        """
        angle = self.angle + steps * self.velocity
        position = [self.centre_point[0] + self.radius_x * math.cos(angle),
                    self.centre_point[1] + self.radius_y * math.sin(angle),
                    self.centre_point[2]]
        position = np.array(position)
        d_p = position - self.position

        tangent_vector = [1, -self.radius_x * math.sin(angle) / self.radius_y, 0]
        tangent_vector /= np.linalg.norm(tangent_vector)
        theta = math.atan2(tangent_vector[1], tangent_vector[0])       
        return [d_p, theta- self.theta] 