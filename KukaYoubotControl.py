# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import Robotics as Robotics
import Utils
import pandas as pd

############################################################################################################################################

# Create Robotics Object 

robotics = Robotics.Robotics()

############################################################################################################################################

# Helper Functions

near_Zero     = lambda matrix : np.where(np.abs(matrix) < 1e-06, 0, matrix)
near_Zero_Val = lambda value : 0.0 if np.abs(value) < 1e-06 else value

############################################################################################################################################

# Trajectory Generator Function 

def reference_Trajectory(initial_configuration, final_configuration, time, time_step, trajectory_type, time_scaling):
    """Function to generate reference trajectory or desired trajectory. ===> Type : Free Function.
       Parameters:
       ===========
                  initial_configuration : The Initial SE(3) Configuration; ===> Type : Numpy Matrix; Shape : (4, 4).
                  final_configuration   : The Final SE(3) Configuration;   ===> Type : Numpy Matrix; Shape : (4, 4).
                  time                  : The Total Time of Trajectory in seconds;   ===> Type : Float.
                  time_step             : The Time Step (dt) in seconds;             ===> Type : Float.
                  trajectory_type       : Screw ('S') or Cartesian ('C') Trajectory; ===> Type : String.
                  time_scaling          : Cubic ('C') or Quintic ('Q') time scaling; ===> Type : String.
       Return:
       ===========
                  trajectory : The Desired Trajectory; ===> Type : Numpy Tensor; Shape : (n_points, 4, 4).
    """
    
    ### Check Inputs
    assert initial_configuration.shape == (4, 4),            "Shape of Initial Configuration Matrix should be (4, 4)."
    assert np.isnan(initial_configuration).any() == False,   "There is a NaN value in Initial Configuration Matrix."
    assert final_configuration.shape   == (4, 4),            "Shape of Final Configuration Matrix should be (4, 4)."
    assert np.isnan(final_configuration).any() == False,     "There is a NaN value in Final Configuration Matrix."
    assert trajectory_type == 'S' or trajectory_type == 'C', "Trajectory Type should be 'S' or 'C'."
    assert time_scaling == 'C' or time_scaling == 'Q',       "Time Scaling should be 'C' or 'Q'."
    
    ### Select Trajectory Function : Screw or Cartesian
    if trajectory_type == 'S':
        trajectory_function = robotics.screw_Trajectory

    elif trajectory_type == 'C':
        trajectory_function = robotics.cartesian_Trajectory
        
    ### Calculate Number of Points in the trajectory
    n_points = int((time/time_step))
    
    ### Generate the Trajectory
    trajectory = trajectory_function(initial_configuration, final_configuration, time, n_points, time_scaling)
        
    return trajectory

# Helper Function to duplicate SE3 while grasping

def duplicate_Trajectory(configuration, n_points):
    """Duplicates the Configuration n times. ===> Type : Free Function.
       Parameters:
       ===========
                  configuration : The SE(3) Configuration; ===> Type : Numpy Matrix; Shape : (4, 4).
                  n_points      : The number of points;    ===> Type : Int.
       Return:
       ===========
                  trajectory : The Trajectory during grasping; ===> Type : Numpy Tensor; Shape : (n_points, 4, 4).
    """
    ### Check Inputs
    assert configuration.shape == (4, 4) ,         "Shape of Configuration Matrix should be (4, 4)."
    assert np.isnan(configuration).any() == False, "There is a NaN value in Configuration Matrix."
    
    ### Create trajectory tensor of shape (n_points, 4, 4)
    trajectory = np.empty((n_points, 4, 4)) * np.nan
    
    ### Assign configuration to all the matrices in the trajectory
    trajectory[:, :, :] = configuration
    
    ### Check if there is no nan values and return
    if np.isnan(trajectory).any() == False:
        return trajectory
    else:
        raise Utils.ThereIsNan
        
# Helper Function to combine all trajectories

def combine_Trajectories(list_trajectory, list_gripper_act):
    """Function to combine reference trajectory and gripper actuation. ===> Type : Free Function.
       Parameters:
       ===========
                  list_trajectory  : The list of trajectory; ===> Type : List; Length : m; 
                                     -------- Data : Numpy Matrix; Shape : (n, 4, 4).
                  list_gripper_act : The list of gripper position @ particular trajectory as per trajectories 
                                     in the list_trajectory; ===> Type : List; Length : m; Data : Int.
       Return:
       ===========
                  reference_trajectory : A list of list containing SE(3) Configuration and gripper actuation; ===> Type : List; 
                                         -------- Data : List of Numpy Matrix(Shape : (4,4)) and Gripper Actuation(Int).                              
    """
    ### Create an empty list
    reference_trajectory = []  
    
    ### Loop through the list_trajectory and list_gripper_act
    for trajectory, gripper in zip(list_trajectory, list_gripper_act):
        for SE_3_matrix in trajectory:
            reference_trajectory.append([SE_3_matrix, gripper])

    ### Return
    return reference_trajectory

############################################################################################################################################

# Kuka Manipulator Screw Information

M0E = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])  # M0E Matrix
TB0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]]) # TB0 Matrix
T0B = robotics.inverse_T_Matrix(TB0)

### Screw Axis
B1 = np.array([0, 0, 1, 0, 0.033, 0])
B2 = np.array([0, -1, 0, -0.5076, 0, 0])
B3 = np.array([0, -1, 0, -0.3526, 0, 0])
B4 = np.array([0, -1, 0, -0.2176, 0, 0])
B5 = np.array([0, 0, 1, 0, 0, 0])

### Screw List
screw_list = [B1, B2, B3, B4, B5]

############################################################################################################################################

# Velocity Integrator Function 

def integrator(c_velocity, p_angle, dt):
    """Function which integrates velocity to angle using first-order Euler. ===> Type : Free Function.
       Parameters:
       ===========
                  c_velocity : The current joint and wheel velocity in rad/s; ===> Type : Numpy Matrix; Shape : (n_joints + n_wheel, 1)
                  p_angle    : The previous joint and wheel angles in rad;    ===> Type : Numpy Matrix; Shape : (n_joints + n_wheel, 1)
                  dt         : The Time Step size; ===> Type : Float.
       Return:
       ===========
                  The current joint and wheel angles in rad; ===> Type : Numpy Array; Shape : (n_joints + n_wheel, 1).
    """
    ### Check Inuts
    assert p_angle.shape == c_velocity.shape,   "Shape of Velocity and Angle Matrix is not equal."
    assert np.isnan(c_velocity).any() == False, "There is a NaN value in Velocity Matrix."
    assert np.isnan(p_angle).any() == False,    "There is a NaN value in Angle Matrix."
    
    ### Return
    return p_angle + (c_velocity * dt) # First Order Euler Step

############################################################################################################################################

# Odometry Function

l_youbot = 0.235  # l value of Youbot; 2l = 0.47 m
w_youbot = 0.150  # w value of Youbot; 2w = 0.30 m
r_youbot = 0.0475 # r value of Youbot; r = 0.0475 m
l_w = 1/(l_youbot + w_youbot) # 1 / (l +w)
r_4 = r_youbot * 0.25         # r / 4

def odometry(delta_theta, p_config_q, p_head_phi):
    """Function which calculates chasis configuration using odometry. ===> Type : Free Function.
       Parameters:
       ===========
                  delta_theta : The change in wheel angles;  ===> Type : Numpy Matrix; Shape : (4, 1).
                  p_config_q  : The Previous Configuration;  ===> Type : Numpy Matrix; Shape : (3, 1).
                  p_head_phi  : The Previous Heading in rad; ===< Type : Float.
       Return:
       ===========
                  c_config_q : The Current Configuration; ===> Type : Numpy Matrix; Shape : (3, 1).
                  TBB_       : The SE(3) Congiguration between previous frame and current frame; ===> Type : Numpy Matrix; Shape : (4, 4).
    """
    ### Check Inputs
    assert delta_theta.shape == (4, 1), "Shape of delta_theta is not (4, 1)."
    assert p_config_q.shape == (3, 1),  "Shape of Velocity and Angle Matrix is not (3, 1)."
    
    ### Compute F Vector
    F  = r_4 * np.array([[-l_w, l_w, l_w, -l_w], [1, 1, 1, 1], [-1, 1, -1, 1]]) # Shape : (3, 4)
    
    ### Compute Planar Twist Vb
    Vb =  np.dot(F, delta_theta) # Shape : (3, 1)
    wbz, vbx, vby = Vb.ravel()   # Destructure the Array
    
    ### Create Saptial Twist Vb_6
    Vb_6 = np.array([[0], [0], [wbz], [vbx], [vby], [0]]) # Shape : (6, 1)
    
    ### Calculate TBB_
    Vb6_se3 = robotics.vector_To_SE3(Vb_6.ravel())             # Shape : (4, 4) in se(3)
    TBB_    = robotics.matrix_Exponential_Screw(Vb6_se3, 1.0)  # Shape : (4, 4) in SE(3)
    
    ### Calculate delta_qb
    if near_Zero_Val(wbz) == 0.0:
        delta_qb = np.array([[0], [vbx], [vby]]) # Shape : (3, 1)
    else:
        delta_qb = np.array([[wbz],
                            [(vbx*np.sin(wbz) + vby*(np.cos(wbz)-1))/wbz],
                            [(vby*np.sin(wbz) + vbx*(1-np.cos(wbz)))/wbz]]) # Shape : (3, 1)
    
    ### Calculate delta_q
    R = np.array([[1, 0, 0], 
                  [0, np.cos(p_head_phi), -np.sin(p_head_phi)], 
                  [0, np.sin(p_head_phi), np.cos(p_head_phi)]])  # Shape : (3, 3)
    delta_q = np.dot(R, delta_qb)                                # Shape : (3, 1)
    
    ### Calculate New Configuration
    c_config_q = p_config_q + delta_q  # Shape : (3, 1)
    
    ### Return 
    return c_config_q, TBB_

############################################################################################################################################

# Control Function

Kp, Ki = 50, 0.5
Kp_ = np.eye(6) * Kp
Ki_ = np.eye(6) * Ki

def control(TSE_desired, TSE_desired_next, TSE_actual, dt, X_int_error):
    """Function to obtain End Effector Twist ===> Type : Free Function.
       Parameters:
       ===========
                  TSE_desired : The desired SE(3) configuration; ===> Type : Numpy Matrix; Shape : (4, 4).
                  TSE_actual  : The actual SE(3) configuration;  ===> Type : Numpy Matrix; Shape : (4, 4).
                  TSE_desired_next : The desired SE(3) configuration in next time step; ===> Type : Numpy Matrix; Shape : (4, 4).
                  dt : The time step;                  ===> Type : Float.
                  X_int_error : The Integration Error; ===> Type : Numpy Matrix; Shape : (6, 1).
       Return:
       ===========
                  V : The End-Effector Twist;                            ===> Type : Numpy Matrix; Shape : (6, 1).
                  X_int_error : The Integration Error after 1 time step; ===> Type : Numpy Matrix; Shape : (6, 1).
    """
    ### Check Inputs
    assert TSE_desired.shape == (4, 4),      "Shape of TSE_desiered is not (4, 4)."
    assert TSE_desired_next.shape == (4, 4), "Shape of TSE_desired_next is not (4, 4)."
    assert TSE_actual.shape == (4, 4),       "Shape of TSE_actual is not (4, 4)."
    assert X_int_error.shape == (6, 1),      "Shape of X_int_error is not (6, 1)."
    assert dt != 0.0,                        "Time step cannot be zero."
    
    ### Finding Twist
    ###### 1. Find Xerror
    Xerror = robotics.matrix_Log_6(np.dot(robotics.inverse_T_Matrix(TSE_actual), TSE_desired)) # Shape : (4, 4)
    Xerror = robotics.SE3_To_Vector(Xerror) # se(3) to screw axis; Shape : (6)
    Xerror = np.expand_dims(Xerror, axis=1) # Shape : (6, 1)
    
    ###### 2. Find Twist that takes desired to desired_next
    Vd = (1/dt)*robotics.matrix_Log_6(np.dot(robotics.inverse_T_Matrix(TSE_desired), TSE_desired_next)) # Shape : (4, 4)
    Vd = robotics.SE3_To_Vector(Vd) # se(3) to screw axis; Shape : (6,)
    Vd = np.expand_dims(Vd, axis=1) # Shape : (6, 1)
    
    ###### 3. Find Adjoint of TSE_actual_inv and TSE_desired
    Ad = robotics.adjoint(np.dot(robotics.inverse_T_Matrix(TSE_actual), TSE_desired)) # Shape : (6, 6)
    
    ###### 4. Find the twist
    X_int_error = X_int_error + Xerror*dt                               # Shape : (6, 1)
    V = np.dot(Ad, Vd) + np.dot(Kp_, Xerror) + np.dot(Ki_, X_int_error) # Shape : (6, 1)
    
    return V, X_int_error, Xerror

############################################################################################################################################

# Joint Limits Testing

max_joint_angles = np.array([2.932, 0.191, 1.914, 0.262, 2.890])
min_joint_angles = np.array([-2.932, -1.117, -1.4, -0.999, -2.890])

def test_Joint_Limits(joint_angles):
    """Tests Joint Angle to avoid singularities. ===> Free Function.
       Parameters:
       ===========
                  joint_angles : The joint angles of the manipulator;     ===> Type : List; Length : 5.
       Return:
       ===========
                  l_ind     : The indices of joint angles which violate joint limits;                ===> Type : List.
                  violation : Boolean Value telling if there is any joint which violates the limits; ===> Type : Bool.
    """
    ### Assert
    assert len(joint_angles) == 5, "The length of joint_angles is not 5."
    
    ### Create variables
    l_ind = []          # List to store indices of joint which violates joint limits
    violation = False   # Variable to indicate that there is violation in joint limits
    
    ### Loop through the joint_angles
    for i, angle in enumerate(joint_angles):
        if angle <= min_joint_angles[i] or angle >= max_joint_angles[i]: ## Check Limits
            l_ind.append(i)  # Append to List 
            violation = True # Make violation as true as there is violation in joint limits
    
    ### Return
    return l_ind, violation 

############################################################################################################################################

# Velocity of Youbot 

def velocity_Youbot(V, joint_angles):
    """Obtains the Velocity of the Youbot by calculating the Jacobian. ===> Type : Free Function.
       Parameters:
       ===========
                  V            : The End-Effector Twist;                      ===> Type : Numpy Matrix; Shape : (6, 1).
                  joint_angles : The Joint Guess list (essentially the previous joint values); ===> Type : List; Length : 5.
       Return:
       ===========
                  The joint-wheel velocities of the youbot; ===> Type : Numpy Matrix; Shape : (9, 1).
    """
    ### Check Inputs
    assert V.shape == (6, 1),            "Shape of Twist is not (6, 1)."

    ###### Find Base Jacobian
    ### Compute T0E = T0B . TBS . TSE
    T0E = robotics.forward_Kinematics_Body(screw_list, joint_angles, M0E)
    
    ### Compute F6
    F6 = np.zeros((6, 4)) # Shape : (6, 4)
    F  = r_4 * np.array([[-l_w, l_w, l_w, -l_w], [1, 1, 1, 1], [-1, 1, -1, 1]]) # Shape : (3, 4)
    F6[2:5, :] = F
    
    ### Computer Jacobian of Chassis
    J_base = np.dot(robotics.adjoint(np.dot(robotics.inverse_T_Matrix(T0E), T0B)) , F6) # Shape : (6, 4)
    
    ###### Find Manuplator Jacobian
    ### Check for joint limits violation
    list_indices, violation = test_Joint_Limits(joint_angles)

    ### Find Jacobian using the obtained joint_angles
    J_manipulator = robotics.jacobian_Body(screw_list, joint_angles) # Shape : (6, 5)
    
    ### Change Manipulator Jacobian if there is violation
#     if violation:
#         for i in list_indices:
#             J_manipulator[:, i] = 0.0
            
    ### Overall Jacobian
    J = np.hstack((J_base, J_manipulator)) # Shape : (6, 9)

    ### Return
    return np.dot(np.linalg.pinv(J, rcond=1e-04), V)  # Shape : (6, 1)

############################################################################################################################################

# Plot Function

def plot_Twist_Error(twist_error):
    """Plots the Twist Error; ===> Type : Free Function.
       Parameters:
       ===========
                  twist_error : The Twist Error at each time step of the simulation; ===> Type : Numpy Matrix; Shape : (n_time_steps, 6).
       Return:
       ===========
                  No return value.
    """
    
    # Extract Log Error
    log_wx = twist_error[:,0]
    log_wy = twist_error[:,1]
    log_wz = twist_error[:,2]
    log_vx = twist_error[:,3]
    log_vy = twist_error[:,4]
    log_vz = twist_error[:,5]
    
    # Plot all Log Error
    plt.figure(figsize=(10, 7))
    plt.plot(log_wx, color = 'red', label='WX')
    plt.plot(log_wy, color = 'green', label='WY')
    plt.plot(log_wz, color = 'blue', label='WZ')
    plt.plot(log_vx, color = 'cyan', label='VX')
    plt.plot(log_vy, color = 'black', label='VY')
    plt.plot(log_vz, color = 'orange', label='WX')
    plt.xlabel("Time Stamps", fontsize=14)
    plt.ylabel("Log Error", fontsize=14)
    plt.title("Time Stamp v/s Log Error", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()
    
############################################################################################################################################