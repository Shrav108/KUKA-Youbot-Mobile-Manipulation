import numpy as np

class Robotics:
    """The Robotics Class with concepts of Modern Robotics."""
    
    def __init__(self):
        """Constructor of Robotics class.
           Parameters:
           ===========
                      None.
        ===========================================================
        """
        self.SO3_To_Vector = lambda matrix : np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])
        self.near_Zero     = lambda matrix : np.where(np.abs(matrix) < 1e-06, 0, matrix)
        self.trans_To_RP   = lambda matrix : [matrix[0:-1, 0:-1], matrix[0:-1, -1]]
    
    ### CHAPTER 3 FUNCTIONS
    def vector_To_SO3(self, vector):
        """Converts a Vector to so3 matrix;  ===> Type : Member Function.
           Parameters:
           ===========
                      vector : The vector of rotation;   ===> Type : Numpy Array; Shape: (3,).
           Return:
           ===========
                      The so3 Matrix form of the vector; ===> Type : Numpy Array; Shape: (3, 3).
        =========================================================================================
        """
        return np.array([[    0,      -vector[2], +vector[1]],
                         [+vector[2],     0,      -vector[0]],
                         [-vector[1], +vector[0],      0   ]])
    
    def vector_To_SE3(self, screw_vector):
        """Converts a Screw Axis to se3 matrix;  ===> Type : Member Function.
           Parameters:
           ===========
                      screw_vector : The screw axis of the joint; ===> Type : Numpy Array; Shape : (6,).
           Return:
           ===========
                      vector_se3 : The se3 Form of the Screw Axis; ===> Type : numpy Array; Shape : (4, 4).
        ====================================================================================================
        """
        vector_se3 = np.zeros((4,4))                       # Initialize Vector SE3
        vector_so3 = self.vector_To_SO3(screw_vector[0:3]) # Get the SO3 Form of the omega vector(angular velocity).
        linear_velocity = screw_vector[3:]                 # Get Linear Velocity Values 
        vector_se3[0:-1, 0:-1] = vector_so3                # Assign the SO3 to SE3.
        vector_se3[0:-1, -1]   = linear_velocity           # Assign Linear Velocity to SE3.
        
        # Return
        return vector_se3
    
    def SE3_To_Vector(self, vector_se3):
        """Converts the matrix in se3 to a screw vector; ===> Type : Member Function.
           Parameters:
           ===========
                      vector_se3 : The se3 Form of the Screw Axis; ===> Type : numpy Array; Shape : (4, 4). 
           Return:
           ===========
                      The screw axes vector of the joint; ===> Type : Numpy Array; Shape : (6,).
        ===================================================================================================
        """
        return np.array([vector_se3[2, 1], vector_se3[0, 2], vector_se3[1, 0], 
                         vector_se3[0,-1], vector_se3[1,-1], vector_se3[2,-1]])
    
    def compute_Star(self, omega_so3, linear_velocity, theta):
        """Computes the Star(*) term in Exponential Matrix of screw axis; ===> Type : Member Function.
           Parameters:
           ===========
                      omega_so3       : The so3 matrix of angular velocity; ===> Type : Numpy Array; Shape : (3, 3).
                      linear_velocity : The linear velocity vector;         ===> Type : Numpy Array; Shape : (3, 1).
                      theta           : The Robot-Joint Value;              ===> Type : Float.
           Return:
           ===========
                      star : The Star(*) term; ===> Type : Numpy Array; Shape : (3, 1).
        =============================================================================================================
        """
        star = np.eye(3)*theta + (1-np.cos(theta))*omega_so3 + ((theta - np.sin(theta))*np.dot(omega_so3, omega_so3))
        star = np.dot(star, linear_velocity)
        
        # Return 
        return star
        
    
    def matrix_Exponential(self, omega_so3, theta=None):
        """Computes the Matrix Exponential / Rodrigues Formula;  ===> Type : Member Function.
           Parameters:
           ===========
                      omega_so3 : The so3 form of omega vector; ===> Type : Numpy Array; Shape : (3, 3).
                      theta     : The Robot-Joint value;        ===> Type : Float.
           Return:
           ===========
                      The Matrix Exponential;  ===> Type : Numpy Array; Shape : (3, 3).
        ================================================================================================
        """
        # If theta is not given
        if theta == None:
            omg_vec = self.SO3_To_Vector(omega_so3)
            if np.linalg.norm(omg_vec) == 0.0:
                return np.eye(3)
            else:
                theta = np.linalg.norm(omg_vec)  # Theta
                return self.near_Zero(self.matrix_Exponential(omega_so3/theta, theta))
            
        return np.eye(3) + (np.sin(theta) * omega_so3) + ((1 - np.cos(theta)) * np.dot(omega_so3, omega_so3))
    
    
    def matrix_Exponential_Screw(self, vector_se3, theta=None):
        """Computes the Matrix Exponential of a Screw Axis. ===> Type : Member Function.
           Parameters:
           ===========
                      vector_se3 : The se3 form of Screw Axis; ===> Type : Numpy Array; Shape : (4, 4).
                      theta      : The Robot Joint value;      ===> Type : Float.
           Return:
           ===========
                      matrix_exp_screw : The matrix Exponential of Screw Axis; ===> Type : Numpy Array; Shape : (4, 4).
        ===============================================================================================================
        """
        matrix_exp_screw = np.zeros((4,4)) # Initialize to a Zero matrix.
        matrix_exp_screw[-1, -1] = 1.0     # Change (4,4) element to 1.
        
        
        # If the theta is None i.e. theta is not given used in self.screw_Trajectory
        if theta == None:
            omg_theta = self.SO3_To_Vector(vector_se3[0:-1, 0:-1]) # Extract SO3 to Vector; Shape: (3, )
            
            if np.linalg.norm(omg_theta) == 0.0: # If there is no rotation
                return_vector = np.eye(4)
                return_vector[0:-1, -1] = vector_se3[0:-1, -1]
                return return_vector
            
            else:
                theta = np.linalg.norm(omg_theta)       # Float
                w_mat = vector_se3[0:-1, 0:-1]/theta    # Shape : (3, 3)
                v = vector_se3[0:-1, -1].reshape(3, 1)  # Shape : (3, 1)
                matrix_exp_screw[0:-1, 0:-1] = self.matrix_Exponential(w_mat, theta)          # Shape : (3, 3)
                matrix_exp_screw[0:-1, -1] = self.compute_Star(w_mat, v, theta).ravel()/theta # Shape : (3, )   
                
                # Return
                return self.near_Zero(matrix_exp_screw)
        
        # Extract Vector SO3 and linear velocity
        omega_so3 = vector_se3[0:-1, 0:-1]                     # Extract the SO3 Matrix in SE3
        linear_velocity = vector_se3[0:-1, -1].reshape((3,1))  # Extract linear velocity from SE3
        
        matrix_exp_screw[0:-1, 0:-1] = self.matrix_Exponential(omega_so3=omega_so3, theta=theta)    # Get Matrix Exponential
        matrix_exp_screw[0:-1, -1] = self.compute_Star(omega_so3=omega_so3, linear_velocity=linear_velocity, 
                                                       theta=theta).ravel() # Get Star Value
        
        # Return
        return matrix_exp_screw
    
    def adjoint(self, t_matrix):
        """Computes the Adjoint of a Transformation Matrix;  ===> Type : Member Function.
           Parameters:
           ===========
                      t_matrix : The Homogenous Transformation Matrix; ===> Type : Numpy Array; Shape : (4, 4).
           Return:
           ===========
                      adoint_t : The Adjoint of Transformation Matrix; ===> Type : Numpy Array; Shape : (6, 6).
        =======================================================================================================
        """
        adjoint_t = np.zeros((6, 6)) # Adjoint Matrix
        
        # Extract Rotation Matrix and Position Matrix
        rotation_matrix = t_matrix[0:-1, 0:-1]    # Rotation Matrix 
        position_matrix = t_matrix[0:-1, -1]      # Position Matrix
        
        # Compute dot product of POSITION_MATRIX_IN_SO3 and ROTATION_MATRIX
        p_r = np.dot(self.vector_To_SO3(position_matrix), rotation_matrix)
        
        # Construct Adjoint Matrix
        adjoint_t[0:3, 0:3] = rotation_matrix
        adjoint_t[3:, 0:3]  = p_r
        adjoint_t[3:, 3:]   = rotation_matrix
        
        # Return
        return adjoint_t
    
    def inverse_T_Matrix(self, T_mat):
        """Computes the Inverse of Transformation Matrix; ===> Type : Member Function.
           Parameters:
           ===========
                      T_mat : The Homogenous Transformation Matrix; ===> Type : Numpy Array; Shape : (4, 4).
           Return:
           ===========
                      T_inv = The Inverse of Transformation Matrix; ===> Type : Numpy Aray; Shape : (4, 4).
        ====================================================================================================
        """
        T_inv = np.eye(4) # Shape : (4, 4)
        
        ### Extract Rotation Matrix and Position Matrix
        R = T_mat[0:3, 0:3]              # Shape : (3, 3)
        p = T_mat[0:-1, -1].reshape((3,1)) # Shape : (3, 1)
        
        T_inv[0:-1, 0:-1] = R.T                   # Shape : (3, 3)
        T_inv[0:-1, -1] = -np.dot(R.T, p).ravel() # Shape : (3, )
        
        return T_inv
    
    def matrix_Log_3(self, R):
        """Obtains the Matrix Logaritm of a Rotation Matrix; ===> Type : Member Function.
           Parameters:
           ===========
                      R : The Rotation Matrix SO3; ===> Type : Numpy Array; Shape : (3, 3).
           Return:
           ===========
                      w_theta_mat : The Omega-Theta Matrix in so3; ===> Type : Numpy Array; Shape : (3, 3).
                      theta       : The theta term;                ===> Type : Float.                
        =======================================================================================================
        """
        #### NOTE:  if you get "NAN" ERROR i.e. because of np.arccos(x). It takes value -1 <= x <= 1.
        arc_cos_theta = (np.trace(R) - 1)/2.0
        
        if arc_cos_theta >=1:             # Maximum trace of a Rotation Matrix is 3 i.e. the frame has no rotation
            return np.zeros((3, 3)), 0.0  # Therefore w_mat is ZERO.
        
        elif arc_cos_theta <=-1:     # Minimum Trace of a Rotation Matrix is -3 i.e. the frame is inverted  
            if (np.abs(1 + R[2, 2]) > 1e-06) :
                w_hat = 1/np.sqrt(2*(1 + R[2, 2])) * np.array([R[0,2], R[1,2], 1+R[2,2]])
                
            elif (np.abs(1 + R[1,1])) > 1e-06 :
                w_hat = 1/np.sqrt(2*(1 + R[1, 1])) * np.array([R[0,1], 1+R[1,1], R[2,1]])
            
            else:
                w_hat = 1/np.sqrt(2*(1 + R[0, 0])) * np.array([1+R[0,0], R[1,0], R[2,0]])
            
            return self.vector_To_SO3(np.pi * w_hat) , np.pi
        
        else:
            theta = np.arccos(arc_cos_theta)
            return theta/(2*np.sin(theta)) * (R - R.T), theta
        
    
    def matrix_Log_6(self, T):
        """Computes the Matrix Logaritm of a Transformation matrix; ===> member Function.
           Paramters:
           ==========
                     T : The Transformation Matrix;  ===> Type : Numpy Array; Shape : (4, 4).
           Return:
           ==========
                     mat_log : The Matrix Logarithm; ===> Type : Numpy Array; Shape : (4, 4).
        ========================================================================================
        """
        mat_log = np.zeros((4,4))
        G_inv   = lambda w, theta : np.eye(3)/theta - w/2.0 + (1/theta - 0.5*(1/np.tan(0.5*theta)))*np.dot(w, w)
        
        ### Extract Rotation and Position from T
        R, p = T[0:-1, 0:-1], T[0:-1, -1]          # Shape : (3, 3) and (3,)
        
        ### Get Matrix Log 3 of Rotation Matrix
        w_theta_mat, theta = self.matrix_Log_3(R)  # Get [W].theta matrix
       
        if np.array_equal(w_theta_mat, np.zeros((3,3))): # If there is no Rotation
            mat_log[0:-1, -1] = p    
            return mat_log
        
        else:
            mat_log[0:-1, 0:-1] = w_theta_mat                     # [w] * theta
            v =  np.dot(G_inv(w_theta_mat/theta, theta), p.reshape((3, 1))) # The v term LOG 6
            mat_log[0:-1, -1] = v[0:, -1]*theta                   # v * theta
            return mat_log
        
        
    ### CHAPTER 4 FUNCTIONS
    def forward_Kinematics_Space(self, screw_list, theta_list, m_matrix):
        """Computes the Forward Kinematics using POE in space frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Screw Axis list of numpy array;         ===> Type : List; Lenght : n_joints; Shape : (6,).
                      theta_list : The Joint Values list;                      ===> Type : List; Length : n_joints; Value Type : Float.
                      m_matrix   : The Transformation Matrix at Zero Position; ===> Type : Numpy Array; Shape : (4, 4).
           Return:
           ===========
                      t_matrix : The final transformation matrix;  ===> Type : Numpy Array; Shape : (4, 4).             
        ===============================================================================================================================
        """
        ### Initialize Transformation matrix as M matrix at Zero Position
        t_matrix = m_matrix
        screw_list_ = screw_list[::-1] # Reverse the list
        theta_list_ = theta_list[::-1] # Reverse the list
        
        for screw_axis, theta in zip(screw_list_, theta_list_):
            ### Get Matrix Exponential of the screw
            vector_se3 = self.vector_To_SE3(screw_axis)                         # Get Screw Axis in SE3 form
            matrix_exp_screw = self.matrix_Exponential_Screw(vector_se3, theta) # Get Matrix Exponential of Screw
            
            ### Compute T(theta)
            t_matrix = np.dot(matrix_exp_screw, t_matrix)                       # Pre multiply the Matrix Exponential of a screw
            
        # Return
        return t_matrix
    
    def forward_Kinematics_Body(self, screw_list, theta_list, m_matrix):
        """Computes the Forward Kinematics using POE in body frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Screw Axis list of numpy array;         ===> Type : List; Lenght : n_joints; Shape : (6,).
                      theta_list : The Joint Values list;                      ===> Type : List; Length : n_joints; Value Type : Float.
                      m_matrix   : The Transformation Matrix at Zero Position; ===> Type : Numpy Array; Shape : (4, 4).
           Return:
           ===========
                      t_matrix : The final transformation matrix;  ===> Type : Numpy Array; Shape : (4, 4).
        ===============================================================================================================================
        """
        ### Initialize Transformation matrix as M matrix at Zero Position
        t_matrix = m_matrix
        
        for screw_axis, theta in zip(screw_list, theta_list):
            ### Get Matrix Exponential of the screw
            vector_se3 = self.vector_To_SE3(screw_axis)                         # Get Screw Axis in SE3 form
            matrix_exp_screw = self.matrix_Exponential_Screw(vector_se3, theta) # Get Matrix Exponential of Screw
            
            ### Compute T(theta)
            t_matrix = np.dot(t_matrix, matrix_exp_screw)                       # Pre multiply the Matrix Exponential of a screw
            
        # Return
        return t_matrix
    
    ### CHAPTER 5 FUNCTIONS
    def jacobian_Space(self, screw_list, theta_list):
        """Computes the Jacobian in Space Frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Screw Axes list of numpy arrays; ===> Type : List; Length : n_joints; Shape : (6,).
                      theta_list : The joint values;                    ===> Type : List; Length : n_joints.
           Return:
           ===========
                      jacobian_space : The Jacobian Matrix in Space Frame; ===> Type : Numpy Array; Shape : (6, n_joints).
        ==================================================================================================================
        """
        jacobian_space = np.zeros((6, len(theta_list))) # Construct an initial Jacobian Matrix
        
        for i in range(len(theta_list)):
            screw_axes = screw_list[i].reshape((6, 1)) # Extract Screw Axes as (6,1) Matrix
            j = 0                                      # Iterator To compute Product of Exponentials ## Method 2 : j = i - 1
            poe_matrix_adj = np.identity(4)            # Matrix to store POE for adjoint computation   
            
            while(j != i):  # Method 2 : j>=0                           
                vector_se3 = self.vector_To_SE3(screw_list[j])                               # Convert Screw Axes to SE3
                matrix_exp_screw = self.matrix_Exponential_Screw(vector_se3, theta_list[j])  # Get POE of Screw Axes in SE3 Form
                poe_matrix_adj = np.dot(poe_matrix_adj, matrix_exp_screw)                    # Compute Dot Product ## Method 2 : np.dot(matrix_exp_screw, poe_matrix_adj)
                j = j + 1  # Decrease the iterator by 1    ## Method 2 : j = j - 1
                
            # Compute Adjoint and Multiply with screw axes
            adjoint_poe_i = self.adjoint(poe_matrix_adj)          # Compute Adjont
            jacobian_joint_i = np.dot(adjoint_poe_i, screw_axes)  # Compute Jacobian of the joint
            jacobian_space[:, i] = jacobian_joint_i[:, 0]         # Store Jacobian of the joint in main jacobian matrix
        
        # Return
        return jacobian_space
    
    def jacobian_Body(self, screw_list, theta_list):
        """Computes the Jacobian in Body Frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Screw Axes list of numpy arrays; ===> Type : List; Length : n_joints; Shape : (6,).
                      theta_list : The joint values;                    ===> Type : List; Length : n_joints.
           Return:
           ===========
                      jacobian_space : The Jacobian Matrix in Body Frame; ===> Type : Numpy Array; Shape : (6, n_joints).
        ==================================================================================================================
        """
        jacobian_body = np.zeros((6, len(theta_list))) # Construct an Initial Jacobian matrix
        
        for i in range(len(theta_list)):
            screw_axes = screw_list[i].reshape((6,1)) # Extract Screw Axes as (6,1) Matrix
            j = len(theta_list) - 1                   # Iterator To compute Product of Exponentials ## Method 2 : j = i+1
            poe_matrix_adj = np.identity(4)           # Matrix to store POE for adjoint computation
            
            while(j != i): ## Method 2: j != len(theta_list)
                vector_se3 = -self.vector_To_SE3(screw_list[j])                             # Convert Screw Axes to SE3
                matrix_exp_screw = self.matrix_Exponential_Screw(vector_se3, theta_list[j]) # Get POE of Screw Axes in SE3 Form
                poe_matrix_adj = np.dot(poe_matrix_adj, matrix_exp_screw)                   # Compute Dot Product ## Method 2: np.dot(matrix_exp_screw, poe_matrix_adj) 
                j = j - 1  # Decrease the iterator by 1 ## Method 2 : j = j + 1
                
            # Compute Adjoint and Multiply with screw axes 
            adjoint_joint_i = self.adjoint(poe_matrix_adj)          # Compute Adjont
            jacobian_joint_i = np.dot(adjoint_joint_i, screw_axes)  # Compute Jacobian of the joint
            jacobian_body[:, i] = jacobian_joint_i[:, 0]            # Store Jacobian of the joint in main jacobian matrix
            
        # Return
        return jacobian_body
    
    def jacobian_Body_Given_Space(self, t_matrix_body_space, jacobian_space):
        """Computes the Body Jacobian given Space Jacobian; ===> Type : Member Function.
           Parameters:
           ===========
                      t_matrix_body_space : The Transformation Matrix of Space w.r.t Body; ===> Type : Numpy Array; Shape : (4, 4).
                      jacobian_space      : The Space Jacobian;  ===> Type : Numpy Array; Shape : (6, n_joints).
           Return:
           ===========
                      The Body Jacobian; ===> Type : Numpy Array; Shape : (6, n_joints).
        ============================================================================================================================
        """
        adjoint_t_bs = self.adjoint(t_matrix_body_space) # Get Adjoint
        
        # Return
        return np.dot(adjoint_t_bs, jacobian_space)
    
    
    def jacobian_Space_Given_Body(self, t_matrix_space_body, jacobian_body):
        """Computes the Space Jacobian given Body Jacobian; ===> Type : Member Function.
           Parameters:
           ===========
                      t_matrix_space_body : The Transformation Matrix of Body w.r.t Space; ===> Type : Numpy Array; Shape : (4, 4).
                      jacobian_body       : The Body Jacobian;  ===> Type : Numpy Array; Shape : (6, n_joints).
           Return:
           ===========
                      The Space Jacobian; ===> Type : Numpy Array; Shape : (6, n_joints).
        ============================================================================================================================
        """
        adjoint_t_sb = self.adjoint(t_matrix_space_body) # Get Adjoint
        
        # Return
        return np.dot(adjoint_t_sb, jacobian_body)
    
    def torque_Given_Wrench(self, jacobian, wrench):
        """Computes Joint Torques given Jacobian and Wrench; ===> Type : Member Function.
           Parameters:
           ===========
                      jacobian : The Body Jacobian or Space Jacobian;     ===> Type : Numpy Array; Shape : (6, n_joints).
                      wrench   : The Wrench Matrix of Moments and Forces; ===> Type : Numpy Array; Shape : (6, 1).
           Return:
           ===========
                      Returns the Joint Torques in Body or Space Frame; ===> Type : Numpy Array; Shape : (n_joints, 1).
        ==================================================================================================================
        """
        # Return
        return np.dot(jacobian.T, wrench)
    
    def manupability(self, jacobian):
        """Computes the Manupability Measures of Angular Velocity, Linear Velocity, Moment and Force Elipsoids; ===> Type : Member Function.
           Parameters:
           ===========
                      jacobian : The Body or Space Jacobian; ===> Type : Numpy Array; Shape : (6, n_joints).
           Return:
           ===========
                      A list of list of Isotropic Number, Condition Number and Volume; ===> Type : List; Length : 4.
        ====================================================================================================================================
        """
        # Some Lambda Functions
        get_eigen_vals = lambda matrix : np.linalg.eigvals(matrix)      # To Get Eigen Values
        get_A_matrix   = lambda jacobian : np.dot(jacobian, jacobian.T) # To get A Matrix
        
        get_isotropic_number = lambda vector : np.sqrt(vector.max())/np.sqrt(vector.min())
        get_condition_number = lambda vector : vector.max()/vector.min()
        get_volume = lambda vector : np.sqrt(np.prod(vector))
        
        # Extract Jw and Jv Matrix
        jacobian_w = jacobian[0:3, :] # Jacobian Angular Velocity
        jacobian_v = jacobian[3:, :]  # Jacobian Linear Velocity
        
        # Get A Matrices
        A_w = get_A_matrix(jacobian_w)
        A_v = get_A_matrix(jacobian_v)
        
        # Get Eigen Values
        lambda_ang_vel = get_eigen_vals(A_w)
        lambda_lin_vel = get_eigen_vals(A_v)
        lambda_moment  = 1/lambda_ang_vel
        lambda_force   = 1/lambda_lin_vel
        
        # Get Manupability Measures
        manu_ang = [get_isotorpic_number(lambda_ang_vel), get_condition_number(lambda_ang_vel), get_volume(lambda_ang_vel)]
        manu_lin = [get_isotorpic_number(lambda_lin_vel), get_condition_number(lambda_lin_vel), get_volume(lambda_lin_vel)]
        manu_moment = [get_isotorpic_number(lambda_moment), get_condition_number(lambda_moment), get_volume(lambda_moment)]
        manu_force  = [get_isotorpic_number(lambda_force), get_condition_number(lambda_force), get_volume(lambda_force)]
        
        # Return
        return [manu_ang, manu_lin, manu_moment, manu_force]
    
    ### Chapter 6 : Inverse Kinematics
    def inverse_Kinematics_Body(self, screw_list, m_matrix, t_desired, theta_guess_list):
        """Calculates the Joint Angles given Desired Configuration in Body Frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Body Screw Axes in Zero Position;       ===> Type : List of Numpy Arrays; Length : n_joints; Shape : (6,).
                      m_matrix   : The Transformation Matrix at Zero Position; ===> Type : Numpy Array; Shape : (4, 4).
                      t_desired  : The Desired Configuration matrix in SE3;    ===> Type : Numpy Array; Shape : (4, 4).
                      theta_guess_list : The initial guess list;               ===> Type : List; Length : n_joints.
           Return:
           ===========
                      theta_list : The list of joint values;                ===> Type : List; Length : n_joints.
                      success    : If inverse kinematics successful or not; ===> Type : Bool.
        =========================================================================================================================================
        """
        epsilon_ang = 0.0001 # Threshold
        epsilon_vel = 0.0001 # Threshold
        
        # Anonymous Function for obtaining Twist
        # Operation Sequence ===> FK_IN_BODY ===> INVERSE_T_MATRIX ===> DOT_TSBinv_TSD ===> MATRIX_LOG_6 ===> SE3_TO_VECTOR
        # x = screw_list , y = theta_guess_list, z = m_matrix, T = t_desired
        # Output Shape : (6, )
        get_twist = lambda x, y, z, T: self.SE3_To_Vector(self.matrix_Log_6(np.dot(self.inverse_T_Matrix(self.forward_Kinematics_Body(x, y, z)), T))) # Anonymous Function
    
        # Anonymous Norm Function
        get_norm  = lambda vector : np.linalg.norm(vector)
        
        # Anonymous Psuedo Inverse of jacobian ===> x = screw_list , y = theta_guess_list, z = V_b
        get_psuedo_J_twist = lambda x, y, z  : list(np.dot(np.linalg.pinv(self.jacobian_Body(x, y)), z.reshape((6, 1))).ravel())
        
        # Get Initial Body Twist 
        V_b = get_twist(screw_list, theta_guess_list, m_matrix, t_desired) # Shape : (6,)
        count = 0
        
        # IK Algorithm in Body Frame
        while (get_norm(V_b[0:3]) >= epsilon_ang) or ((get_norm(V_b[3:])) >= epsilon_vel) :
            
            # Update Theta Guess list
            theta_guess_list = list(np.add(theta_guess_list, get_psuedo_J_twist(screw_list, theta_guess_list, V_b)))
            
            # Find New Body Twist
            V_b = get_twist(screw_list, theta_guess_list, m_matrix, t_desired)
            
            # Increase Count
            count+=1
            if count == 200:
                break
            
        if (get_norm(V_b[0:3]) <= epsilon_ang) and ((get_norm(V_b[3:])) <= epsilon_vel) :
            success = True
        else:
            success = False
        
        
        return theta_guess_list, success
    
    def inverse_Kinematics_Space(self, screw_list, m_matrix, t_desired, theta_guess_list):
        """Calculates the Joint Angles given Desired Configuration in Space Frame; ===> Type : Member Function.
           Parameters:
           ===========
                      screw_list : The Space Screw Axes in Zero Position;      ===> Type : List of Numpy Arrays; Length : n_joints; Shape : (6,).
                      m_matrix   : The Transformation Matrix at Zero Position; ===> Type : Numpy Array; Shape : (4, 4).
                      t_desired  : The Desired Configuration matrix in SE3;    ===> Type : Numpy Array; Shape : (4, 4).
                      theta_guess_list : The initial guess list;               ===> Type : List; Length : n_joints.
           Return:
           ===========
                      theta_list : The list of joint values;                ===> Type : List; Length : n_joints.
                      success    : If inverse kinematics successful or not; ===> Type : Bool.
        =========================================================================================================================================
        """
        epsilon_ang = 0.001 # Threshold
        epsilon_vel = 0.001 # Threshold
        
        # Anonymous Function for obtaining twist
        # x = screw_list , y = theta_guess_list, z = m_matrix, T = t_desired
        # get_twist ===> output shape : (6,)
        get_twist = lambda x, y, z, T : np.dot(self.adjoint(self.forward_Kinematics_Space(x, y, z)[0]), self.SE3_To_Vector(self.matrix_Log_6(
                                               np.dot(self.inverse_T_Matrix(self.forward_Kinematics_Space(x, y, z)[0]), T))).reshape((6,1))).ravel()
          
        # Anonymous Function for multiplication if Psuedo Inverse of  J-space and Space twist
        # Parameters ===> x = screw_list , y = theta_guess_list, z = V_s (Twist Spatial)
        # Output Shape : list of (n_joints, )
        get_psuedo_J_twist = lambda x, y, z : list(np.dot(np.linalg.pinv(self.jacobian_Space(x, y)), z.reshape((6,1))).ravel())
        
        # Anonymous function for L2 Norm
        get_norm = lambda vector : np.linalg.norm(vector)
        
        
        ## Get Initial twist of Guess Vector
        V_s = get_twist(screw_list, theta_guess_list, m_matrix, t_desired) # Shape (6,)
        count = 0
        
        # IK Algorithm in Space Frame
        while (get_norm(V_s[0:3]) >= epsilon_ang) or (get_norm(V_s[3:]) >= epsilon_vel):
            
            # Update theta guess list
            theta_guess_list = list(np.add(theta_guess_list, get_psuedo_J_twist(screw_list, theta_guess_list, V_s)))
            
            # Find new Spatial twist
            V_s = get_twist(screw_list, theta_guess_list, m_matrix, t_desired) # Shape (6,)
            
            # Update Count
            count += 1
            if count == 100:
                break
                
        # Check if IK Successful or not       
        if (get_norm(V_s[0:3]) <= epsilon_ang) and ((get_norm(V_s[3:])) <= epsilon_vel) :
            success = True
        else:
            success = False
         
        # Return
        return theta_guess_list, success
    
    ## Chapter 8: Dynamics of Robots
    def ad(self, twist):
        """Calculates the 6x6 [adv] matrix of the given Twist; ===> Type : Member Function.
           Parameters:
           ===========
                      twist : The Twist Vector; ===> Type : Numpy Array; Shape : (6, ).
           Return:
           ===========
                      adv : The [adv] matrix;   ===> Type : Numpy Array; Shape ; (6, 6).
        ===================================================================================
        """
        adv = np.zeros((6,6)) # Main vector
        adv[0:3, 0:3] = self.vector_To_SO3(twist[0:3]) # [w]
        adv[3:, 3:]   = self.vector_To_SO3(twist[0:3]) # [w]
        adv[3:, 0:3]  = self.vector_To_SO3(twist[3:])  # [v]
        
        # Return
        return adv
    
    def get_Inertia_Matrix(self, inertia_list):
        """Returns the Spatial Inertial Matrix for inverse Dynamics; ===> Type : Member Function.
           Parameters:
           ===========
                      inertia_list : The array of Diagonal Elements of Ib(Principal Axes Aligned with {i}) 
                                     and Mass of Link; ===> Type : Numpy Array; Shape : (n_links, 4).
           Return:
           ===========
                      inertia_array : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
        =================================================================================================================            
        """
        inertia_array = []
        for i in range(inertia_list.shape[0]):
            Gb = np.eye(6)
            # Construct Ib part of Gb
            Gb[0,0] = inertia_list[i, 0]  
            Gb[1,1] = inertia_list[i, 1]
            Gb[2,2] = inertia_list[i, 2]

            # Construct mI part of Gb
            Gb[3:, 3:] = np.eye(3)*inertia_list[i, 3]
            
            # Append
            inertia_array.append(Gb) # Append to list
        
        inertia_array = np.array(inertia_array) # Convert List To Numpy Array
        
        # Return
        return inertia_array

    def inverse_Dynamics(self, theta_array, d_theta_array, d2_theta_array, gravity_array, wrench_vector, 
                         M_array, inertia_array, screw_array):
        """Computes the Inverse Dynamics in the Space Frame for Open Chains; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      d_theta_array  : The joint rates i.e. velocity;  ===> Type : Numpy Array; Shape : (n_joints,).
                      d2_theta_array : The joints acceleration;        ===> Type : Numpy Array; Shape : (n_joints,).
                      gravity_array  : The gravity vector @ {0};       ===> Type : Numpy Array; Shape : (3,).
                      wrench_vector  : The Wrench at the end-effector; ===> Type : Numpy Array; Shape : (6, ).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      torque_array   : The Torques at the joint; ===> Type : Numpy Array; Shape : (n_joints, ).
        =====================================================================================================================
        """
        n_joints = theta_array.shape[0]             # Extract Number of joints
        M_00     = np.eye(4).reshape((1,4,4))       # {0} frame congiguration w.r.t {0}; Shape : (1,4,4)
        M_array  = np.append(M_00, M_array, axis=0) # Append M00 to M_array
        
        twist_V   = np.zeros((6, n_joints+1)) # Twist V;
        twist_V_d = np.zeros((6, n_joints+1)) # Twist Derrivative V_dot
        twist_V_d[:, 0] = np.array([0, 0, 0, -gravity_array[0], -gravity_array[1], -gravity_array[2]])
        
        F_matrix = np.zeros((6, n_joints+1))
        F_matrix[:, -1] = wrench_vector
        torque_array = np.zeros((n_joints))
        
        # Forward iteration from Link 1 to n
        for i in range(n_joints):
            
            # Step 1: Find transformation matrix ==>Ti,i-1
            Mi = np.dot(self.inverse_T_Matrix(M_array[i+1]), M_array[i])                                     # Cross Product: Mi_inv x Mi-1; Shape : (4,4)
            Ai = np.dot(self.adjoint(self.inverse_T_Matrix(M_array[i+1])), screw_array[i, :].reshape((6,1))) # Adj(M0i)_inv x Si; Shape : (6, 1)
            Ti = np.dot(self.matrix_Exponential_Screw(-self.vector_To_SE3(Ai.ravel()), theta_array[i]), Mi)  # Transformation Matrix of Link i; Shape : (4,4)

            # Step 2: Find Twist Vi
            Vi = np.dot(self.adjoint(Ti), twist_V[:, i].reshape((6,1))) + (Ai * d_theta_array[i]) # Shape : (6,1)
            twist_V[:, i+1] = Vi.ravel() # Shape : (6,)
            
            # Step 3: Find Derrivative of Twist Vi
            Vi_d = np.dot(self.adjoint(Ti), twist_V_d[:, i].reshape((6,1))) + (np.dot(self.ad(Vi.ravel()), Ai) * d_theta_array[i]) + (Ai*d2_theta_array[i])  # Shape : (6, 1)
            twist_V_d[:, i+1] = Vi_d.ravel() # Shape : (6,)
            
            
        # Backward iteration from Link n to 1:
        for i in range(n_joints-1, -1, -1): # reverse loop; if n_joints = 3 then i = 2, 1, 0 
            
            # Step 1: Find transformation matrix ==>Ti+1,i
            Mi = np.dot(self.inverse_T_Matrix(M_array[i+2]), M_array[i+1]) #  Shape : (4,4)
            Ai = np.dot(self.adjoint(self.inverse_T_Matrix(M_array[i+1])), screw_array[i, :].reshape((6,1))) #  Shape : (6, 1)        
            Ti = np.dot(self.matrix_Exponential_Screw(-self.vector_To_SE3(Ai.ravel()), theta_array[i]), Mi) # Shape : (4,4)

            # Step 2: Find Wrench Fi
            Fi = np.dot(self.adjoint(Ti).T, F_matrix[:, i+1].reshape((6,1))) + \
                 np.dot(inertia_array[i], twist_V_d[:, i+1].reshape((6,1))) - \
                 np.dot(self.ad(twist_V[:, i+1]).T, np.dot(inertia_array[i], twist_V[:, i+1].reshape((6,1)))) # Shape : (6, 1)
            F_matrix[:, i] = Fi.ravel()   # Shape : (6,)
            torque_array[i] = np.dot(Fi.T, Ai) 
            
        return torque_array
    
    def mass_Matrix(self,theta_array, M_array, inertia_array, screw_array):
        """Computes the Mass Matrix; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      mass_matrix  : The Mass Matrix; ===> Type : Numpy Array; Shape : (n_joints, n_joints).
        =====================================================================================================================
        """
        n_joints = theta_array.shape[0]      # Extract number of Joints
        g_array  = np.zeros((3))             # Gravity vector is zero
        wrench_vector = np.zeros((6))        # Wrench is set to zero
        d_theta_array = np.zeros((n_joints)) # Velocity is set to 0
        mass_matrix = np.zeros((n_joints, n_joints)) # Mass Matrix
        
        for i in range(n_joints):
            d2_theta_array = np.zeros((n_joints)) # Construct a Zero Vector
            d2_theta_array[i] = 1                 # The ith element is set to 1
            mass_matrix[:, i] = self.inverse_Dynamics(theta_array, d_theta_array, d2_theta_array, g_array, 
                                                      wrench_vector, M_array, inertia_array, screw_array)
            
        # Return
        return mass_matrix
    
    def quad_Velocity_Forces(self, theta_array, d_theta_array, M_array, inertia_array, screw_array):
        """Computes the Quadratic Velocity Forces; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      d_theta_array  : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      The Quadratic Velocity Forces; ===> Type : Numpy Array; Shape : (n_joints,).
        =====================================================================================================================
        """
        g_array  = np.zeros((3))              # Gravity vector is zero
        wrench_vector  = np.zeros((6))        # Wrench is set to zero
        d2_theta_array = np.zeros((theta_array.shape[0])) # Construct a Zero Vector for acceleration
        
        return self.inverse_Dynamics(theta_array, d_theta_array, d2_theta_array, g_array, 
                                     wrench_vector, M_array, inertia_array, screw_array)
    
    def gravity_Forces(self, theta_array, gravity_array, M_array, inertia_array, screw_array):
        """Computes the Gravity Forces; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      gravity_array  : The joint values;               ===> Type : Numpy Array; Shape : (3,).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      The Gravity Forces; ===> Type : Numpy Array; Shape : (n_joints,).
        =====================================================================================================================
        """
        wrench_vector = np.zeros((6))                     # Wrench is set to zero
        d_theta_array = np.zeros((theta_array.shape[0]))  # Velocity is set to 0
        d2_theta_array = np.zeros((theta_array.shape[0])) # Construct a Zero Vector for acceleration
        
        return self.inverse_Dynamics(theta_array, d_theta_array, d2_theta_array, gravity_array, 
                                     wrench_vector, M_array, inertia_array, screw_array)
    
    def end_Effector_Forces(self, theta_array, wrench_vector, M_array, inertia_array, screw_array):
        """Computes the End Effector Forces; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      wrench_vector  : The Wrench at the end-effector; ===> Type : Numpy Array; Shape : (6, ).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      The End Effector Forces; ===> Type : Numpy Array; Shape : (n_joints,).
        =====================================================================================================================
        """
        d_theta_array = np.zeros((theta_array.shape[0]))  # Velocity is set to 0
        d2_theta_array = np.zeros((theta_array.shape[0])) # Construct a Zero Vector for acceleration
        g_array = np.zeros((3)) # The Gravity vector is 0
        
        return self.inverse_Dynamics(theta_array, d_theta_array, d2_theta_array, g_array, 
                                     wrench_vector, M_array, inertia_array, screw_array)
    
    def forward_Dynamics(self, theta_array, d_theta_array, torque_array, gravity_array, wrench_vector, 
                         M_array, inertia_array, screw_array):
        """Computes the Forward Dynamics in the Space Frame for Open Chains; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The joint values;               ===> Type : Numpy Array; Shape : (n_joints,).
                      d_theta_array  : The joint rates i.e. velocity;  ===> Type : Numpy Array; Shape : (n_joints,).
                      torque_array   : The Torques at the joint;       ===> Type : Numpy Array; Shape : (n_joints,).
                      gravity_array  : The gravity vector @ {0};       ===> Type : Numpy Array; Shape : (3,).
                      wrench_vector  : The Wrench at the end-effector; ===> Type : Numpy Array; Shape : (6, ).
                      M_array        : The Transformation Matrix at Zero Position for each Link including 
                                       end-effector w.r.t base {0};    ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array  : The Spatial Intertial Matrix;   ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array    : The Screw Axes Array of joints; ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      d2_theta_array : The Acceleration at the joints; ===> Type : Numpy Array; Shape : (n_joints, ).
        =====================================================================================================================
        """
        n_joints = theta_array.shape[0]
        torque = torque_array.reshape((n_joints, 1)) # Shape : (n_joints, 1)
        mass_matrix = self.mass_Matrix(theta_array, M_array, inertia_array, screw_list) # Shape : (n_joints, n_joints)
        quad_vel_forces = self.quad_Velocity_Forces(theta_array, d_theta_array, M_array, inertia_array, screw_list).reshape((n_joints, 1)) # Shape : (n_joints, 1)
        gravity_forces  = self.gravity_Forces(theta_array, gravity_array, M_array, inertia_array, screw_array).reshape((n_joints, 1))      # Shape : (n_joints, 1)
        end_eff_forces  = self.end_Effector_Forces(theta_array, wrench_vector, M_array, inertia_array, screw_array).reshape((n_joints, 1)) # Shape : (n_joints, 1)
        
        d2_theta_array = np.dot(np.linalg.inv(mass_matrix), torque-quad_vel_forces-gravity_forces-end_eff_forces).ravel() # Shape : (n_joints,)
        
        return d2_theta_array
    
    ### Chapter 9 : Trajectory
    def cubic_Time_Scaling(self, total_time, current_time):
        """Computes the cubic polynomial s(t); ===> Type : Member Function.
           Parameters:
           ===========
                      total_time   : The Total Time of the Trajectory;   ===> Type : Float.
                      current_time : The current time in the Trajectory; ===> Type : Float.
           Return:
           ===========
                      The Cubic Polynomial s(t); ===> Type : Float.
        """
        return (3*(current_time/total_time)**2) - (2*(current_time/total_time)**3)
    
    def quintic_Time_Scaling(self, total_time, current_time):
        """Computes the quintic polynomial s(t); ===> Type : Member Function.
           Parameters:
           ===========
                      total_time   : The Total Time of the Trajectory;   ===> Type : Float.
                      current_time : The current time in the Trajectory; ===> Type : Float.
           Return:
           ===========
                      The Quintic Polynomial s(t); ===> Type : Float.
        ======================================================================================
        """
        return 10*(current_time/total_time)**3 - (15*(current_time/total_time)**4) + (6*(current_time/total_time)**5)
    
    def joint_Trajectory(self, theta_start, theta_end, total_time, n_points, method):
        """Computes the Straight Line Trajectory in Joint Space; ===> Member Function.
           Parameters:
           ===========
                      theta_start : The starting Joint Variables; ===> Type : Numpy Array; Shape : (n_joints, ).
                      theta_end   : The ending Joint Variables;   ===> Type : Numpy Array; Shape : (n_joints, ).
                      total_time  : The Total Time of Trajectory; ===> Type : Float.
                      n_points    : The number of points in the line; ===> Type : Int.
                      method      : Type of Time Scaling 'C' or 'Q';  ===> Type : String.
           Return:
           ===========
                      trajectory : The Joint Trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints).
        ===========================================================================================================
        """
        # Assign Time Scaling
        assert method == 'C' or method == 'Q', "Time Scaling should be C or Q!"
        if method == 'C':
            time_scaling = self.cubic_Time_Scaling
        else:
            time_scaling = self.quintic_Time_Scaling
            
        # Define Variables
        n_points = int(n_points)       # Convert to int
        dt = total_time/(n_points - 1) # The Time Division
        trajectory = np.zeros((n_points, theta_start.shape[0])) # Trajectory; Shape : (n_points, n_joints+1)
        
        for i in range(n_points):
            s = time_scaling(total_time, dt*i)                           # Get s(t)
            trajectory[i, :] = theta_start + s*(theta_end - theta_start) # Compute trajectory
            
        return trajectory
    
    def screw_Trajectory(self, X_start, X_end, total_time, n_points, method):
        """Computes the Trajectory as a screw motion about a space screw axis; ===> Type : Member Function.
           Parameters:
           ===========
                      X_start    : The starting point in SE3;        ===> Type : Numpy Array; Shape : (4, 4).
                      X_end      : The ending point in SE3;          ===> Type : Numpy Array; Shape : (4, 4).
                      total_time : The Total Time of Trajectory;     ===> Type : Float.
                      n_points   : The number of points in the line; ===> Type : Int.
                      method     : Type of Time Scaling 'C' or 'Q';  ===> Type : String.
           Return:
           ===========
                      trajectory : The Screw Trajectory; ===> Type : Numpy Array; Shape : (n_points, 4, 4).
        ======================================================================================================
        """
        # Assign Time Scaling
        assert method == 'C' or method == 'Q', "Time Scaling should be C or Q!"
        if method == 'C':
            time_scaling = self.cubic_Time_Scaling
        else:
            time_scaling = self.quintic_Time_Scaling
        
        # Define Variables
        n_points = int(n_points)       # Convert to int
        dt = total_time/(n_points - 1) # The Time Division
        trajectory = []                # Trajectory; Shape : (n_points, n_joints+1)
        
        for i in range(n_points):
            traj = np.dot(X_start, self.matrix_Exponential_Screw(self.matrix_Log_6(np.dot \
                            (self.inverse_T_Matrix(X_start), X_end)) * time_scaling(total_time, dt*i)))  # Shape : (4, 4)
            trajectory.append(traj) # Append 
            
        # Return
        return np.array(trajectory)
    
    def cartesian_Trajectory(self, X_start, X_end, total_time, n_points, method):
        """Computes the Trajectory as a straight line; ===> Type : Member Function.
           Parameters:
           ===========
                      X_start    : The starting point in SE3;        ===> Type : Numpy Array; Shape : (4, 4).
                      X_end      : The ending point in SE3;          ===> Type : Numpy Array; Shape : (4, 4).
                      total_time : The Total Time of Trajectory;     ===> Type : Float.
                      n_points   : The number of points in the line; ===> Type : Int.
                      method     : Type of Time Scaling 'C' or 'Q';  ===> Type : String.
           Return:
           ===========
                      trajectory : The Cartesian Trajectory; ===> Type : Numpy Array; Shape : (n_points, 4, 4).
        ========================================================================================================
        """
        assert method == 'C' or method == 'Q', "Time Scaling should be C or Q!"
        if method == 'C':
            time_scaling = self.cubic_Time_Scaling
        else:
            time_scaling = self.quintic_Time_Scaling
            
        # Define Variables
        n_points = int(n_points)                     # Convert to int
        dt = total_time/(n_points - 1)               # The Time Division
        R_start, p_start = self.trans_To_RP(X_start) # Extract R and p
        R_end, p_end = self.trans_To_RP(X_end)       # Extract R and p
        trajectory = []                              # Trajectory
        
        for i in range(n_points):
            traj = np.eye(4) 
            traj[0:-1, 0:-1] = np.dot(R_start, self.matrix_Exponential(self.matrix_Log_3(np.dot(R_start.T, R_end))[0] * time_scaling(total_time, dt*i))) # Shape : (3, 3)
            traj[0:-1, -1] = p_start + (time_scaling(total_time, dt*i) * (p_end - p_start)) # Shape : (3, )
            
            trajectory.append(traj) # Append
        
        # Return
        return np.array(trajectory)
    
    ### Dynamics with Trajectory
    def euler_Step(self, theta_array, d_theta_array, d2_theta_array, dt):
        """Computes the Euler Integration; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array    : The Joint Variables;                      ===> Type : Numpy Array; Shape : (n_joints, ).
                      d_theta_array  : The derivative of Joint Variables;        ===> Type : Numpy Array; Shape : (n_joints, ).
                      d2_theta_array : The second derivative of Joint Variables; ===> Type : Numpy Array; Shape : (n_joints, ).
                      dt : The time step; ===> Type : Float.
           Return:
           ===========
                      theta_array_next   : The next time step Joint Variables;                ===> Type : Numpy Array; Shape : (n_joints, ).
                      d_theta_array_next : The next time step derrivative of Joint Variables; ===> Type : Numpy Array; Shape : (n_joints, ).
        """
        theta_array_next   = theta_array + (d_theta_array * dt)   # Theta next 
        d_theta_array_next = d_theta_array +(d2_theta_array * dt) # Derivative Theta next
        
        # Return
        return theta_array_next, d_theta_array_next
    
    
    def get_Derivatives_Trajectory(self, theta_array_traj, total_time):
        """Computes the First and Decond Derivatives of a given Trajectory; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array_traj : The joint values with trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints).
                      total_time  : The Total Time of Trajectory; ===> Type : Float.
           Return:
           ===========
                      d_theta_array_traj  : The 1st Derivative joint values for trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints,).
                      d2_theta_array_traj : The 2nd Derivativejoint values for trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints,).
        """
        n_points = theta_array_traj.shape[0] # N_Points in trajectory
        n_joints = theta_array_traj.shape[1] # Number of Joints
        dt = total_time/(n_points - 1.0)     # Time Step
        d_theta_array_traj  = np.zeros((n_points, n_joints)) # The 1st Derivative joint values for trajectory
        d2_theta_array_traj = np.zeros((n_points, n_joints)) # The 2nd Derivative joint values for trajectory
        
        for i in range(n_points):
            d_theta_array_traj[i+1, :]  = (theta_array_traj[i+1, :] - theta_array_traj[i, :]) / dt
            d2_theta_array_traj[i+1, :] = (d_theta_array_traj[i+1, :] - d_theta_array_traj[i, :]) / dt
            
        # Return
        return d_theta_array_traj, d2_theta_array_traj
    
    
    def inverse_Dynamics_Trajectory(self, theta_array_traj, d_theta_array_traj, d2_theta_array_traj, 
                                    gravity_array, wrench_vector, M_array, inertia_array, screw_array):
        """Computes the Inverse Dynamics in the Space Frame for Open Chains given a trajectory; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array_traj    : The joint values with trajectory;                  ===> Type : Numpy Array; Shape : (n_points, n_joints,).
                      d_theta_array_traj  : The 1st Derivative joint values for trajectory;    ===> Type : Numpy Array; Shape : (n_points, n_joints,).
                      d2_theta_array_traj : The 2nd Derivative joint values for trajectory;    ===> Type : Numpy Array; Shape : (n_points, n_joints,).
                      gravity_array       : The gravity vector @ {0};                          ===> Type : Numpy Array; Shape : (3,).
                      wrench_vector       : The Wrench at the end-effector for the trajectory; ===> Type : Numpy Array; Shape : (n_points, 6).
                      M_array             : The Transformation Matrix at Zero Position for each Link including 
                                            end-effector w.r.t base {0};      ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array       : The Spatial Intertial Matrix;     ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array         : The Screw Axes Array of joints;   ===> Type : Numpy Array; Shape : (n_joints, 6).
           Return:
           ===========
                      torque_array_traj   : The Torques at the joint for the trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints).
        =====================================================================================================================
        """
        n_points = theta_array_traj.shape[0] # N_Points in trajectory
        n_joints = theta_array_traj.shape[1] # Number of Joints
        torque_array_traj = np.zeros((n_points, n_joints)) # Torque Matrix
        
        # Compute Torque of Joints for all points in the trajectory
        for i in range(n_points):
            torque_array_traj[i, :] = self.inverse_Dynamics(theta_array_traj[i, :], d_theta_array_traj[i, :], \
                                                            d2_theta_array_traj[i, :], gravity_array, \
                                                            wrench_vector[i, :], M_array, inertia_array, screw_array)
        # Return 
        return torque_array_traj
    
    
    def forward_Dynamics_Trajectory(self, theta_array, d_theta_array, torque_array_traj, gravity_array,
                                    wrench_vector, M_array, inertia_array, screw_array, dt, int_res):
        """Simulates the motion of a Serial Chain; ===> Type : Member Function.
           Parameters:
           ===========
                      theta_array       : The initial Joint values; ===> Type : Numpy Array; Shape : (n_joints, ).
                      d_theta_array     : The initial Joint rates;  ===> Type : Numpy Array; Shape : (n_joints, ).
                      torque_array_traj : The Torques at the joint for the trajectory;       ===> Type : Numpy Array; Shape : (n_points, n_joints).
                      gravity_array     : The gravity vector @ {0};                          ===> Type : Numpy Array; Shape : (3,).
                      wrench_vector     : The Wrench at the end-effector for the trajectory; ===> Type : Numpy Array; Shape : (n_points, 6).
                      M_array           : The Transformation Matrix at Zero Position for each Link including 
                                          end-effector w.r.t base {0};      ===> Type : Numpy Array; Shape : (n_joints + 1, 4, 4).
                      inertia_array     : The Spatial Intertial Matrix;     ===> Type : Numpy Array; Shape : (n_joints, 6, 6).
                      screw_array       : The Screw Axes Array of joints;   ===> Type : Numpy Array; Shape : (n_joints, 6).
                      dt      : The Time Step ; ===> Type : Float.
                      int_res : The Integration Resolution between 1 time step; ===> Type : Int.
           Return:
           ===========
                      theta_matrix   : The Joint Values for the Trajectory; ===> Type : Numpy Array; Shape : (n_points, n_joints).
                      d_theta_matrix : The joint Rates for the Trajectory;  ===> Type : Numpy Array; Shape : (n_points, n_joints).
        """
        n_points = theta_array.shape[0] # N_Points in trajectory
        n_joints = theta_array.shape[1] # Number of Joints
        theta_matrix   = np.zeros((n_points, n_joints))
        theta_matrix[0, :] = theta_array
        d_theta_matrix = np.zeros((n_points, n_joints))
        d_theta_matrix[0, :] = d_theta_array

        # Compute Joint Values and Rates
        for i in range(n_points):
            for j in range(int_res):
                d2_theta_array = self.forward_Dynamics(theta_array, d_theta_array, torque_array_traj[i, :],
                                 gravity_array, wrench_vector[i, :], M_array, inertia_array, screw_array)    # Shape : (n_joints, )
                
                theta_array, d_theta_array = self.euler_Step(theta_array, d_theta_array, d2_theta_array, dt) # Shape : (n_joints, ), (n_joints, )
                
            theta_matrix[i+1, :]   = theta_array   # Shape : (n_joints, )
            d_theta_matrix[i+1, :] = d_theta_array # Shape : (n_joints, )
        
        # Return 
        return theata_matrix, d_theta_matrix