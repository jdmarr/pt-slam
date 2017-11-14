""""""

# Places that the user should modify are marked with a "# USER:" comment

# wframe - world (inertial) reference frame
# lframe - Lidar sensor base frame
# iframe - IMU sensor base frame

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import copy as copy
from util import *

# USER: Set plotting flags if desired
disp_scan = 0 # Plot the Lidar scan at each timestep?
disp_poses = 0 # Plot the Lidar position as it evolves?

# USER: Modify file paths as appropriate
rng_data = np.loadtxt('sensor-data/ranges-wimubias.txt')[:, 1:] # import the Lidar data
# import the ground truth IMU pose changes (that is delta_x, delta_y, and delta_theta, NOT INERTIAL DATA)
imu_data_gt = np.loadtxt('sensor-data/imuRelPoses-gt.txt')[:,0:4]
imu_meas = np.loadtxt('sensor-data/imuMeas-biased.txt')[:,:] #import the inertial data

N = rng_data.shape[0] - 1 # number of timesteps

# USER: Set the angles at which your Lidar measures
th_beam_lframe = np.linspace(-np.pi/2, np.pi/2, 301)

# USER: Set the ground truth pose of the lidar at the first moment
# They are set to match the vehicle position from the first moment, but could equivalently be (0,0)
# if we initialized our inertial frame at the first pose of the Lidar
x_lidar = 14.067162
y_lidar = 8.774753
th_lidar = -2.59593853

# lidar pose matrix at time k
T_w_lk = np.identity(4)
T_w_lk[0,3] = x_lidar
T_w_lk[1,3] = y_lidar
# theta is specified as the world to lidar rotation angle, while the matrix is the lidar to world transform. So the
# rotation matrix is built as the transpose of what one might expect (+sine terms switched with -sine terms)
T_w_lk[0:2,0:2] = np.array([[np.cos(th_lidar), -np.sin(th_lidar)],[np.sin(th_lidar), np.cos(th_lidar)]])
T_lk_w = invert_transform(T_w_lk)

# initialize ground truth pose
T_w_lk_gt = copy.copy(T_w_lk)
T_lk_w_gt = copy.copy(T_lk_w)

# USER: Set the initial calibration parameter guesses.
x_calib = 0.25
y_calib = 0.15
th_calib = 0

# Calibration matrix (IMU -> Lidar transform)
T_l_i = np.identity(4)
T_l_i[0,3] = x_calib
T_l_i[1,3] = y_calib
T_l_i[0:2,0:2] = np.array([[np.cos(th_calib), -np.sin(th_calib)],[np.sin(th_calib), np.cos(th_calib)]])
T_i_l = invert_transform(T_l_i)  # preallocate the Lidar -> IMU transform since we'll need it many times

# USER: Set the ground truth calibration matrix to allow comparison of the estimated quantities with ground truth
x_calib_gt = 0.2
y_calib_gt = 0.2
th_calib_gt = 0

T_l_i_gt = np.identity(4)
T_l_i_gt[0,3] = x_calib_gt
T_l_i_gt[1,3] = y_calib_gt
T_l_i_gt[0:2,0:2] = np.array([[np.cos(th_calib_gt), -np.sin(th_calib_gt)],[np.sin(th_calib_gt), np.cos(th_calib_gt)]])
T_i_l_gt = invert_transform(T_l_i_gt)  # preallocate

# IMU pose matrix at time k (for visualization)
T_w_ik = invert_transform(np.dot(T_i_l_gt, T_lk_w))
T_w_ik_init = copy.copy(T_w_ik)
T_w_ikm1 = copy.copy(T_w_ik) # T_w_i(k-1)

# USER: Number of measurements between stored poses. In our data, we optimize Lidar pose every 1s, and our IMU operates at 100 Hz => the rate is 100*1
imu_rate = 100

# Need a ground truth value for IMU initial velocity. Keep the platform still at the start of data collection to make it 0.
imu_init_vel = np.array([0, 0, 0]) 
# Combine the IMU's 1) position, 2) R-P-Y angles, 3) linear velocity into a 9-element state
imu_init_state = np.append(np.dot(T_w_lk, T_l_i)[0:3,3], np.append(rot_inv_map(np.dot(T_w_lk, T_l_i)[0:3,0:3]), imu_init_vel))
imu_state = copy.copy(imu_init_state)

# stores the x-y coordinates of Lidar points, in the Lidar measurement frame
point_list = np.array([])
point_list.shape = (0, 2)

# Set of planes - makes up our SLAM map
plane_list = np.array([])
plane_list.shape = (0, 3)

# array for holding the set of plane covariance matrices
cov_list = np.array([])
cov_list.shape = (0, 3)

# arrays for plotting the estimated and ground truth lidar position
x_est = np.array([])
y_est = np.array([])
x_gt = np.array([])
y_gt = np.array([])
x_unopt = np.array([])
y_unopt = np.array([])
x_worstcase = np.array([])
y_worstcase = np.array([])

error_vals = np.zeros((6,N)) # array of errors in each parameter of the Lidar pose at each timestep

# Loop over Lidar points from the current scan and add them to the list as (x,y) pairs in Lidar frame
for j in range(rng_data.shape[1]):
    rng_val = rng_data[0, j]
    point = np.array([rng_val * np.cos(th_beam_lframe[j]), rng_val * np.sin(th_beam_lframe[j])])
    point.shape = (1, 2)
    point_list = np.append(point_list, point, axis=0)

if disp_scan:
    # convert points to world frame for plotting if desired
    point_list_aug = np.append(np.append(point_list, np.zeros((point_list.shape[0],1)), axis=1), np.ones((point_list.shape[0],1)), axis=1)
    point_list_wframe = np.dot(T_w_lk, point_list_aug.T).T
    plt.plot(point_list_wframe[:, 0], point_list_wframe[:, 1], '.', markersize=5)
    plt.xlim((-1, 21))
    plt.ylim((-1, 21))
    plt.show()

# Construct initial set of planes from the first Lidar scan using iterative end-point filter
endpts, numlnpts = iepf(point_list, 0.01)
start = 0
for l in range(len(numlnpts)):
    end = start + numlnpts[l] - 1

    if numlnpts[l] > 5:  # if iepf picks out a "line" with <5 pts, ignore it
        start_pt = point_list[start]
        end_pt = point_list[end]

        m = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0]) # slope of the line in x-y plane
        b = start_pt[1] - m*start_pt[0] # y intercept

        plane_normal = np.array([1, -1/m, 0])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # construct plane with two angles and a distance
        d = np.dot(plane_normal, np.array([start_pt[0], start_pt[1], 0]))
        if d < 0: # constrain the distance to always be positive
            plane_normal = -plane_normal
            d = np.dot(plane_normal, np.array([start_pt[0], start_pt[1], 0]))  # could replace this with d = -d
        plane_angles = norm_2_ang(plane_normal[0], plane_normal[1], plane_normal[2])
        plane_lframe = np.array([plane_angles[0], plane_angles[1], d])
        plane_wframe = transform_plane(plane_lframe, T_w_lk[0:3,0:3], T_w_lk[0:3,3])

        #Get the covariance matrix for the plane parameters in the world frame
        c_o_m = np.mean(point_list[start:end+1], axis=0)  # center of mass of the plane
        point_spread = np.cov(point_list[start:end + 1].T)
        eigval, eigvec = np.linalg.eig(point_spread)
        plane_cov = build_cov_2d(0.01, c_o_m, plane_normal, np.amax(eigval))
        cov_wframe = transform_cov(plane_cov, plane_wframe, c_o_m, T_w_lk)

        # Check if we've seen this plane before
        seen_flag = 0
        for p in range(plane_list.shape[0]):
            if is_same_plane(plane_wframe, plane_list[p]):
                seen_flag = 1

        if not seen_flag: # if we haven't seen the plane before, add it to the list of planes
            plane_wframe.shape = (1,3)
            plane_list = np.append(plane_list, plane_wframe, axis = 0)
            cov_list = np.append(cov_list, cov_wframe, axis=0)

    start = end

for i in range(N): # main SLAM loop

    # when we encounter new planes, we don't want to add them to the map until we've optimized the pose at that timestep
    # This requires us to temporarily store the plane parameters, covariance and center of mass until after optimization
    plane_tmp_list = np.array([])
    plane_tmp_list.shape = (0, 3)
    cov_tmp_list = np.array([])
    cov_tmp_list.shape = (0, 3)
    c_o_m_tmp_list = np.array([])
    c_o_m_tmp_list.shape = (0, 2)

    # initialize all the arrays that we will pass as arguments to the cost function
    plane_matches_lframe = np.array([])
    plane_matches_lframe.shape = (0,3)
    plane_matches_wframe = np.array([])
    plane_matches_wframe.shape = (0, 3)
    cov_matches_pframe = np.array([])
    cov_matches_pframe.shape = (0, 3)
    cov_matches_wframe = np.array([])
    cov_matches_wframe.shape = (0, 3)
    c_o_m_matches_lframe = np.array([])
    c_o_m_matches_lframe.shape = (0, 2)

    # Reset point list at each timestep
    point_list = np.array([])
    point_list.shape = (0, 2)

    # list of flags for which planes match those already in the map
    p_match_list = np.array([])

    # loop over and integrate the IMU measurements we obtained between our pose updates to get a first estimate of the new IMU pose
    for j in range(imu_rate):
        imu_state = imu_integrate_basic(imu_meas[imu_rate*i+j,0], imu_meas[imu_rate*i+j+1,0], imu_state, imu_meas[imu_rate*i+j,1:4], imu_meas[imu_rate*i+j,4:7], np.array([0, 0, 0]))
    
    angs = imu_state[3:6]
    angs.shape = (3,1)
    T_w_ik = np.append(np.append(rot_for_map(angs), np.array([[imu_state[0]], [imu_state[1]], [imu_state[2]]]), axis=1), np.array([[0, 0, 0, 1]]), axis=0)
   
    Xi_imu = np.dot(invert_transform(T_w_ik), T_w_ikm1) # essentially T_ik_i(k-1), the matrix that implements the IMU pose change
    T_w_ikm1 = copy.copy(T_w_ik)

    Xi_lidar = np.dot(T_l_i, np.dot(Xi_imu, T_i_l))  # T_lk_l(k-1) = T_l_i * T_ik_i(k-1) * T_i_l

    T_lk_w = np.dot(Xi_lidar, T_lk_w)
    T_w_lk = invert_transform(T_lk_w)

    Xi_imu_gt = np.identity(4)  # ground truth value of T_ik_i(k-1)
    Xi_imu_gt[0, 3] = -imu_data_gt[i, 1]
    Xi_imu_gt[1, 3] = -imu_data_gt[i, 2]
    Xi_imu_gt[0:2, 0:2] = np.array(
        [[np.cos(imu_data_gt[i, 3]), np.sin(imu_data_gt[i, 3])],
         [-np.sin(imu_data_gt[i, 3]), np.cos(imu_data_gt[i, 3])]])

    Xi_lidar_gt = np.dot(T_l_i_gt, np.dot(Xi_imu_gt, T_i_l_gt))  # ground truth T_lk_l(k-1)

    T_lk_w_gt = np.dot(Xi_lidar_gt, T_lk_w_gt)
    T_w_lk_gt = invert_transform(T_lk_w_gt)

    """DUPLICATE CODE FROM THE FIRST TIMESTEP STARTS"""

    # Loop over Lidar points from the current scan and add them to the list as (x,y) pairs in Lidar frame
    for j in range(rng_data.shape[1]):
        rng_val = rng_data[i+1, j]
        point = np.array([rng_val * np.cos(th_beam_lframe[j]), rng_val * np.sin(th_beam_lframe[j])])
        point.shape = (1, 2)
        point_list = np.append(point_list, point, axis=0)

    if disp_scan:
        # Tranform points to world frame for plotting if desired
        point_list_aug = np.append(np.append(point_list, np.zeros((point_list.shape[0], 1)), axis=1),
                                   np.ones((point_list.shape[0], 1)), axis=1)
        point_list_wframe = np.dot(T_w_lk, point_list_aug.T).T
        plt.plot(point_list_wframe[:, 0], point_list_wframe[:, 1], '.', markersize=5)
        plt.xlim((-1, 21))
        plt.ylim((-1, 21))
        plt.show()

    # Extract line segments
    endpts, numlnpts = iepf(point_list, 0.01)
    start = 0
    for l in range(len(numlnpts)): # Loop over extracted line segments
        end = start + numlnpts[l] - 1

        if numlnpts[l] > 5:  # if iepf picks out a "line" with <5 pts, ignore it
            start_pt = point_list[start]
            end_pt = point_list[end]

            m = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0])
            b = start_pt[1] - m * start_pt[0]

            plane_normal = np.array([1, -1 / m, 0])
            plane_normal = plane_normal / np.linalg.norm(plane_normal)

            # construct plane with two angles and a distance
            d = np.dot(plane_normal, np.array([start_pt[0], start_pt[1], 0]))
            if d < 0: # constrain the distance to always be positive
                plane_normal = -plane_normal
                d = np.dot(plane_normal, np.array([start_pt[0], start_pt[1], 0]))  # could replace this with d = -d
            plane_angles = norm_2_ang(plane_normal[0], plane_normal[1], plane_normal[2])
            plane_lframe = np.array([plane_angles[0], plane_angles[1], d])
            plane_wframe = transform_plane(plane_lframe, T_w_lk[0:3, 0:3], T_w_lk[0:3, 3])

            # Get the covariance matrix for the plane parameters in the world frame
            c_o_m = np.mean(point_list[start:end + 1], axis=0)  # center of mass of the plane
            point_spread = np.cov(point_list[start:end + 1].T)
            eigval, eigvec = np.linalg.eig(point_spread)
            plane_cov = build_cov_2d(0.01, c_o_m, plane_normal, np.amax(eigval))

            # Check if we've seen this plane before
            seen_flag = 0
            for p in range(plane_list.shape[0]):
                if is_same_plane(plane_wframe, plane_list[p]):
                    # If we've seen this plane before, we add the matched planes to the list to constrain Lidar pose
                    seen_flag = 1

                    """DUPLICATE CODE FROM THE FIRST TIMESTEP ENDS"""

                    p_match_list = np.append(p_match_list, p)
                    plane_lframe.shape = (1,3)
                    plane_matches_lframe = np.append(plane_matches_lframe, plane_lframe, axis=0)
                    plane_matches_wframe = np.append(plane_matches_wframe, plane_list[p:p+1], axis=0)
                    cov_matches_pframe = np.append(cov_matches_pframe, plane_cov, axis=0)
                    cov_matches_wframe = np.append(cov_matches_wframe, cov_list[3*p:3+3*p, 0:3], axis=0)

                    # We need the centers of mass (in the Lidar frame) to allow us to transform the covariance of the
                    # new plane to the world frame, within the cost function
                    c_o_m_matches_lframe = np.append(c_o_m_matches_lframe, [c_o_m], axis=0)
                    break

            if not seen_flag:
                # store the plane in the Lidar frame. We'll add it to the world after correcting the Lidar pose
                plane_lframe.shape = (1, 3)
                plane_tmp_list = np.append(plane_tmp_list, plane_lframe, axis=0)
                c_o_m.shape = (1, 2)
                c_o_m_tmp_list = np.append(c_o_m_tmp_list, c_o_m, axis=0)
                cov_tmp_list = np.append(cov_tmp_list, plane_cov, axis=0)

        start = end

    # Optimize for T_w_lk
    pose_vec = np.append(T_w_lk[0:2,3], rot_inv_map(T_w_lk[0:3, 0:3])[2]) # Our starting point for the optimization (x, y, theta)

    cost = chi_squared_cost(pose_vec, plane_matches_lframe, plane_matches_wframe, c_o_m_matches_lframe, cov_matches_pframe, cov_matches_wframe)
    sol = sp.optimize.minimize(chi_squared_cost, pose_vec, args=(plane_matches_lframe, plane_matches_wframe, c_o_m_matches_lframe, cov_matches_pframe, cov_matches_wframe))
    cost_opt = chi_squared_cost(sol.x, plane_matches_lframe, plane_matches_wframe, c_o_m_matches_lframe,
                            cov_matches_pframe, cov_matches_wframe)

    sol_pose_vec = np.array([sol.x[0], sol.x[1], 0, 0, 0, sol.x[2]])
    T_w_lk = np.append(np.append(rot_for_map(np.array([sol_pose_vec[3:6]]).T), np.array([sol_pose_vec[0:3]]).T, axis=1), np.array([[0, 0, 0, 1]]), axis=0)
    T_lk_w = invert_transform(T_w_lk)

    T_w_lkm1 = copy.copy(T_w_lk)
    T_w_lkm1_gt = copy.copy(T_w_lk_gt)

    # After optimizing the pose:
    # 1) Merge the matched planes, ostensibly improving the estimates of their parameters
    for p in range(plane_matches_lframe.shape[0]):
        p_match = p_match_list[p]
	# plane merging commented out atm as leaving it out can sometimes improve results
        # plane_list[p_match], cov_list[3*p_match:3+3*p_match, 0:3] = merge_planes(plane_matches_lframe[p], cov_matches_pframe[3*p:3+3*p,0:3], c_o_m_matches_lframe[p], plane_matches_wframe[p], cov_matches_wframe[3*p:3+3*p,0:3], T_w_lk)

    # 2) Add the stored new planes to the map by transforming them from Lidar -> world frame
    for j in range(plane_tmp_list.shape[0]):
        plane_wframe = transform_plane(plane_tmp_list[j], T_w_lk[0:3, 0:3], T_w_lk[0:3, 3])
        cov_wframe = transform_cov(cov_tmp_list[3 * j:3 + 3 * j, 0:3], plane_wframe, c_o_m_tmp_list[j], T_w_lk)
        plane_wframe.shape = (1,3)
        plane_list = np.append(plane_list, plane_wframe, axis=0)
        cov_list = np.append(cov_list, cov_wframe, axis=0)

    # Store some information for visualizing the pose error
    error_mat = np.dot(T_w_lk, T_lk_w_gt)
    error_vals[:, i:i + 1] = pose_inv_map(error_mat)
    x_est = np.append(x_est, T_w_lk[0, 3])
    y_est = np.append(y_est, T_w_lk[1, 3])
    x_gt = np.append(x_gt, T_w_lk_gt[0, 3])
    y_gt = np.append(y_gt, T_w_lk_gt[1, 3])
    x_unopt = np.append(x_unopt, np.dot(T_w_ik, T_i_l_gt)[0, 3])
    y_unopt = np.append(y_unopt, np.dot(T_w_ik, T_i_l_gt)[1, 3])
    x_worstcase = np.append(x_worstcase, np.dot(T_w_ik, T_i_l)[0, 3])
    y_worstcase = np.append(y_worstcase, np.dot(T_w_ik, T_i_l)[1, 3])

    if disp_poses:
        # Plot the estimated and ground truth positions up to now to see how they're evolving
        plt.plot(x_est, y_est, 'o', markersize=5, color='red')
        plt.plot(x_gt, y_gt, 'o', markersize=5, color='green')
        plt.xlim((-1, 21))
        plt.ylim((-1, 21))
        plt.show()

plt.plot(x_est, y_est, '-o', markersize=5, color='red')
plt.plot(x_gt, y_gt, '-o', markersize=5, color='green')
plt.xlim((9, 16))
plt.ylim((4, 11))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend(('Est\'d Lidar postion','Gound truth'))
plt.title('SLAM Results')
plt.show()

plt.plot(x_est, y_est, '-o', markersize=5, color='red')
plt.plot(x_unopt, y_unopt, '-o', markersize=5, color='green')
plt.plot(x_worstcase, y_worstcase, '-o', markersize=5, color='blue')
plt.xlim((9, 16))
plt.ylim((4, 11))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend(('SLAM','Dead reckoning, perfect calibration','Dead reckoning, poor calibration'))
plt.title('SLAM Improvement over dead reckoning')
plt.show()

plt.plot(range(len(x_est)), x_est - x_gt)
plt.xlabel('Timestep')
plt.ylabel('Error in Lidar x coordinate (m)')
plt.show()

plt.plot(range(len(x_est)), y_est - y_gt)
plt.xlabel('Timestep')
plt.ylabel('Error in Lidar y coordinate (m)')
plt.show()

rmse = np.sqrt(np.mean(np.square(error_vals), axis=1))
print('RMSE =' + str(rmse))
