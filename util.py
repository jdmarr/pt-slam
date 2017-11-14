import numpy as np
from scipy.linalg import logm, block_diag
from scipy.integrate import ode
import copy as copy


def ang_2_norm(alpha, beta):  # convert unit normal spec'd in cylindrical coords to a euclidean vector

    # NOTE: beta is defined s.t. b=0 implies z=0. This is contrary to many utilizations of spherical coordinates
    # where b=0 implies z=1 (i.e., the North pole)
    normal_euclid = np.array([np.cos(beta)*np.cos(alpha), np.cos(beta)*np.sin(alpha), np.sin(beta)])

    return normal_euclid

def norm_2_ang(x, y, z): # inverse of ang_2_norm. beta is defined as in ang_2_norm

    normal_ang = np.array([np.arctan2(y, x), np.arctan(z/(np.sqrt(np.square(x) + np.square(y))))])

    return normal_ang


# transform angle-distance planes from a sensor frame to the base frame
def transform_plane(plane, sensor_rot, sensor_trans):

    rotated_normal = np.dot(sensor_rot, ang_2_norm(plane[0], plane[1]))

    angles_prime = norm_2_ang(rotated_normal[0], rotated_normal[1], rotated_normal[2])
    #if np.abs(angles_prime[0]) > 1.4:
	#	angles_prime[0] = np.pi / 2

    d_prime = np.dot(sensor_trans.T, rotated_normal) + plane[2]

    if d_prime < 0:
        d_prime = -d_prime
        angles_prime = norm_2_ang(-rotated_normal[0], -rotated_normal[1], -rotated_normal[2])

    return np.array([angles_prime[0], angles_prime[1], d_prime])

def is_same_plane(plane1, plane2, thresh_ang = 0.1, thresh_dist = 0.7):

	# construct unit normals from the first 3 elements of each plane
	unorm1 = ang_2_norm(plane1[0], plane1[1])
	unorm2 = ang_2_norm(plane2[0], plane2[1])

	# get distance from the origin for each plane
	d1 = plane1[2]
	d2 = plane2[2]

	if np.arccos(np.abs(np.dot(unorm1, unorm2))) < thresh_ang and np.abs(d1-d2) < thresh_dist:
		return 1
	else:
		return 0


def iepf(point_list, threshold):

    d = np.abs(np.cross(point_list[:] - point_list[0], point_list[:] - point_list[-1])) / np.linalg.norm(point_list[-1] - point_list[0])

    if np.amax(d) > threshold:
        rec_endpoint_list, rec_numpoint_list = iepf(point_list[0:np.argmax(d) + 1,:], threshold)
        endpoint_list = rec_endpoint_list
        numpoint_list = rec_numpoint_list
        rec_endpoint_list, rec_numpoint_list = iepf(point_list[np.argmax(d):, :], threshold)
        endpoint_list = np.append(endpoint_list, rec_endpoint_list[1:, :], axis=0)
                                                     # don't add the first element in the second half as it is the same
                                                     # as the last element in the first half
        numpoint_list = np.append(numpoint_list, rec_numpoint_list, axis=0)
    else:
        endpoint_list = np.array([point_list[0, :],point_list[-1, :]])
        numpoint_list = np.array([point_list.shape[0]])

    return endpoint_list, numpoint_list

def build_cov_3d(sig_squared_p, c_o_m, eigvec0, eigval1, eigval2):
	c_o_m_normalized = c_o_m / np.linalg.norm(c_o_m)
	sig_squared_d = sig_squared_p * np.abs(np.dot(eigvec0, c_o_m_normalized))

	sig_d = np.sqrt(sig_squared_d)

	sig_squared_alpha = np.square(np.arctan(sig_d/np.sqrt(eigval1)))
	sig_squared_beta = np.square(np.arctan(sig_d / np.sqrt(eigval2)))

	cov = np.diag(sig_squared_alpha, sig_squared_beta, sig_squared_d)

	return cov

def build_cov_2d(sig_squared_p, c_o_m, eigvec0, eigval1):
	c_o_m_aug = np.array([c_o_m[0], c_o_m[0], 0])  #augment the c_o_m with a zero z vector component
	c_o_m_normalized = c_o_m_aug / np.linalg.norm(c_o_m_aug)
	sig_squared_d = sig_squared_p * np.abs(np.dot(eigvec0, c_o_m_normalized))

	sig_d = np.sqrt(sig_squared_d)

	sig_squared_beta = np.square(np.arctan(sig_d/np.sqrt(eigval1)))
	sig_squared_alpha = 0.0000000001  # arbitrarily small since beta IN THE WORLD FRAME is always 0 in the 2d case
	# This is confusing but beta in world frame = alpha in plane frame because of how the coordinate axes are defined

	cov = np.diag(np.array([sig_squared_alpha, sig_squared_beta, sig_squared_d]))

	return cov

def skew(p):
	p_skew = np.array([[0,-1*p[2],p[1]],[p[2],0,-1*p[0]],[-1*p[1],p[0],0]])

	return p_skew


def del_theta(C, alpha, beta):

    rotated_normal = np.dot(C, ang_2_norm(alpha, beta))

    x = rotated_normal[0]
    y = rotated_normal[1]
    z = rotated_normal[2]

    elem11 = (-y/np.square(x))/(1+np.square(y/x))
    elem12 = (1/x)/(1+np.square(y/x))

    elem21 = (-x*z/np.power(np.square(x)+np.square(y), 3/2))/(1 + np.square(z)/(np.square(x)+np.square(y)))
    #elem22 = (-y*z/np.power(np.square(x)+np.square(y), 3/2))/(1 + np.square(z)/(np.square(x)+np.square(y)))
    elem22 = (y/x)*elem21
    elem23 = (1/np.sqrt(np.square(x)+np.square(y)))/(1 + np.square(z)/(np.square(x)+np.square(y)))


    del_theta_del_big_x = np.array([[elem11, elem12, 0], [elem21, elem22, elem23]])

    del_theta_del_rot = -1*np.dot(del_theta_del_big_x, skew(rotated_normal))

    del_theta_del_alpha = np.dot(del_theta_del_big_x, np.dot(C, np.array([[-np.cos(beta)*np.sin(alpha)], [np.cos(beta)*np.cos(alpha)], [0]])))
    del_theta_del_beta = np.dot(del_theta_del_big_x, np.dot(C, np.array([[-np.sin(beta)*np.cos(alpha)], [-np.sin(beta)*np.sin(alpha)], [np.cos(beta)]])))

    # del_theta = [del_theta_del_rot, del_theta_del_alpha, del_theta_del_beta]
    #               ^2x3                ^2x1                2x1
    # del_theta_del_trans and del_theta_del_d are 0
    return del_theta_del_rot, del_theta_del_alpha, del_theta_del_beta


def del_d(C, r, alpha, beta):

    del_d_del_trans = np.dot(C, ang_2_norm(alpha, beta))

    del_d_del_rot = -1*np.dot(r, skew(del_d_del_trans))  # reuse the fact that del_d_del_r = C*N(alpha, beta)

    del_d_del_alpha = np.dot(r, np.dot(C,
        np.array([[-np.cos(beta) * np.sin(alpha)], [np.cos(beta) * np.cos(alpha)], [0]])))
    del_d_del_beta = np.dot(r, np.dot(C,
        np.array([[-np.sin(beta) * np.cos(alpha)], [-np.sin(beta) * np.sin(alpha)], [np.cos(beta)]])))

    # del_theta = [del_d_del_trans, del_d_del_rot, del_d_del_alpha, del_d_del_beta, del_d_del_d_nought]
    #               ^1x3                ^1x3            ^1x1            ^1x1            ^1x1
    # del_d_del_d_nought = 1

    return del_d_del_trans, del_d_del_rot, del_d_del_alpha, del_d_del_beta, 1

def build_jacobian(C, r, alpha, beta, d_nought):
    dtd_rot, dtda, dtdb = del_theta(C, alpha, beta)
    ddd_trans, ddd_rot, ddda, dddb, dddd = del_d(C, r, alpha, beta)

    # build the 3x9 Jacobian matrix (derivatives of 3 plane params wrt 9 inputs (6DOF transform + 3 input plane params))
    jacobian = np.array([[0, 0, 0, dtd_rot[0, 0], dtd_rot[0, 1], dtd_rot[0, 2], dtda[0], dtdb[0], 0], [0, 0, 0, dtd_rot[1, 0], dtd_rot[1, 1], dtd_rot[1, 2], dtda[1], dtdb[1], 0], [ddd_trans[0], ddd_trans[1], ddd_trans[2], ddd_rot[0], ddd_rot[1], ddd_rot[2], ddda, dddb, dddd]])

    return jacobian[:, 0:6], jacobian[:, 6:9]

def transform_cov(cov_pframe, plane_wframe, c_o_m, cur_pose):
	alpha_w = plane_wframe[0]

	C_1 = np.array([[np.cos(alpha_w), -np.sin(alpha_w), 0], [np.sin(alpha_w), np.cos(alpha_w), 0], [0, 0, 1]])
	C_2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
	C_wp = np.dot(C_1, C_2)

	c_o_m_aug = np.array([[c_o_m[0]], [c_o_m[0]], [0], [1]])

	r_w_pw = np.dot(cur_pose, c_o_m_aug)[0:3]
	r_w_pw.shape = (3)

	tmp, plane_jacobian = build_jacobian(C_wp, r_w_pw, 0, 0, 0)  # the plane parameters in its own frame are always (0, 0, 0)

	cov_wframe = np.dot(plane_jacobian, np.dot(cov_pframe, plane_jacobian.T))

	return cov_wframe

def pose_inv_map(T_mat):

	w, v = np.linalg.eig(T_mat[0:3,0:3])
	

	index = np.argmin(np.absolute(w.imag))
	a = v[:,index].real

	phi = np.arccos( (0.5)*(np.trace(T_mat[0:3,0:3])-1) )
	phi = np.nan_to_num(phi)
	if phi != 0:
		lnT = logm(T_mat)
		th1 = lnT[2,1]
		th2 = lnT[0,2]
		th3 = lnT[1,0]
		rho = lnT[0:3,3]
	else:
		rho = T_mat[0:3,3]
		th1 = 0
		th2 = 0
		th3 = 0

	t_vec = np.array([[rho[0]],[rho[1]],[rho[2]],[th1],[th2],[th3]])

	return t_vec


def rot_inv_map(R_mat):
	w, v = np.linalg.eig(R_mat)

	index = np.argmin(np.absolute(w.imag))
	a = v[:, index].real

	phi = np.arccos((0.5) * (np.trace(R_mat) - 1))
	phi = np.nan_to_num(phi)
	if phi != 0:
		lnR = logm(R_mat)
		th1 = lnR[2, 1]
		th2 = lnR[0, 2]
		th3 = lnR[1, 0]
	else:
		th1 = 0
		th2 = 0
		th3 = 0

	r_vec = np.array([[th1], [th2], [th3]])

	return r_vec


def rot_for_map(r_vec):

	#rotation forward mapping
	a = r_vec#column of rotational deltas
	phi = np.linalg.norm(a)
	a = (1/phi)*a
	a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

	R_mat = np.cos(phi)*np.identity(3)
	R_mat = R_mat + (1-np.cos(phi))*np.dot( a , a.T )
	R_mat = R_mat + np.sin(phi)*a_skew

	return R_mat


def pose_for_map(t_vec):

	#pose forward mapping
	a = t_vec[3:6]#column of rotational deltas
	phi = np.linalg.norm(a)
	if phi != 0:
		a = (1/phi)*a
		a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

		C = np.cos(phi)*np.identity(3)
		C = C + (1-np.cos(phi))*np.dot( a , a.T )
		C = C + np.sin(phi)*a_skew

		J = (1/phi)*np.sin(phi)*np.identity(3)
		J = J + (1 - (1/phi)*np.sin(phi))*np.dot( a , a.T )
		J = J + (1/phi)*(1 - np.cos(phi))*a_skew

		r = np.dot( J , t_vec[0:3] )

		T_mat = np.identity(4)
		T_mat[0:3,0:3] = C
		T_mat[0:3,3:4] = r
	else:
		T_mat = np.identity(4)
		T_mat[0:3,3:4] = t_vec[0:3]

	return T_mat

def invert_transform(T_inv):
	#build a transformation from its inverse (faster than T = np.linalg.inv(T_inv))

	T = np.identity(4)
	T[0:3,0:3] = T_inv[0:3,0:3].T
	T[0:3,3] = -1 * np.dot( T_inv[0:3,0:3].T , T_inv[0:3,3] )

	return T

def rot_2_trans(R_mat):

	T_mat = np.append(R_mat, np.array([0, 0, 0]).reshape(1,3), axis=0)
	T_mat = np.append(T_mat, np.array([[0], [0], [0], [1]]), axis=1)

	return T_mat

def chi_squared_cost(pose_vec, matches_lframe, matches_wframe, new_coms_lframe, covs_pframe, covs_wframe):
	chi_squared = 0

	pose_vec_aug = np.array([pose_vec[0], pose_vec[1], 0, 0, 0, pose_vec[2]])

	trans_vec = pose_vec_aug[0:3]
	Rot_mat = rot_for_map(np.array([pose_vec_aug[3:6]]).T)
	cur_pose = np.append(np.append(Rot_mat, np.array([trans_vec]).T, axis=1), np.array([[0, 0, 0, 1]]), axis=0)

	for i in range(matches_lframe.shape[0]):
		new_plane_wframe = transform_plane(matches_lframe[i], Rot_mat, trans_vec)
		e = (new_plane_wframe - matches_wframe[i]).T
		if np.abs(e[0]) > 3*np.pi/2:
			e[0] = np.abs(e[0]) - 2*np.pi
		elif np.abs(e[0]) > np.pi/2:
			e[0] = np.abs(e[0]) - np.pi
			e[2] = new_plane_wframe[2] + matches_wframe[i, 2]
		cov_sum = transform_cov(covs_pframe[3*i:3+3*i, 0:3], new_plane_wframe, new_coms_lframe[i], cur_pose) + covs_wframe[3*i:3+3*i, 0:3]
		chi_squared = chi_squared + np.dot(e.T, np.linalg.solve(cov_sum, e))

	return chi_squared

def merge_planes(new_plane_lframe, new_cov_pframe, new_com_lframe, old_plane_wframe, old_cov_wframe, cur_pose):

	new_plane_wframe = transform_plane(new_plane_lframe, cur_pose[0:3,0:3], cur_pose[0:3,3])
	new_cov_wframe = transform_cov(new_cov_pframe, new_plane_wframe, new_com_lframe, cur_pose)

	if (new_plane_wframe[0] - old_plane_wframe[0]) > np.pi/2:
		new_plane_wframe[0] = new_plane_wframe[0] - np.sign(new_plane_wframe[0])*np.pi
		new_plane_wframe[2] = -new_plane_wframe[2]

	new_cov_inv = np.linalg.inv(new_cov_wframe)
	old_cov_inv = np.linalg.inv(old_cov_wframe)
	cov_merged = np.linalg.inv(new_cov_inv + old_cov_inv)

	plane_merged = np.dot(cov_merged, np.dot(new_cov_inv, new_plane_wframe) + np.dot(old_cov_inv, old_plane_wframe))

	return plane_merged, cov_merged

def Xvec_ode(t, Xt, a_i, w_i, g_w):
	Xdot = np.zeros((9))

	angs = Xt[3:6]
	angs.shape = (3,1)
	R_w_i = rot_for_map(angs)

	# deriv of position = velocity
	Xdot[0:3] = Xt[6:9]

	# deriv of euler angles = angular velocity (need to transform it to world frame)
	# Note that in the 2D case w_w = w_i because of the speicific form of R_w_i
	w_i.shape = (3,1)
	Xdot[3:6] = np.dot(R_w_i, w_i)[:,0]

	# deriv of velocity = acceleration (need to transform it to world frame)
	a_i.shape = (3,1)
	Xdot[6:9] = np.dot(R_w_i, a_i)[:,0] + g_w

	return Xdot

def imu_integrate_basic(ti, tf, Xti, a_i, w_i, g_w):

	if np.abs(tf - ti) < 0.0000000000001:
		return Xti

	r = ode(Xvec_ode).set_integrator('dopri5')
	r.set_initial_value(Xti, ti).set_f_params(a_i, w_i, g_w)

	return r.integrate(tf)


	return np.dot(L.T, delta)[:,0]
