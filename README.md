Planar Target SLAM
==================

Author: Jordan Marr
E-mail: jordan(.)marr(at)robotics(.)utias(.)utoronto(.)ca 
Date:     July 19th, 2017 
Modified: November 14th, 2017

Python scripts for performing SLAM with an IMU-Lidar pair with a planar target environment. The use of planes makes the algorithm robust to poor initial extrinisic calibration between the sensors (i.e., a poor guess of the 6 degree of freedom spatial transform between them).

The planar SLAM algorithm is based on the following paper:

Tae-kyeong Lee, Seungwook Lim, Seongsoo Lee, Shounan An, and Se-young Oh.
Indoor Mapping Using Planes Extracted from Noisy RGB-D Sensors. 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012.

I take no credit for design of the algorithm.

Using the code
--------------

The user must supply IMU and Lidar measurements in a format exemplified in the included sensor data.
In addition, with simulated data, the user may include a file containing the ground truth IMU
pose changes (ie. the delta_x, delta_y, delta_theta values) in metres/radians. This is different 
from inertial data which measures accelerations and velocities. Including this file allows for
evaluation of the SLAM routine against ground truth.

Any fields the user should change are marked with a "# USER: " comment.
