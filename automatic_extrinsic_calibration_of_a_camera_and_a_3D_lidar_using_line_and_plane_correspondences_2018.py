import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

from find_plane_equation_edges_equation_in_lidar_camera_coordinate import \
    calculate_plane_equation_edges_equation_in_lidar_camera_coordinate
from read_calibration_file import read_yaml_file

def is_rotation_matrix(R) :
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def init_estimate_R_one_pose(lidar_plane_equation, lidar_edges_equation, camera_coordinate_plane_equation, camera_coordinate_edges_equation):
    n_l = lidar_plane_equation.tolist()[0:3]
    l1_l =  lidar_edges_equation['line_equation_left_lower'][1].tolist()
    l2_l =  lidar_edges_equation['line_equation_left_upper'][1].tolist()
    l3_l =  lidar_edges_equation['line_equation_right_upper'][1].tolist()
    l4_l =  lidar_edges_equation['line_equation_right_lower'][1].tolist()
    
    m_l = np.stack((n_l, l1_l, l2_l, l3_l, l4_l)).T
    
    n_c = camera_coordinate_plane_equation.tolist()[0:3]
    l1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][1].tolist()
    l2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][1].tolist()
    l3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][1].tolist()
    l4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][1].tolist()

    m_c = np.stack((n_c, l1_c, l2_c, l3_c, l4_c)).T

    m = np.dot(m_l, m_c.T)

    u, s, v_t = np.linalg.svd(a=m)

    r_estimate_matrix = np.dot(v_t.T, u.T)
    r_estimate_vector_radian = rotation_matrix_to_euler_angles(r_estimate_matrix)
    r_estimate_vector_degree = r_estimate_vector_radian * (180/np.math.pi)

    return {'rotation_matrix': r_estimate_matrix, 'rotation_vector_radian':r_estimate_vector_radian, 'rotation_vector_degree':r_estimate_vector_degree}


    
def automatic_extrinsic_calibration_of_a_camera_and_a_3D_lidar_using_line_and_plane_correspondences(
    calibration_data,
    camera_coordinate_plane_equation,
    camera_coordinate_edges_equation,
    lidar_plane_equation,
    lidar_plane_centroid,
    lidar_edges_equation,
    lidar_edges_centroid
):
    
    # initial estimate for R
    estimated_rotation_results = init_estimate_R_one_pose(
                                        lidar_plane_equation=lidar_plane_equation,
                                        lidar_edges_equation=lidar_edges_equation,
                                        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
                                        camera_coordinate_edges_equation=camera_coordinate_edges_equation
                                        )
    estimated_rotation_matrix = estimated_rotation_results['rotation_matrix']

    

if __name__ == '__main__':
    
    ################################################################
    # Read image
    ################################################################
    # read image
    img_bgr = cv.imread('/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png')

    # convert BGR to RGB
    rgb_image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    ################################################################
    # Read Point Cloud
    ################################################################
    point_cloud = np.load('example_real_img_lidar_points/selected_points_in_lidar-1.npy')
    # convert to mm
    point_cloud *= 1000

    ################################################################
    # calibration information related to camera
    ################################################################
    path = '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/left_camera_calibration_parameters.yaml'
    calibration_data = read_yaml_file(path=path)
    print("Calibration Parameters:\n", calibration_data)

    ################################################################
    # Calculate plane equation and edges equations for calibration
    # target inside lidar and camera coordinate system
    ################################################################
    plane_edges_equations_in_lidar_camera_coordinate = calculate_plane_equation_edges_equation_in_lidar_camera_coordinate(
                                                            point_cloud=point_cloud,
                                                            maximim_distance_two_consecutive_points_in_ray=100,
                                                            calibration_data=calibration_data,
                                                            rgb_image=rgb_image,
                                                            num_row=6,
                                                            num_col=8,
                                                            square=152,
                                                            display=False
                                                        )

    for key in plane_edges_equations_in_lidar_camera_coordinate:
        if key != 'image_process':
            print('>    {}'.format(key))
            print(plane_edges_equations_in_lidar_camera_coordinate[key])

    #for img in plane_edges_equations_in_lidar_camera_coordinate['image_process']:
    #    plt.figure()
    #    plt.imshow(img)
    #plt.show()

    ###################################################################
    #       Calculate R and t
    ###################################################################
    automatic_extrinsic_calibration_of_a_camera_and_a_3D_lidar_using_line_and_plane_correspondences(
        calibration_data=calibration_data,
        camera_coordinate_plane_equation=plane_edges_equations_in_lidar_camera_coordinate['camera_coordinate_plane_equation'],
        camera_coordinate_edges_equation=plane_edges_equations_in_lidar_camera_coordinate['camera_coordinate_edges_equation'],
        lidar_plane_equation=plane_edges_equations_in_lidar_camera_coordinate['lidar_plane_equation'],
        lidar_plane_centroid=plane_edges_equations_in_lidar_camera_coordinate['lidar_plane_centroid'],
        lidar_edges_equation=plane_edges_equations_in_lidar_camera_coordinate['lidar_edges_equation'],
        lidar_edges_centroid=plane_edges_equations_in_lidar_camera_coordinate['lidar_edges_centroid']
    )