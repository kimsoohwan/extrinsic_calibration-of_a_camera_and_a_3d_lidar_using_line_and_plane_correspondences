from array import array
import numpy as np
import matplotlib.pyplot as plt


def show_point_cloud(point_cloud):
    
    point_could_temp = np.vstack((point_cloud, point_cloud[0,:]))
    
    plt.figure()
    plt.autoscale(False)
    ax = plt.axes(projection='3d')
    
    # plot calibration target
    ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target')
    
    # plot lidar
    ax.scatter3D(0, 0, 0, 'green', label='LiDAR')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    len_str = ""
    for point_index in range(len(point_could_temp)-1):
        len_str += "{}, ".format(np.math.sqrt(np.sum((point_could_temp[point_index]-point_could_temp[point_index+1]) **2)))
    plt.title(len_str)
    plt.legend()
    

def get_rotation_matrix(rotation_vector):

    rotation_vector *= -1 

    # degree to radian
    theta_x, theta_y, theta_z = rotation_vector * np.math.pi / 180.0

    rx = np.array([[1, 0                   , 0],
                   [0, np.math.cos(theta_x), -np.math.sin(theta_x)], 
                   [0, np.math.sin(theta_x), np.math.cos(theta_x)]])

    ry = np.array([[np.math.cos(theta_y) , 0, np.math.sin(theta_y)],
                   [0                    , 1 , 0],
                   [-np.math.sin(theta_y), 0, np.math.cos(theta_y)]])

    rz = np.array([[np.math.cos(theta_z), -np.math.sin(theta_z), 0],
                   [np.math.sin(theta_z),  np.math.cos(theta_z), 0],
                   [0                   , 0                    , 1]])

    # rotation matrix
    rotation_matrix = np.dot(rz, np.dot(ry, rx))

    return rotation_matrix


def generate_a_lidar_plane_in_3D(
        target_width=1000, 
        target_height=1000, 
        max_number_of_ray=5, 
        hor_distance=10, 
        lidar_noise_type_x_dir='uniform',
        lidar_noise_type_y_dir='uniform', lidar_uniform_error_y_dir=(-5, 5),
        lidar_noise_type_z_dir='uniform', lidar_uniform_error_z_dir=(-5, 5),
        translation_vector=np.array([0.0, 0.0, 0.0]),
        rotation_vector=np.array([0.0, 0.0, 0.0]),
        liadar_range=100000,
        horizontal_resolution=0.2,
        vertical_resolution=2,
        range_accuracy=(-30, 30),
        horizontal_field_of_view = 360,
        vertical_field_of_view = 30,
        display=False
    ):
    """
    target_width: Calibration width in 3D scence (mm)
    target_height: Calibration height in 3D scence (mm)
    max_number_of_ray: maximum number of lidar ray that hit the target
    hor_distance: horizontal distance between two consequitive points on a ray 
    lidar_noise_type_x_dir: noise type
    lidar_uniform_error_x_dir: error for uniform distribution
    lidar_noise_type_y_dir: noise type
    lidar_uniform_error_y_dir: error for uniform distribution
    lidar_noise_type_z_dir: noise type
    lidar_uniform_error_z_dir: error for uniform distribution
    translation_vector: translate target in 3D world from the origin
    rotation_vector: rotate calibration target along x, y, and z axis (clockwise rotation, degree)
    lidar_range: lidar maximum range
    horizontal_resolution: in degree degree, usually between 0.1 degree and 0.4 degree
    vertical_resolution: in degree degree, usually between 2 degree and 3 degree
    range_accuracy: accuracy in x direction in mm
    horizontal_field_of_view: in degree
    vertical_field_of_view: in degree

    coordinate systerm: Coordinate System: X Forward, Y Left, Z Up
    width is in Z direction and height is in Y direction
    """

    # corners of target before translating and rotating
    target_init_corners = np.array([[0, target_width/2.0, target_height/2.0], [0, -target_width/2.0, target_height/2.0], [0, -target_width/2.0, -target_height/2.0], [0, target_width/2.0, -target_height/2.0]])
    if display:
        show_point_cloud(point_cloud=target_init_corners)

    # get rotation marix
    rotation_matrix = get_rotation_matrix(rotation_vector=rotation_vector)

    # rotate target by rotation angels
    target_rotated_corners = np.dot(rotation_matrix, target_init_corners.T).T
    if display:
        show_point_cloud(point_cloud=target_rotated_corners)

    # transform the target by tarnsform vector
    target_rotated_and_translated_corners = target_rotated_corners + translation_vector
    if display:
        show_point_cloud(point_cloud=target_rotated_and_translated_corners)

    plt.show()

if __name__ == '__main__':
    generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([10.0, 0.0, 0.0]), 
                                    translation_vector=np.array([500.0, 200.0, 0.0]),
                                    display=True
                                )
