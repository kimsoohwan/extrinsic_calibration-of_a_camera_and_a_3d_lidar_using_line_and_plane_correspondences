from array import array
from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def show_point_cloud(point_cloud, normal_vector=None, intersection_points=None, title=None):
    
    point_could_temp = np.vstack((point_cloud, point_cloud[0,:]))
    
    plt.figure()
    plt.autoscale(False)
    ax = plt.axes(projection='3d')

    # plot calibration target
    ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target')
    
    # plot lidar
    ax.scatter3D(0, 0, 0, 'green', label='LiDAR')
    
    # plot normal vector
    if normal_vector is not None:
        middle = np.mean(point_cloud, axis=0)
        ax.scatter3D(middle[0], middle[1], middle[2], 'purple')
        ax.plot3D([middle[0], middle[0]+normal_vector[0]*100],
                  [middle[1], middle[1]+normal_vector[1]*100], [middle[2], middle[2]+normal_vector[2]*100],
                  'red', label='Normal Vector')

    if intersection_points is not None:
        ax.scatter3D(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], 'blue', label='intersecion points')


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    len_str = ""
    for point_index in range(len(point_could_temp)-1):
        len_str += "{:.2f} ".format(np.math.sqrt(np.sum((point_could_temp[point_index]-point_could_temp[point_index+1]) **2)))
    
    if normal_vector is None:
        plt.title(title)
    else:
        plt.title("{}\nNormal: {}".format(title, normal_vector))

    plt.legend()
    

def get_rotation_matrix(rotation_vector):
 
    # degree to radian
    theta_x, theta_y, theta_z =  -1 * rotation_vector * np.math.pi / 180.0

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


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )

def distance_two_points(point1, point2):
    return np.math.sqrt(np.sum((point1-point2)**2))

def generate_a_lidar_plane_in_3D(
        target_width=1000, 
        target_height=1000,  
        translation_vector=np.array([0.0, 0.0, 0.0]),
        rotation_vector=np.array([0.0, 0.0, 0.0]),
        liadar_range=100000,
        horizontal_resolution=0.2,
        vertical_resolution=2,
        range_accuracy=30,
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
    
    # normal vector of plane
    plane_normal = np.array([1, 0, 0])
    
    # display plane (calibration target)
    if display:
        show_point_cloud(point_cloud=target_init_corners, normal_vector=plane_normal, title='Target at origin')


    # get rotation marix
    rotation_matrix = get_rotation_matrix(rotation_vector=rotation_vector)
    
    # rotate target by rotation angels
    target_rotated_corners = np.dot(rotation_matrix, target_init_corners.T).T
    
    # rotate normal vector of plane (calibration target)
    plane_normal = np.dot(rotation_matrix, plane_normal.T).T
    plane_normal /= np.linalg.norm(plane_normal)
    
    if display:
        show_point_cloud(point_cloud=target_rotated_corners, normal_vector=plane_normal, title='Target after rotaion: {}'.format(rotation_vector))


    # transform the target by tarnsform vector
    target_rotated_and_translated_corners = target_rotated_corners + translation_vector
    if display:
        show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, title='Target after translation: {}'.format(translation_vector))

    # all points on the palne (calibration target)
    all_intersection = []
    all_noisy_intersection = []

    # calculate intersection of LiDAR rays to plane (calibation target)
    all_vertical_range = [-vertical_field_of_view/2 + vertical_resolution * v_r for v_r in range(int(np.math.ceil(vertical_field_of_view/vertical_resolution))+1)]
    all_horizontal_range = [-horizontal_field_of_view/2 + horizontal_resolution * v_r for v_r in range(int(np.math.ceil(horizontal_field_of_view/horizontal_resolution))+1)]
    
    for z_range in all_vertical_range:
        for y_range in all_horizontal_range:
            
            if y_range < 0:
                continue

            # start of ray: lidar position
            start_ray = np.array([0, 0, 0])

            # another point of ray
            direction_ray = np.array([1, 0, 0])
            ray_rotation_matrix = get_rotation_matrix(rotation_vector=np.array([0, z_range, y_range-90]))
            direction_ray = np.dot(ray_rotation_matrix, direction_ray.T).T            
            another_ray = start_ray + direction_ray

            intersection_pos = isect_line_plane_v3(p0=start_ray, 
                                                   p1=another_ray,
                                                   p_co=target_rotated_and_translated_corners[0, :],
                                                   p_no=plane_normal,
                                                   epsilon=1e-6)

            if intersection_pos is not None:
                # check intersection is on plane (calibration target)
                if np.dot(target_rotated_and_translated_corners[1, :]-target_rotated_and_translated_corners[0, :], intersection_pos-target_rotated_and_translated_corners[0, :]) >= 0:
                    if np.dot(target_rotated_and_translated_corners[2, :]-target_rotated_and_translated_corners[1, :], intersection_pos-target_rotated_and_translated_corners[1, :]) >= 0:
                        if np.dot(target_rotated_and_translated_corners[3, :]-target_rotated_and_translated_corners[2, :], intersection_pos-target_rotated_and_translated_corners[2, :]) >= 0:
                            if np.dot(target_rotated_and_translated_corners[0, :]-target_rotated_and_translated_corners[3, :], intersection_pos-target_rotated_and_translated_corners[3, :]) >= 0:
                                all_intersection.append(intersection_pos)

                                # add noise to the intersection
                                lidar_intersection_line = intersection_pos - start_ray
                                lidar_intersection_line /= np.linalg.norm(lidar_intersection_line)
                                intersection_pos_noisy = np.random.choice([-1, 1]) * lidar_intersection_line * np.random.uniform(low=-range_accuracy, high=range_accuracy) + intersection_pos

                                all_noisy_intersection.append(intersection_pos_noisy)

    # reshape intersection points
    if len(all_intersection) != 0 and display:
        all_intersection = np.array(all_intersection)
        all_intersection = np.reshape(all_intersection, newshape=(-1, 3))

        all_noisy_intersection = np.array(all_noisy_intersection)
        all_noisy_intersection = np.reshape(all_noisy_intersection, newshape=(-1, 3))

        #show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, intersection_points=all_intersection, title='Intersection of LiDAR and Target, No Noise')
        show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, intersection_points=all_noisy_intersection, title='Intersection of LiDAR and Target, With Noise')


    if display:
        plt.show()

if __name__ == '__main__':
    generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([45.0, 0.0, 0.0]), 
                                    translation_vector=np.array([5000.0, 0.0, 0.0]),
                                    display=True
                                )
