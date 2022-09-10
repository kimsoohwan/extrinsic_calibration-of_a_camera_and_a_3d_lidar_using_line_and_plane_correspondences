import cv2 as cv
import numpy as np

from camera_coordinate_find_plane_equation_calibration_target import \
    camera_coordinate_plane_equation_calibration_target
from camera_image_find_edges_of_calibration_target import \
    line_equation_four_edges_calibration_target_in_camera_image
from lidar_find_line_equation import calculate_line_equition
from read_calibration_file import read_yaml_file


def find_intersection_of_ray_plane(image_point, plane, camera_matrix):
    """
    This function find the intersection of ray that pass 'image_point' on the image and a plane
    """
    
    # convert to homogenous format
    p = np.array([[image_point[0]], [image_point[1]], [1]])

    # ray
    p_norm = np.dot(np.linalg.inv(camera_matrix), p)
    p_norm /= p_norm[2, 0]

    t = -(plane[3]) / (plane[0] * p_norm[0, 0] + plane[1] * p_norm[1, 0] + plane[2] * 1)

    # intersection point in camera coordinate
    point_intersection = t * p_norm

    return point_intersection

def find_edge_equation_in_camera_coordinate(line_equation_image, plane_camera_coordinate, camera_matrix):
    """
    line_equation_image: line equation in image
    plane_camera_coordinate: calibration target plane equation in camera coordinate
    camera_matrix: 3 * 3 camera matrix
    """
    # two points of line in the image
    point_1 = line_equation_image[0] + 1 * line_equation_image[1]
    point_2 = line_equation_image[0] + 2 * line_equation_image[1]
    
    # finds intersection of ray that passes those two lines and plane equation of calibration target
    # inside camera coordinate
    point_intersection_1 = find_intersection_of_ray_plane(image_point=point_1, plane=plane_camera_coordinate, camera_matrix=camera_matrix)
    point_intersection_2 = find_intersection_of_ray_plane(image_point=point_2, plane=plane_camera_coordinate, camera_matrix=camera_matrix)

    # line equation in camera coordinate
    two_points = np.vstack((point_intersection_1.T, point_intersection_2.T))

    # line equation in formap (x0: a point on line, t: line direction)
    line_equation_camera_coordinate = calculate_line_equition(two_points=two_points)

    return line_equation_camera_coordinate

if __name__ == '__main__':

    for img_path in ['/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png']:
        
        #####################################
        #   Read Image
        #####################################
        # read image
        img_bgr = cv.imread(img_path)

        # convert BGR to RGB
        rgb_image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        ######################################
        # read camera calibration information
        ######################################
        path = '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/left_camera_calibration_parameters.yaml'
        calibration_data = read_yaml_file(path=path)
        
        ######################################
        # find calibration target plane 
        # equation in camera coordinate
        ######################################
        plane_equation = camera_coordinate_plane_equation_calibration_target(
                            rgb_img=rgb_image,
                            num_row=6,
                            num_col=8,
                            square=152,
                            camera_matrix=calibration_data['camera_matrix'],
                            distortion_coefficients=calibration_data['distortion_coefficients'],
                            display=True
                        )
        
        #####################################
        #   Finds edges
        #   equations in camera image 
        #####################################
        lines_equations = line_equation_four_edges_calibration_target_in_camera_image(rgb_image=rgb_image, display=True)

        print('All line equations for four edges of calibration target in image')
        for line_name in lines_equations:
            
            print('Line name: {}'.format(line_name))
            print('Line equation (point, direction): {}'.format(lines_equations[line_name]))
        print('=' * 100)
        ######################################
        #   calculate line equation in  
        #   camera coordinate
        ######################################
        lines_equation_camera_coordinate = {}
        for line_name in lines_equations:
            line_equation_camera_coordinate = find_edge_equation_in_camera_coordinate(
                line_equation_image=lines_equations[line_name],
                plane_camera_coordinate=plane_equation,
                camera_matrix=calibration_data['camera_matrix']
                )

            lines_equation_camera_coordinate[line_name] = np.copy(line_equation_camera_coordinate)
            print(">    Line equation,{}, in camera coordinate:".format(line_name))
            print(line_equation_camera_coordinate)
