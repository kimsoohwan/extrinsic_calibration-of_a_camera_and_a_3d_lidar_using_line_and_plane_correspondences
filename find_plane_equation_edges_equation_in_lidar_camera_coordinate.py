import cv2 as cv
import numpy as np
from read_calibration_file import read_yaml_file
from camera_coordinate_find_plane_equation_calibration_target import camera_coordinate_plane_equation_calibration_target
from camera_image_find_edges_of_calibration_target import line_equation_four_edges_calibration_target_in_camera_image
from camera_coordinae_find_edges_equation_calibration_target import find_edge_equation_in_camera_coordinate
import traceback


def calculate_plane_equation_edges_equation_in_lidar_camera_coordinate(calibration_data, rgb_image, num_row, num_col, square, display=False):
    
    # plane equation of calibration target in camera coordinate system
    try:
        image_coordinate_plane_equation = camera_coordinate_plane_equation_calibration_target(
                                rgb_img=rgb_image,
                                num_row=num_row,
                                num_col=num_col,
                                square=square,
                                camera_matrix=calibration_data['camera_matrix'],
                                distortion_coefficients=calibration_data['distortion_coefficients'],
                                display=display
                            )
        print('Plane Equation in Camera Coordinate:')
        print(image_coordinate_plane_equation)
    except Exception:
        traceback.print_exc()

    
    #  Finds edges equations in camera image 
    lines_equations, image_process_1 = line_equation_four_edges_calibration_target_in_camera_image(
                                            rgb_image=rgb_image, display=False)

    print('All line equations for four edges of calibration target in image')
    for line_name in lines_equations:
            
        print('Line name: {}'.format(line_name))
        print('Line equation (point, direction): {}'.format(lines_equations[line_name]))
    print('=' * 100)

    # calculate line equation in  camera coordinate
    lines_equation_camera_coordinate = {}
    for line_name in lines_equations:
        line_equation_camera_coordinate = find_edge_equation_in_camera_coordinate(
                line_equation_image=lines_equations[line_name],
                plane_camera_coordinate=image_coordinate_plane_equation,
                camera_matrix=calibration_data['camera_matrix']
                )

        lines_equation_camera_coordinate[line_name] = np.copy(line_equation_camera_coordinate)
        print(">    Line equation,{}, in camera coordinate:".format(line_name))
        print(line_equation_camera_coordinate)


if __name__ == '__main__':
    
    ################################################################
    # Read image
    ################################################################
    # read image
    img_bgr = cv.imread('/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png')

    # convert BGR to RGB
    rgb_image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

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
    calculate_plane_equation_edges_equation_in_lidar_camera_coordinate(
            calibration_data,
            rgb_image=rgb_image,
            num_row=6,
            num_col=8,
            square=152,
            display=True
        )