import numpy as np
from read_calibration_file import read_yaml_file
from find_calibration_target_in_camera_image import find_corners_on_calibration_target
import cv2 as cv


def rotation_and_translation_of_target_in_camera_image(object_points, image_points, camera_matrix, distortion_coefficients):
    """
    object_points: Array of object points in the object coordinate space
    image_points: Array of corresponding image points
    camera_matrix: Input camera intrinsic matrix (3 in 3)
    distortion_coefficients: Input vector of distortion coefficients



    Output (rvec and tvec), rotation vector that, together with translation vector, brings points from the model coordinate system
    to the camera coordinate system. 
    rvec is in radian
    tvec is in mm
    """
    retval, rvec, tvec = cv.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=camera_matrix, distCoeffs=distortion_coefficients)

    if retval == True:
        return {'rotation_vector': rvec, 'translation_vector': tvec}
    else:
        return None

if __name__ == '__main__':
    
    ################################################################
    # calibration information related to camera
    ################################################################
    path = '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/left_camera_calibration_parameters.yaml'
    calibration_data = read_yaml_file(path=path)
    print(calibration_data)

    ################################################################
    # read image and find corners in calibration target
    ################################################################
    # read image from file
    img = cv.imread('/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png')

    # find corners on calibration target
    points_3d_image_image_subpix = find_corners_on_calibration_target(img=img, num_row=6, num_col=8, square=152, display=True)
    print(points_3d_image_image_subpix)

    ################################################################
    # find rotation and translation from object to camera coordinate
    # system
    ################################################################
    rotation_translation = rotation_and_translation_of_target_in_camera_image(
                                object_points=points_3d_image_image_subpix['points_in_3D'],
                                image_points=points_3d_image_image_subpix['points_in_image_sub_pixel'],
                                camera_matrix=calibration_data['camera_matrix'],
                                distortion_coefficients=calibration_data['distortion_coefficients']
                            )

    print(rotation_translation)