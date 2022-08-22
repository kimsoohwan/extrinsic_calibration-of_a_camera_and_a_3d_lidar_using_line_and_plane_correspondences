from unittest.mock import patch
import yaml
import numpy as np

def read_yaml_file(path):
    with open(path, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError('{}'.format(exc))

    calibration_data = {
        'img_col_px': yaml_data['image_width'], 
        'img_row_px': yaml_data['image_height'],
        'camera_matrix': np.reshape(yaml_data['camera_matrix']['data'], newshape=(yaml_data['camera_matrix']['rows'], yaml_data['camera_matrix']['cols'])),
        'distortion_model': yaml_data['distortion_model'],
        'distortion_coefficients': np.reshape(yaml_data['distortion_coefficients']['data'], newshape=(yaml_data['distortion_coefficients']['rows'], yaml_data['distortion_coefficients']['cols'])),
        'rectification_matrix': np.reshape(yaml_data['rectification_matrix']['data'], newshape=(yaml_data['rectification_matrix']['rows'], yaml_data['rectification_matrix']['cols'])),
        'projection_matrix': np.reshape(yaml_data['projection_matrix']['data'], newshape=(yaml_data['projection_matrix']['rows'], yaml_data['projection_matrix']['cols'])),
    }

    return calibration_data

if __name__ == '__main__':
    path = '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/left_camera_calibration_parameters.yaml'

    calibration_data = read_yaml_file(path=path)

    print(calibration_data)