import os
import sqlite3
import pandas as pd
import struct
import numpy as np
from help_scripts.python_scripts.scripts_from_colmap import read_write_model

def create_database():
    #GET THE PATH TO THE MAIN SCRIPT
    dirname = os.path.dirname(__file__)
    os.chdir(dirname + '/COLMAP')
    database_path = dirname + '/COLMAP/database.db'

    # CREATE DATABASE
    os.system('colmap database_creator \
              --database_path ' + database_path)
    return database_path

def feature_extraction(database_path):
    #PERFORM FEATURE EXTRACTION
    dirname = os.path.dirname(__file__)
    images_path = dirname + '/COLMAP/images'
    os.system('colmap feature_extractor \
        --database_path ' + database_path + ' \
        --image_path ' + images_path)
    return
def match_features(database_path):
    #PERFORM MATCHING
    os.system('colmap exhaustive_matcher \
        --database_path ' + database_path)
    return

def automatic_reconstructor():
    dirname = os.path.dirname(__file__)
    os.chdir(dirname + '/../../COLMAP_w_CUDA')
    workspace_path = os.getcwd()
    image_path = workspace_path + '/images'
    os.system('colmap automatic_reconstructor \
              --camera_model SIMPLE_RADIAL ' +
            '--workspace_path ' + workspace_path +
            ' --image_path ' + image_path +
              ' --dense 1 ' +
              '--use_gpu 0 ')
    return


def get_data_from_binary():
    # os.chdir('..')
    # os.chdir('..')
    # dirname = os.path.dirname(__file__)
    dirname = os.getcwd()
    # print(dirname)
    database_path = dirname + '/database.db'
    # camera_path = dirname + '/dense/0/den/cameras.bin'
    # points_path = dirname + '/dense/0/den/points3D.bin'
    # images_path = dirname + '/dense/0/den/images.bin'
    camera_path = dirname + '/dense/cameras.bin'
    points_path = dirname + '/dense/points3D.bin'
    images_path = dirname + '/dense/images.bin'
    # camera_path = dirname + '/sparse/0/cameras.bin'
    # points_path = dirname + '/sparse/0/points3D.bin'
    # images_path = dirname + '/sparse/0/images.bin'
    cameras = read_write_model.read_cameras_binary(camera_path)
    points3D = read_write_model.read_points3d_binary(points_path)
    images = read_write_model.read_images_binary(images_path)
    return cameras, points3D, images

def build_intrinsic_matrix(camera):
    if camera.model == 'SIMPLE_RADIAL':
        params = camera.params
        # K = [f, 0, cx;
        #      0, f, cy;
        #      0, 0, 1];
        K = np.asarray([[params[0], 0, params[1]],[0, params[0], params[2]],[0, 0, 1]])
        dist_params = [params[3],0]
    elif camera.model == 'PINHOLE':
        params = camera.params
        # K = [f, 0, cx;
        #      0, f, cy;
        #      0, 0, 1];
        K = np.asarray([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        dist_params = [0, 0]
    else:
        K = 1
        dist_params = 1
        print('Camera model must be either SIMPLE_RADIAL or PINHOLE')
    return K, dist_params


def stereo_fusion():
    project_path = os.getcwd()
    workspace_path = project_path + '/dense/0'
    # print(workspace_path)
    # print('Här')
    output_path = workspace_path + '/..'
    os.system('colmap stereo_fusion \
                 --output_type bin ' +
              '--output_path ' + output_path +
              # ' --project_path ' + project_path +
              ' --workspace_path ' + workspace_path) # +
              # '--workspace_format COLMAP ' +
              # '--input_type geometric ')

def image_undistorter():
    project_path = os.getcwd()
    workspace_path = project_path + '/dense/0'
    input_path = project_path + '/sparse/0'
    image_path = project_path + '/images'
    # print(workspace_path)
    # print('Här')
    output_path = workspace_path #+ '/..'
    os.system('colmap image_undistorter \
              --output_path ' + output_path +
              # ' --project_path ' + project_path +
              ' --image_path ' + image_path +
              ' --output_type COLMAP ' +
              '--input_path ' + input_path +
              ' --max_image_size 2000')
    # '--workspace_format COLMAP ' +
    # '--input_type geometric ')

