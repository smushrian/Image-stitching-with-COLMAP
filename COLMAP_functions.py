import os
import sqlite3
import pandas as pd
import struct
from help_scripts.python_scripts import read_write_model

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
    os.chdir(dirname + '/COLMAP')
    workspace_path = dirname + '/COLMAP'
    image_path = workspace_path + '/images'
    os.system('colmap automatic_reconstructor \
            --workspace_path ' + workspace_path +
            ' --image_path ' + image_path +
              ' --dense 1')
    return


def get_data_from_binary():
    dirname = os.path.dirname(__file__)
    os.chdir(dirname + '/COLMAP')
    database_path = dirname + '/COLMAP/database.db'
    camera_path = dirname + '/COLMAP/dense/0/sparse/cameras.bin'
    points_path = dirname + '/COLMAP/dense/0/sparse/points3D.bin'
    images_path = dirname + '/COLMAP/dense/0/sparse/images.bin'
    cameras = read_write_model.read_cameras_binary(camera_path)
    points3D = read_write_model.read_points3d_binary(points_path)
    images = read_write_model.read_images_binary(images_path)
    return cameras, points3D, images
