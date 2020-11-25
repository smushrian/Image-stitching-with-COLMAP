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
            ' --image_path ' + image_path)
    return
def fetch_cameras_from_database():
    try:
        sqliteConnection = sqlite3.connect('database.db')
        # cursor = sqliteConnection.cursor()
        # conn = sql.connect('weather.db')
        print("Connected to database")

        sqlite_select_query = """SELECT * from cameras"""

        cameras_table = pd.read_sql(sqlite_select_query, sqliteConnection)
        # cursor.execute(sqlite_select_query)
        # records = cursor.fetchall()
        # print("Total rows are:  ", len(records))
        # print("Printing each row")
        # print(records)

        # cursor.close()
        print(cameras_table.loc[1,:]

)
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
        return
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
    return cameras_table

def fetch_keypoints_from_database():
    try:
        sqliteConnection = sqlite3.connect('database.db')
        print("Connected to database")

        sqlite_select_query = """SELECT * from keypoints"""
        keypoints_table = pd.read_sql(sqlite_select_query, sqliteConnection)

        print(keypoints_table)
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
        return
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
    return keypoints_table

def fetch_matches_from_database():
    try:
        sqliteConnection = sqlite3.connect('database.db')
        print("Connected to database")

        sqlite_select_query = """SELECT * from matches"""
        matches_table = pd.read_sql(sqlite_select_query, sqliteConnection)

        print(matches_table)
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
        return
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
    return matches_table

def fetch_geometries_from_database():
    try:
        sqliteConnection = sqlite3.connect('database.db')
        print("Connected to database")

        sqlite_select_query = """SELECT * from two_view_geometries"""
        geometries_table = pd.read_sql(sqlite_select_query, sqliteConnection)

        # print(geometries_table)
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
        return
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
    return geometries_table
    # cameras
    # images
    # keypoints
    # descriptors
    # matches
    # two_view_geometries

def bin_to_float(b):
    """ Convert binary string to a tuple of floats. """
    # bf = int(b, 2).to_bytes()  # 8 bytes needed for IEEE 754 binary64.
    s = struct.Struct('f')
    # file_size = float.from_bytes(b.read(2), byteorder='big')
    return s.unpack_from(b)

def get_data_from_binary():
    dirname = os.path.dirname(__file__)
    os.chdir(dirname + '/COLMAP')
    database_path = dirname + '/COLMAP/database.db'
    camera_path = dirname + '/COLMAP/sparse/0/cameras.bin'
    points_path = dirname + '/COLMAP/sparse/0/points3D.bin'
    images_path = dirname + '/COLMAP/sparse/0/images.bin'
    cameras = read_write_model.read_cameras_binary(camera_path)
    points3D = read_write_model.read_points3d_binary(points_path)
    images = read_write_model.read_images_binary(images_path)
    print(cameras)
    print(points3D)
    print(images)
    return cameras, points3D, images
# database_path = create_database()

# feature_extraction(database_path)
# match_features(database_path)
# geometries_table = fetch_geometries_from_database()
# print(geometries_table.loc[2,:])
# test = bin_to_float(geometries_table['data'][2])
# print(test)
# automatic_reconstructor()
# cameras, points3D, images = get_data_from_binary()
# print(points3D[1].xyz)
# {1: Point3D(id=1, xyz=array([4.62656199, -0.45836651, 18.59784168]), rgb=array([12, 16, 19]),
                # error=array(2.15046715), image_ids=array([1, 2]), point2D_idxs=array([0, 1])),}