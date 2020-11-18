import os
import sqlite3
import pandas as pd
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
    # cameras
    # images
    # keypoints
    # descriptors
    # matches
    # two_view_geometries
database_path = create_database()
feature_extraction(database_path)
match_features(database_path)
fetch_cameras_from_database()