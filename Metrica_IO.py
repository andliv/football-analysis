#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import pandas as pd
import csv as csv
import numpy as np
import requests
from io import StringIO


def read_match_data(DATADIR,gameid):
    '''
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    '''
    tracking_home = tracking_data(DATADIR,gameid,'Home')
    tracking_away = tracking_data(DATADIR,gameid,'Away')
    events = read_event_data(DATADIR,gameid)
    return tracking_home,tracking_away,events

def read_event_data(file_id):
    """
    Reads Metrica event data from a Google Drive file ID.

    Arguments:
    - file_id: Google Drive file ID for the CSV file.

    Returns:
    - Pandas DataFrame with event data.
    """

    # Construct Google Drive direct download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from Google Drive.")
    
    # Read the CSV file into a Pandas DataFrame
    # Open the CSV file content directly from the response
    csv_data = StringIO(response.text)

    # Read the event data into pandas
    events = pd.read_csv(csv_data)

    return events

def tracking_data(file_id, teamname):
    """
    Reads Metrica tracking data from a Google Drive file ID.

    Arguments:
    - file_id: Google Drive file ID for the CSV file.
    - teamname: 'Home' or 'Away' (used for column formatting).

    Returns:
    - Pandas DataFrame with tracking data.
    """

    # Construct Google Drive direct download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from Google Drive.")
    
    # Read the CSV file into a Pandas DataFrame
    # Open the CSV file content directly from the response
    from io import StringIO
    csv_data = StringIO(response.text)

    # Read the CSV with pandas
    reader = csv.reader(csv_data)
    
    # First: deal with file headers to get the player names correct
    teamnamefull = next(reader)[3].lower()  # team name from the header row
    print("Reading team: %s" % teamnamefull)

    # Construct column names
    jerseys = [x for x in next(reader) if x != '']  # Extract player jersey numbers
    columns = next(reader)  # Read the columns
    for i, j in enumerate(jerseys):  # Create x & y position column headers for each player
        columns[i*2+3] = f"{teamname}_{j}_x"
        columns[i*2+4] = f"{teamname}_{j}_y"
    columns[-2] = "ball_x"  # Column headers for the x & y positions of the ball
    columns[-1] = "ball_y"

    # Second: read in tracking data into pandas DataFrame
    tracking = pd.read_csv(csv_data, names=columns, index_col='Frame', skiprows=3)

    return tracking

def merge_tracking_data(home,away):
    '''
    merge home & away tracking data files into single data frame
    '''
    return home.drop(columns=['ball_x', 'ball_y']).merge( away, left_index=True, right_index=True )
    
def to_metric_coordinates(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    ''' 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    '''
    return data

def to_single_playing_direction(home,away,events):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home,away,events]:
        second_half_idx = team.Period.idxmax(2)
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home,away,events

def find_playing_direction(team,teamname):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    '''    
    GK_column_x = teamname+"_"+find_goalkeeper(team)+"_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])
    
def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
    GK_col = team.iloc[0][x_columns].abs().idxmax()
    return GK_col.split('_')[1]
    