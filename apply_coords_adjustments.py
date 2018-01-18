"""
This was adapted from code given to me by Ganesh (@oilersnerdalert)
"""

import numpy as np
import pandas as pd
import rink_coords_adjustment as rda

pd.options.mode.chained_assignment = None  # default='warn' -> Stops it from giving me some error


def fix_df(df):
    """
    Fixes df so it can be processed for rink adjustments
    
    :param df: Full DataFrame
    
    :return: Fixed DataFrame
    """
    # Only take events we need
    pbp_df = df.loc[df.Event.isin(["SHOT", "GOAL", "MISS"])]

    # add a 'Direction' column to indicate the primary direction for shots. The heuristic to determine
    # direction is the sign of the median of the X coordinate of shots in each period. This then lets us filter
    # out shots that originate from back in the defensive zone when the signs don't match
    gp_groups = pbp_df.groupby(by=['Date', 'Game_Id', 'Period'])['xC', 'yC']
    meanies = gp_groups.transform(np.median)  # will give us game/period median for X and Y for every data point
    pbp_df['Direction'] = np.sign(meanies['xC'])

    # should actually write this to a CSV as up to here is the performance intensive part
    pbp_df['xC'], pbp_df['yC'] = zip(*pbp_df.apply(lambda x: (x.xC, x.yC) if x.xC > 0 else (-x.xC, -x.yC), axis=1))

    return pbp_df


def create_cdfs(shots_df, rink_adjuster):
    """
    Goes through and creates cdf for each team
    
    :param shots_df: df with only - Goals, SOG, and Misses
    :param rink_adjuster: RinkAdjust object
    :return: None
    """
    # Now rip through each team and create a CDF for that team and for the other 29 teams in the league
    # For each home rink
    for team in sorted(shots_df.Home_Team.unique()):
        # Split shots into team arena and all other rinks
        shot_data = shots_df
        rink_shots = shot_data[shot_data.Home_Team == team]
        rest_of_league = shot_data[shot_data.Home_Team != team]

        # Create teamxcdf and otherxcdf for rink adjustment
        rink_adjuster.addTeam(team, rink_shots, rest_of_league)


def adjust_play(play, rink_adjuster):
    """
    Apply rink adjustments to play
    
    :param play: given play in game
    :param rink_adjuster: RinkAdjust object
    
    :return: newx, newy
    """
    if play['Event'] in ["SHOT", "GOAL", "MISS"]:
        # abs() for xC because all coordinates are made positive for cdf (to make it normal)
        newx, newy = rink_adjuster.rink_bias_adjust(abs(play['xC']), play['yC'], play['Home_Team'])

        # if xC is really negative (because cdf only deals in positives) change it back
        newx = -newx if play['xC'] < 0 else newx
    else:
        newx, newy = play['xC'], play['yC']

    return newx, newy


def adjust_df(pbp_df, rink_adjuster):
    """
    Apply rink adjustments to PBP. Iterates through every play and adjusts from there
    
    :param pbp_df: PBP DataFrame
    :param rink_adjuster: RinkAdjust object
    
    :return: Adjusted DataFrame
    """
    df_dict = pbp_df.to_dict('records')

    pbp_df['xC_adj'], pbp_df['yC_adj'] = map(list, zip(*[adjust_play(row, rink_adjuster) for row in df_dict]))

    return pbp_df


def adjust(df):
    """
    Take a DataFrame and:
    1. Creates CDF's for each team
    2. Adjusts given games
    
    **Note: I advise not feeding this a DataFrame with less than one year's worth of data
    
    :param df: DataFrame of games
    
    :return: DataFrame with distance Rink Adjusted
    """
    rink_adjuster = rda.RinkAdjust()
    create_cdfs(fix_df(df), rink_adjuster)
    pbp_df = adjust_df(df, rink_adjuster)

    return pbp_df







