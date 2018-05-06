import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from math import sqrt
import apply_coords_adjustments as rink_adjust
from db_info import *


def check_columns(data):
    """
    Checks columns for final DataFrame.. If column isn't there it adds it in.
    Ex: adds '6x4' dummy variable if in sample that strength wasn't played

    Note: Doesn't apply to building the model but when using it on subset of data

    :param data: Data used in model

    :return: fixed up df
    """
    cols = ['Type_BACKHAND', 'Type_DEFLECTED', 'Type_SLAP SHOT', 'Type_SNAP SHOT', 'Type_TIP-IN', 'Type_WRAP-AROUND',
            'Type_WRIST SHOT', 'Strength_3x3', 'Strength_3x4', 'Strength_3x5', 'Strength_3x6', 'Strength_4x3',
            'Strength_4x4', 'Strength_4x5', 'Strength_4x6', 'Strength_5x3', 'Strength_5x4', 'Strength_5x5',
            'Strength_5x6', 'Strength_6x3', 'Strength_6x4', 'Strength_6x5', 'score_cat_-3', 'score_cat_-2', 'score_cat_-1',
            'score_cat_0', 'score_cat_1', 'score_cat_2', 'score_cat_3',
            ]

    for col in cols:
        if col not in data.columns:
            # If not there just fill it with zeros
            data[col] = 0

    return data


def fix_prev_event(row):
    """
    More mistakes by the NHL. A Faceoff must come between each of these event so just input a faceoff from center ice
    That'll only be be "wrong" for ESITR (sometimes). 
    
    :param row: play
    
    :return: fixed row
    """
    if row['prev_event'] in ["PSTR", "GOAL", "EISTR"]:
        row['prev_event'] = "FAC"
        row['prev_xC_adj'] = 0
        row['prev_yC_adj'] = 0

    return row


def fix_score_cat(row):
    """
    When a goal was scored the I recorded the score with the goal which is obviously incorrect
    
    :param row: play
    
    :return: fixed score
    """
    if row['Event'] == "GOAL":
        if row['Ev_Team'] == row['Home_Team']:
            row["Home_Score"] -= 1
        else:
            row["Away_Score"] -= 1

    return row["Home_Score"], row["Away_Score"]


def if_empty_net(row):
    """
    Check if it's an empty net. 
    
    :param row: play in pbp
    
    :return: if non-event's team net is empty
    """
    if row['Ev_Team'] == row['Home_Team']:
        return 1 if row['Away_Goalie'] == "Empty" else 0
    else:
        return 1 if row['Home_Goalie'] == "Empty" else 0


def get_angle_change(row):
    """
    Calculate the angle change on a rebound shot
    
    :param row: play
    
    :return: angle change if rebound otherwise zero
    """
    if row['is_rebound'] == 0:
        return 0

    current_angle = 90-abs(row['Angle'])
    prev_angle = 90-abs(row['prev_angle'])

    if np.sign(row['Angle']) == np.sign(row['prev_angle']):
        return abs(current_angle-prev_angle)
    else:
        return abs(current_angle+prev_angle)


def get_prev_angle(row):
    """
    Get the angle for the previous shot (only for rebound)
    
    :param row: play 
    
    :return: angle for shot before rebounds
    """
    if row['prev_event'] == "SHOT" and row['Event'] == "SHOT":
        return 90 if row['prev_yC_adj'] == 0 else np.arctan((89.45 - abs(row['prev_xC_adj'])) / row['prev_yC_adj']) * (180/np.pi)
    else:
        return 0


def get_distance(row):
    try:
        return sqrt((row['xC_adj'] - row['prev_xC_adj']) ** 2 + (row['yC_adj'] - row['prev_yC_adj']) ** 2)
    except TypeError:
        # Debugging
        print(row["prev_event"])
        exit()


def fix_strength(row):
    """
    Fix the strength - flip if away team is event team
    
    :param row: play
    
    :return: "fixed" strength
    """
    if row['Ev_Team'] == row['Home_Team']:
        return row['Strength']
    else:
        return "x".join([row['Strength'][2], row['Strength'][0]])


def get_shift_player(play, shifts_df, player):
    """
    NOTE: Not used in model
    Get start of current shift for player
    
    :param play: play in game
    :param shifts_df: DataFrame of shit info for game
    :param player: player we are getting info for - Either shooter or home/ away players (goalies are already screened)
    
    :return: current time on ice for player
    """
    shift_starts = shifts_df[(shifts_df['Player_Id'] == play[player]) & (shifts_df['Period'] == play['Period']) &
                             (shifts_df['Start'] <= play['Seconds_Elapsed'])]

    if shift_starts.empty:
        return 0
    else:
        return play['Seconds_Elapsed'] - shift_starts.iloc[shift_starts.shape[0] - 1]['Start']


def get_shifts_play(play, shifts_df):
    """
    NOTE: Not used in model
    Get shift times for given play
    
    :param play: play in given game
    :param shifts_df: DataFrame of shit info for game
    
    :return: shift time for shooter, mean shift time for home team, mean shift time for away team
    """
    # Get it for shooter
    shooter_shift = get_shift_player(play, shifts_df, 'p1_ID')

    home = []
    away = []
    # Get it for both teams
    for i in range(1, 7):
        if play['homePlayer{}_id'.format(i)] != play['Home_Goalie_Id'] and not np.isnan(play['homePlayer{}_id'.format(i)]):
            home.append(get_shift_player(play, shifts_df, 'homePlayer{}_id'.format(i)))

        if play['awayPlayer{}_id'.format(i)] != play['Away_Goalie_Id'] and not np.isnan(play['awayPlayer{}_id'.format(i)]):
            away.append(get_shift_player(play, shifts_df, 'awayPlayer{}_id'.format(i)))

    return shooter_shift, sum(home)/float(len(home)), sum(away)/float(len(away))


def get_shifts_time(pbp_df):
    """
    NOTE: Not used in model
    Get current time on ice for each play
    
    :param pbp_df: DataFrame of pbp
    
    :return: Dataframe of pbp with info
    """
    year = pbp_df.iloc[0]['Date'][:4]
    shifts_df = pd.read_csv("../data/nhl_shifts{}.csv".format(year + str(int(year)+1)))
    games = list(set(pbp_df['Game_Id'].tolist()))

    game_dfs = []
    for game in games:
        print(game)
        # Subset game
        game_df = pbp_df[pbp_df['Game_Id'] == game]
        game_shifts = shifts_df[shifts_df['Game_Id'] == game]

        # Get shift times
        plays = game_df.to_dict('records')
        game_df['Shooter_Shift'], game_df['Home_Shift'], game_df['Away_Shift'] = map(list, zip(*[get_shifts_play(play, game_shifts) for play in plays]))

        game_df['Team_Shift'] = np.where(game_df['Ev_Team'] == game_df['Home_Team'], game_df['Home_Shift'], game_df['Away_Shift'])
        game_df['Opp_Shift'] = np.where(game_df['Ev_Team'] == game_df['Home_Team'], game_df['Away_Shift'], game_df['Home_Shift'])

        game_dfs.append(game_df)

    df = pd.concat(game_dfs)
    df = df.reset_index(drop=True)

    return df


def get_handedness(player_id, players):
    """
    Return handedness of player - alerts me if can't find player
    
    :param player_id: player's id number
    :param players: dict of players
    
    :return: handedness
    """
    try:
        return players[str(player_id)]["hand"]
    except (KeyError, ValueError):
        print("Player id " + str(player_id) + " not found")
        return ''


def get_position(player_id, players):
    """
    Return position of player - alerts me if can't find player

    :param player_id: player's id number
    :param players: dict of players

    :return: position
    """
    try:
        return players[str(player_id)]["position"]
    except KeyError:
        print("Player id " + str(player_id) + " not found")
        return ''


def get_shooter_info(pbp):
    """
    Get info for shooter
    1. Handedness of player
    2. Position of player
    
    :param pbp: Play by Play
    
    :return: list containing list of handedness and positions
    """

    engine = create_engine('postgresql://{user}:{passw}@localhost:5432/nhl_data'.format(user=USERNAME, passw=PASSWORD))
    players_df = pd.read_sql_table('nhl_players', engine)

    # Get list of all players and positions
    players_series = players_df.apply(lambda row: [row['id'], row['shoots_catches'], row['position']], axis=1)
    players_set = set(tuple(x) for x in players_series.tolist())
    players_list = [list(x) for x in players_set]

    # Dict of players -> ID is key
    players = dict()
    for p in players_list:
        players[str(p[0])] = {"hand": p[1], "position": p[2]}

    pbp_dict = pbp.to_dict("records")
    shooters_hand = [get_handedness(int(play['p1_ID']), players) for play in pbp_dict]
    shooters_pos = [get_position(int(play['p1_ID']), players) for play in pbp_dict]

    return [shooters_hand, shooters_pos]


def if_off_wing(row):
    """
    Check if the shot was taken by a player on their off wing
    1. Home Team: negative y axis is left side
    2. Away Team: positive y axis is left side
    
    :param row: given event
    
    :return: boolean - yes for off wing no for same
    """

    # L=Left, R=Right
    if row['Home_Team'] == row['Ev_Team']:
        direction = "L" if row['Period'] % 2 != 0 else "R"
    else:
        direction = "L" if row['Period'] % 2 == 0 else "R"

    if row['shooter_hand'] != direction:
        return 1 if row['yC_adj'] < 0 else 0
    else:
        return 0 if row['yC_adj'] < 0 else 1


def get_previous_event_info(df):
    """
    Get the info for the last event
    Note: The pbp is already sorted by game, period, and time...so don't need to worry about time
    
    :param df: DataFrame
    
    :return: df with stuff added
    """
    # Get All previous shit
    df['prev_event'] = df.groupby(['Game_Id', 'Period'])['Event'].shift(1)
    df['prev_ev_team'] = df.groupby(['Game_Id', 'Period'])['Ev_Team'].shift(1)
    df['prev_seconds'] = df.groupby(['Game_Id', 'Period'])['Seconds_Elapsed'].shift(1)
    df['time_elapsed'] = df['Seconds_Elapsed'] - df['prev_seconds']
    df['prev_ev_zone'] = df.groupby(['Game_Id', 'Period'])['Ev_Zone'].shift(1)
    df['prev_home_zone'] = df.groupby(['Game_Id', 'Period'])['Home_Zone'].shift(1)
    df['prev_xC_adj'] = df.groupby(['Game_Id', 'Period'])['xC_adj'].shift(1)
    df['prev_yC_adj'] = df.groupby(['Game_Id', 'Period'])['yC_adj'].shift(1)

    # Change giveaway to takeaway for other team
    df['prev_ev_team'] = np.where(df['prev_event'] != "GIVE", df["prev_ev_team"],
                                  np.where(df['prev_ev_team'] == df['Home_Team'], df["Away_Team"], df["Home_Team"]))
    df['prev_event'] = np.where(df['prev_event'] == "GIVE", "TAKE", df['prev_event'])

    # If last event was by event team
    df['if_prev_ev_team'] = np.where(df['Ev_Team'] == df['prev_ev_team'], 1, 0)

    # Get if last event was by event team for specified events
    df['prev_evTeam_Fac'] = np.where((df['if_prev_ev_team'] == 1) & (df['prev_event'] == "FAC"), 1, 0)
    df['prev_evTeam_NonSog'] = np.where((df['if_prev_ev_team'] == 1) & (df['prev_event'].isin(["MISS", "BLOCK"])), 1, 0)
    df['prev_evTeam_NonShot'] = np.where((df['if_prev_ev_team'] == 1) & (df['prev_event'].isin(["TAKE", "HIT"])), 1, 0)
    df['prev_evTeam_Sog'] = np.where((df['if_prev_ev_team'] == 1) & (df['prev_event'] == "SHOT"), 1, 0)

    # Get if last event was by non-event team for specified events
    df['prev_non_evTeam_Fac'] = np.where((df['if_prev_ev_team'] == 0) & (df['prev_event'] == "FAC"), 1, 0)
    df['prev_non_evTeam_NonSog'] = np.where((df['if_prev_ev_team'] == 0) & (df['prev_event'].isin(["MISS", "BLOCK"])),1, 0)
    df['prev_non_evTeam_NonShot'] = np.where((df['if_prev_ev_team'] == 0) & (df['prev_event'].isin(["TAKE", "HIT"])), 1,0)
    df['prev_non_evTeam_Sog'] = np.where((df['if_prev_ev_team'] == 0) & (df['prev_event'] == "SHOT"), 1, 0)

    # Rebound - less than 2
    df['is_rebound'] = np.where((df['prev_event'] == "SHOT") & (df['Seconds_Elapsed'] - df['prev_seconds'] <= 2.0)
                                & (df['Ev_Team'] == df['prev_ev_team']), 1, 0)

    # Rush shot defined like how Manny does
    df['is_rush'] = np.where(
        ((df['Seconds_Elapsed'] - df['prev_seconds'] <= 4.0) & (
        (df['Home_Zone'] != df['prev_home_zone']) & (df['prev_home_zone'] != "Neu")))
        |
        ((df['Seconds_Elapsed'] - df['prev_seconds'] <= 4.0) & (df['prev_event'].isin(["TAKE", "GIVE"]))),
        1, 0
    )

    # Non_Shot_rebound - miss or block and less or equal to 2
    df['non_sog_rebound'] = np.where(
        (df['prev_event'].isin(["MISS", "BLOCK"])) & (df['Seconds_Elapsed'] - df['prev_seconds'] <= 2.0)
        & (df['Ev_Team'] == df['prev_ev_team']), 1, 0)

    return df


def clean_pbp(pbp):
    """
    Clean the pbp:
    1. Add new columns
    2. Get rid of unnecessary columns
    
    :param pbp: DataFrame for pbp
    
    :return: cleaned up pbp
    """
    pbp = pbp.sort_values(['Game_Id', 'Period', 'Seconds_Elapsed'], ascending=True)
    pbp = pbp[~pbp.Event.isin(["STOP", "PENL", "PEND"])]

    # Fix Scores!!
    pbp['Home_Score'], pbp["Away_Score"] = zip(*pbp.apply(lambda row: fix_score_cat(row), axis=1))

    # Get rid of shootouts
    pbp.drop(pbp[(pbp['Period'] == 5) & (pbp['Game_Id'] < 30000)].index, inplace=True)

    # Get rid of plays without coordinates
    pbp = pbp[pbp["xC"].notnull()]
    pbp = pbp[pbp["yC"].notnull()]

    # Adjust xC and yC
    print("Rink Adjust")
    pbp = rink_adjust.adjust(pbp)

    pbp = pbp[pbp["xC_adj"].notnull()]
    pbp = pbp[pbp["yC_adj"].notnull()]

    # Get previous event info
    print("Get previous info")
    pbp = get_previous_event_info(pbp)

    # "Legal" Strengths
    strengths = ['5x5', '6x5', '5x6', '5x4', '4x5', '5x3', '3x5', '4x3', '4x4', '3x4', '3x3', '6x4', '4x6', '6x3', '3x6']
    pbp = pbp[pbp.Strength.isin(strengths)]

    pbp = pbp[pbp["prev_xC_adj"].notnull()]
    pbp = pbp[pbp["prev_yC_adj"].notnull()]

    # Now just need these event
    pbp = pbp[pbp.Event.isin(["SHOT", "GOAL", "MISS"])]

    # Make score category for home team
    pbp['score_cat'] = np.where(pbp['Home_Score'] - pbp['Away_Score'] >= 3, 3,
                                np.where(pbp['Home_Score'] - pbp['Away_Score'] <= -3, -3, pbp['Home_Score'] - pbp['Away_Score']))

    # Fill nan's with NA
    pbp['Type'].fillna("NA", inplace=True)

    # Misclassify some this way but probably better
    pbp['Distance'] = pbp.apply(lambda row: sqrt(((89.45 - abs(row['xC_adj']))**2 + (row['yC_adj'] ** 2))), axis=1)
    pbp['xC_adj'] = np.where(pbp['xC_adj'] == 0, 1, pbp['xC_adj'])
    pbp['Angle'] = pbp.apply(lambda row: 90 if row['yC_adj'] == 0 else np.arctan((89.45-abs(row['xC_adj']))/row['yC_adj'])
                                                                        * (180/np.pi), axis=1)

    print("Get Shooter info")
    pbp['shooter_hand'], pbp["shooter_pos"] = get_shooter_info(pbp)

    print("Off wing")
    pbp['off_wing'] = pbp.apply(lambda row: if_off_wing(row), axis=1)

    # Adjust if away team who shot it
    pbp['score_cat'] = np.where(pbp['Ev_Team'] == pbp['Home_Team'], pbp['score_cat'], -pbp['score_cat'])
    pbp['Strength'] = pbp.apply(lambda row: fix_strength(row), axis=1)
    pbp['if_home'] = np.where(pbp['Ev_Team'] == pbp['Home_Team'], 1, 0)

    # If Empty net
    print("Empty nets")
    pbp['Home_Goalie'].fillna("Empty", inplace=True)
    pbp['Away_Goalie'].fillna("Empty", inplace=True)
    pbp['empty_net'] = pbp.apply(lambda row: if_empty_net(row), axis=1)

    # Fix previous row
    pbp = pbp.apply(lambda row: fix_prev_event(row), axis=1)

    print("Get New Stuff")

    pbp['prev_angle'] = pbp.groupby(['Game_Id', 'Period'])['Angle'].shift(1)
    pbp['angle_change'] = pbp.apply(lambda row: get_angle_change(row), axis=1)
    pbp['Angle'] = abs(pbp['Angle'])
    pbp.drop(['prev_angle'], axis=1, inplace=True)

    # Distance from last event
    pbp['distance_change'] = pbp.apply(lambda x: get_distance(x), axis=1)

    # Get rid of goalies who took shots
    pbp = pbp[pbp.shooter_pos != "G"]

    # Change RW, LW, C --> F
    pbp['if_forward'] = np.where(pbp['shooter_pos'].isin(["LW", "RW", "C"]), 1, 0)

    # Label outcomes
    pbp['Outcome'] = np.where(pbp['Event'] == "GOAL", 2, np.where(pbp['Event'] == "SHOT", 1,
                                                                  np.where(pbp['Event'] == "MISS", 0, 3)))
    pbp = pbp[pbp['Outcome'] != 3]

    # Get rid of duplicates
    pbp.drop_duplicates(['Game_Id', 'Period', 'Event', 'Seconds_Elapsed'], inplace=True)

    return pbp


def convert_data(pbp):
        """      
        Convert the data to use in model
        
        """

        all_variables = ['off_wing',
                         'Distance', 'Angle',
                         'empty_net',
                         'angle_change',
                         'distance_change', 'time_elapsed',
                         'Type_BACKHAND', 'Type_DEFLECTED', 'Type_SLAP SHOT', 'Type_SNAP SHOT', 'Type_TIP-IN',
                         'Type_WRAP-AROUND', 'Type_WRIST SHOT',
                         'Strength_3x3', 'Strength_3x4', 'Strength_3x5', 'Strength_3x6', 'Strength_4x3', 'Strength_4x4',
                         'Strength_4x5', 'Strength_4x6', 'Strength_5x3', 'Strength_5x4', 'Strength_5x5', 'Strength_5x6',
                         'Strength_6x3', 'Strength_6x4', 'Strength_6x5',
                         'score_cat_-3', 'score_cat_-2', 'score_cat_-1', 'score_cat_0', 'score_cat_1', 'score_cat_2', 'score_cat_3',
                         'if_forward',
                         'if_home',
                         'prev_evTeam_Fac', 'prev_evTeam_NonSog', 'prev_evTeam_NonShot', 'prev_evTeam_Sog',
                         'prev_non_evTeam_Fac', 'prev_non_evTeam_NonSog', 'prev_non_evTeam_NonShot', 'prev_non_evTeam_Sog',
                         ]

        categorical_variables = ['Type', 'score_cat', 'Strength']
        labels = ['Outcome']

        df_dummies = pd.get_dummies(pbp, columns=categorical_variables)
        df_dummies = check_columns(df_dummies)
        model_df = df_dummies[all_variables + ["Outcome"]]
        model_df.dropna(inplace=True)

        model_features = model_df[all_variables].values.tolist()
        model_labels = model_df[labels].values.tolist()

        return model_features, model_labels


def get_data():
    """
    Get DataFrame with all the data 
    
    :return: DataFrame with info from 2007-2016
    """
    year = 2007

    dfs = []
    while year < 2017:
        print(year)
        df = pd.read_csv("nhl_pbp{a}{b}.csv".format(a=str(year), b=str(year+1)), sep=',')
        year += 1
        dfs.append(clean_pbp(df))

    master_df = pd.concat(dfs)

    # Don't know why I'm doing this here...
    master_df.drop(['p2_name', 'p2_ID', 'p3_name', 'p3_ID', 'awayPlayer1', 'awayPlayer1_id', 'awayPlayer2',
                    'awayPlayer2_id', 'awayPlayer3', 'awayPlayer3_id', 'awayPlayer4', 'awayPlayer4_id', 'awayPlayer5',
                    'awayPlayer5_id', 'awayPlayer6', 'awayPlayer6_id', 'homePlayer1', 'homePlayer1_id', 'homePlayer2',
                    'homePlayer2_id', 'homePlayer3', 'homePlayer3_id', 'homePlayer4', 'homePlayer4_id', 'homePlayer5',
                    'homePlayer5_id', 'homePlayer6', 'homePlayer6_id'], axis=1, inplace=True)

    return master_df.reset_index(drop=True)





