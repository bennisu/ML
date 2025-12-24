import pandas as pd
import numpy as np
import os
import typing


def _switch_home_and_away_str(input_str: str) -> str:
    if "home" in input_str:
        return input_str.replace("home", "away")
    elif "away" in input_str:
        return input_str.replace("away", "home")
    else:
        return input_str


def _compute_form(previous_match_classifier: int, previous_form: float, previous_form_opponent: float, stealing_fraction: float) -> float:
    if previous_match_classifier == 0:
        form = previous_form - stealing_fraction*(previous_form - previous_form_opponent)
        return form
    else:
        form = previous_form + previous_match_classifier*stealing_fraction*previous_form_opponent
        return form


def _get_single_value_from_series(input_series: pd.Series) -> typing.Any:
    series_values = input_series.values
    if len(series_values) == 1:
        return series_values[0]
    else:
        print(f"Expecting single value in series {input_series}. Multiple values detected! Please chek!")
        exit(-1)


def get_avg_features(matchday: int, lookback_matches: int, team_name: str, season_dataframe: pd.DataFrame) -> pd.Series:
    """Compute externally pre-defined average features for a given matchday based on the past lookback_matches matches.
    The average features are computed for the team given by the team_name.

    To be used as part of the feature creation for a given match.

    Args:
        matchday (int): current matchday to compute feature for
        lookback_matches (int): number of past matches to average over
        team_name (str): name of the team for which to compute the feature
        season_dataframe (pd.DataFrame): full data of the season

    Returns:
        pd.Series: Series with computed features
    """
    assert matchday > lookback_matches
    ht_column_mapping = {"home_team_goals": "scored_goals",
                         "away_team_goals": "received_goals",
                         "home_team_passes": "passes",
                         "home_team_passing_accuracy": "passing_accuracy"}
    at_column_mapping = {_switch_home_and_away_str(key): value for key, value in ht_column_mapping.items()}
    season_dataframe = season_dataframe[(season_dataframe["matchday"] >= matchday-lookback_matches)
                                        & (season_dataframe["matchday"] < matchday)]
    matches_as_ht = season_dataframe[season_dataframe["home_team"] == team_name]
    matches_as_ht = matches_as_ht[list(ht_column_mapping.keys())]
    matches_as_ht.rename(columns=ht_column_mapping, inplace=True)
    matches_as_at = season_dataframe[season_dataframe["away_team"] == team_name]
    matches_as_at = matches_as_at[list(at_column_mapping.keys())]
    matches_as_at.rename(columns=at_column_mapping, inplace=True)
    relevant_matches_df = pd.concat([matches_as_ht, matches_as_at])
    return relevant_matches_df.mean()


def update_form(matchday: int, team_name: str, stealing_fraction: float, forms_df: pd.DataFrame, season_results: pd.DataFrame) -> float:
    """Update the form of a given team for a fixed matchday based on previous form and the stealing fraction.

    Args:
        matchday (int): current matchday to compute form feature for
        team_name (str): current team to compute form feature for
        stealing_fraction (float): parameter for form feature update (see paper)
        forms_df (pd.DataFrame): dataframe with previous form values per team and matchday
        season_results (pd.DataFrame): results of the season under consideration

    Returns:
        float: form feature of team <team_name> for matchday <matchday>
    """
    assert matchday > 1
    previous_match_result = -2
    opponent_name = ""
    previous_matchday_df = season_results[season_results["matchday"] == matchday-1]
    if not previous_matchday_df[previous_matchday_df["home_team"] == team_name].empty:
        previous_matchday_df = previous_matchday_df[previous_matchday_df["home_team"] == team_name]
        opponent_name = _get_single_value_from_series(previous_matchday_df["away_team"])
        previous_match_result = _get_single_value_from_series(previous_matchday_df["classifier"])
    elif not previous_matchday_df[previous_matchday_df["away_team"] == team_name].empty:
        previous_matchday_df = previous_matchday_df[previous_matchday_df["away_team"] == team_name]
        opponent_name = _get_single_value_from_series(previous_matchday_df["home_team"])
        # home team win/loss = +1/-1, away team win/loss = -1/+1
        # update function needs win = +1, loss = -1 so sign is switched
        previous_match_result = -_get_single_value_from_series(previous_matchday_df["classifier"])
    else:
        print(f"No match for team {team_name} on matchday {matchday} found.\n Cannot update form! \n Please check the data!")
        exit()
    previous_form = _get_single_value_from_series(forms_df[(forms_df["matchday"] == matchday-1) & (forms_df["team_name"] == team_name)]["form"])
    previous_form_opponent = _get_single_value_from_series(forms_df[(forms_df["matchday"] == matchday-1) & (forms_df["team_name"] == opponent_name)]["form"])
    return _compute_form(previous_match_result, previous_form, previous_form_opponent, stealing_fraction)


def compute_form_df_for_season(results_df: pd.DataFrame, stealing_fraction: float) -> pd.DataFrame:
    """Compute all form features for every team and matchday for a fixed season based on a fixed stealing fraction.

    Args:
        results_df (pd.DataFrame): contains the full season results
        stealing_fraction (float): parameter for form update

    Returns:
        pd.DataFrame: contains the form features per team and matchday for this season
    """
    team_names = results_df["home_team"].drop_duplicates().values
    initial_matchday = [1]*len(team_names)
    initial_form = [1]*len(team_names)
    form_df = pd.DataFrame({"team_name": team_names, "matchday": initial_matchday, "form": initial_form})
    # Skip first matchday due to initialization
    for matchday in results_df["matchday"].drop_duplicates().values[1::]:
        matchday_forms = []
        current_teams = []
        for team in team_names:
            current_teams.append(team)
            matchday_forms.append(update_form(matchday, team, stealing_fraction, form_df, results_df))
        matchday_df = pd.DataFrame({"team_name": current_teams, "matchday": [matchday]*len(team_names), "form": matchday_forms})
        form_df = pd.concat([form_df, matchday_df], ignore_index=True)
    return form_df


def _turn_classifier_into_home_points(classifier: int) -> int:
    if classifier == -1:
        return 0
    elif classifier == 0:
        return 1
    elif classifier == 1:
        return 3
    else:
        print(f"Uknown result classifier {classifier}! Please check the input data!")
        exit(-1)


def _turn_classifier_into_away_points(classifier: int) -> int:
    if classifier == -1:
        return 3
    elif classifier == 0:
        return 1
    elif classifier == 1:
        return 0
    else:
        print(f"Uknown result classifier {classifier}! Please check the input data!")
        exit(-1)


def _compute_streak_weight(current_matchday: int, considered_matchday: int | None = None, lookback_matches: int | None = None) -> float:
    assert considered_matchday is not None
    assert lookback_matches is not None
    return 2*(current_matchday - considered_matchday + lookback_matches + 1)/(3*lookback_matches*(lookback_matches + 1))

def compute_streak(team_name: str, matchday: int, lookback_matches: int, results_df: pd.DataFrame) -> tuple[float, float]:
    """Compute streak and weighted streak for a single team on a single matchday.

    Args:
        team_name (str): team for which the values are computed
        matchday (int): matchday for which the values should be used as input
        lookback_matches (int): number of matches to consider in the streak computation
        results_df (pd.DataFrame): results of the full season

    Returns:
        streak (float), weighted_streak (float): computed streak and weighted streak
    """
    relevant_matchdays = season_data[(results_df["matchday"] < matchday) & (results_df["matchday"] > matchday - lookback_matches - 1)]
    home_match_classifiers = relevant_matchdays[relevant_matchdays["home_team"] == team_name]["classifier"]
    home_match_matchdays = relevant_matchdays[relevant_matchdays["home_team"] == team_name]["matchday"]
    home_match_matchdays = home_match_matchdays.apply(_compute_streak_weight, considered_matchday=matchday, lookback_matches=lookback_matches)
    home_match_points = home_match_classifiers.map(_turn_classifier_into_home_points)
    home_match_points = pd.merge(home_match_points, home_match_matchdays, on="uid")
    away_match_classifiers = relevant_matchdays[relevant_matchdays["away_team"] == team_name]["classifier"]
    away_match_matchdays = relevant_matchdays[relevant_matchdays["away_team"] == team_name]["matchday"]
    away_match_matchdays = away_match_matchdays.apply(_compute_streak_weight, considered_matchday=matchday, lookback_matches=lookback_matches)
    away_match_points = away_match_classifiers.map(_turn_classifier_into_away_points)
    away_match_points = pd.merge(away_match_points, away_match_matchdays, on="uid")
    points_df = pd.concat([home_match_points, away_match_points])
    points_df.rename({"classifier": "points", "matchday": "streak_weight"}, inplace=True, axis=1)
    weighted_streak = np.sum(points_df["points"]*points_df["streak_weight"])
    streak = np.sum(points_df["points"]/(3*lookback_matches))
    return streak, weighted_streak


def compute_streak_df(results_df: pd.DataFrame, lookback_matches: int) -> pd.DataFrame:
    """Compute all streaks and weighted streaks for a given season and number of lookback matches.

    Args:
        results_df (pd.DataFrame): dataframe containing the results of a season
        lookback_matches (int): number of matches to base the streaks on

    Returns:
        pd.DataFrame: dataframe containing team, matchday and the streaks to be used as features for this matchday
    """
    team_names = results_df["home_team"].drop_duplicates().values
    streak_df = pd.DataFrame(columns=["team_name", "matchday", "streak", "weighted_streak"])
    for matchday in results_df["matchday"].drop_duplicates().values[lookback_matches::]:
        matchday_streaks = []
        matchday_weighted_streaks = []
        sorted_team_list = []
        for team in team_names:
            streak, weighted_streak = compute_streak(team, matchday, lookback_matches, results_df)
            matchday_streaks.append(streak)
            matchday_weighted_streaks.append(weighted_streak)
            sorted_team_list.append(team)
        matchday_df = pd.DataFrame({"team_name": sorted_team_list, "matchday": [matchday]*len(sorted_team_list), "streak": matchday_streaks, "weighted_streak": matchday_weighted_streaks})
        streak_df = pd.concat([streak_df, matchday_df], ignore_index=True)
    return streak_df

def add_goal_efficiencies_to_df(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add home and away team goal efficiencies to a set of matches stored in result_df.
    When shots_on_target is 0, the goal efficiency is set to 0.

    Args:
        result_df (pd.DataFrame): dataframe containing the match results for which the efficiencies are added

    Returns:
        pd.DataFrame: input dataframe with home and away team goal efficienciy columns added
    """
    column_prefixes = ["home_team", "away_team"]
    for column_prefix in column_prefixes:
        result_df[f"{column_prefix}_goal_efficiency"] = result_df[f"{column_prefix}_goals"]/result_df[f"{column_prefix}_shots_on_target"]
        result_df.fillna({f"{column_prefix}_goal_efficiency": 0}, inplace=True)
    return result_df


if __name__ == "__main__":
    season = "2021-2022"
    data_dir_path = os.path.join(os.path.expanduser("~"), "Data Science & AI", "football_analytics")
    data_path = os.path.join(data_dir_path, f"season_{season}.ods")
    season_data = pd.read_excel(data_path)
    season_data.set_index("uid", inplace=True)

    #new_form = update_form(2, "fc-augsburg", 0.33, form_df, season_data)
    #print(new_form)
    #print(get_avg_features(3, 2, "1-fc-koeln", season_data))
    print(compute_streak_df(season_data, 3))


    ## TODO: prepare goal difference, goal efficiency, shot accuracy, and bring all the features together for first analyses
    # In the analyses consider "different time scales" like, short-term streak and long-term streak