import pandas as pd
import numpy as np
import os


def _switch_home_and_away_str(input_str: str) -> str:
    if "home" in input_str:
        return input_str.replace("home", "away")
    elif "away" in input_str:
        return input_str.replace("away", "home")
    else:
        return input_str


def get_avg_features(matchday: int, lookback_matches: int, team_name: str, season_dataframe: pd.DataFrame) -> pd.Series:
    """_summary_

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

if __name__ == "__main__":
    season = "2021-2022"
    data_dir_path = os.path.join(os.path.expanduser("~"), "Data Science & AI", "football_analytics")
    data_path = os.path.join(data_dir_path, f"season_{season}.ods")
    season_data = pd.read_excel(data_path)
    season_data.set_index("uid", inplace=True)
    print(get_avg_features(3, 2, "1-fc-koeln", season_data))



