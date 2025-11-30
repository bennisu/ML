import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


if __name__ == "__main__":
    #seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
    # prepare everything for one season, then loop over all seasons
    seasons = ["2021-2022"]
    plot_classifiers = False
    plot_home_team_possession = False
    for season in seasons:
        data_dir_path = os.path.join(os.path.expanduser("~"), "Data Science & AI", "football_analytics")
        data_path = os.path.join(data_dir_path, f"season_{season}.ods")
        season_data = pd.read_excel(data_path)
        season_data.set_index("uid", inplace=True)

        if plot_classifiers:
            plt.style.use('_mpl-gallery')

            # plot simple classifier distribution
            fig, ax = plt.subplots()

            ax.hist(season_data["classifier"], bins=3,  edgecolor="white")

            ax.set(xlim=(-1.2, 1.2),
                   ylim=(0, 200), yticks=np.linspace(0, 200, 41))
            ax.set_yticklabels(np.linspace(0, 200, 41), fontsize=3)
            ax.set_xticklabels([0, "Away Win", "Draw", "Home Win"], fontsize=3)

            fig.savefig(os.path.join(data_dir_path, f"classifier_dist_{season}.eps"), bbox_inches='tight')

        if plot_home_team_possession:
            fig, ax = plt.subplots()

            ax.hist(season_data["home_team_possession"], bins=10,  edgecolor="white")

            ax.set(xlim=(0, 1),
                   ylim=(0, 100))
            #ax.set_ylabel("Occurences",fontsize="small")
            ax.set_yticklabels(np.linspace(0, 100, 11), fontsize=3)
            # ax.set_xticklabels([0, "Away Win", "Draw", "Home Win"], fontsize=3)

            fig.savefig(os.path.join(data_dir_path, f"home_team_possession_dist_{season}.eps"), bbox_inches='tight')

        team_name = "fc-bayern-muenchen"

        team_data = season_data[(season_data["home_team"]==team_name) | (season_data["away_team"]==team_name)]
        home_possesion = team_data[team_data["home_team"]==team_name][["home_team_possession","matchday"]]
        away_possesion = team_data[team_data["away_team"]==team_name][["away_team_possession","matchday"]]
        home_possesion.rename(columns={"home_team_possession": "possession"}, inplace=True)
        away_possesion.rename(columns={"away_team_possession": "possession"}, inplace=True)
        possession_data = pd.concat([home_possesion, away_possesion])
        possession_data.sort_values("matchday", inplace=True)
        number_of_matches_to_aggregate = 1
        lower_bound = 1
        upper_bound = lower_bound + number_of_matches_to_aggregate
        means = []
        while upper_bound <= 34:
            sub_series = possession_data[(possession_data["matchday"] < upper_bound) & (possession_data["matchday"] >= lower_bound)]["possession"]
            lower_bound += 1
            upper_bound = lower_bound + number_of_matches_to_aggregate
            means.append(sub_series.mean())


        mean_series = pd.Series(means)
        with open(os.path.join(data_dir_path, f"{team_name}_possession_dist_{season}_{number_of_matches_to_aggregate}_match_average.txt"), "w") as f:
                  f.write(mean_series.describe().to_string())
        fig, ax = plt.subplots()
        ax.hist(means, bins=10,  edgecolor="white")
        ax.set(xlim=(0, 1),
                   ylim=(0, 10))
        ax.set_yticklabels(np.linspace(0, 10, 11), fontsize=10)
        #    # ax.set_xticklabels([0, "Away Win", "Draw", "Home Win"], fontsize=3)
        fig.savefig(os.path.join(data_dir_path, f"{team_name}_possession_dist_{season}_{number_of_matches_to_aggregate}_match_average.eps"), bbox_inches='tight')
        #plt.show()

