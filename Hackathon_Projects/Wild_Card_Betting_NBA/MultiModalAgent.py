import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random


"""



dataset1 = pd.read_excel('advancedNbaStats.xlsx')
dataset2 = pd.read_excel('estimateStatsNba.xlsx')

dataset1_cleaned = dataset1.dropna(axis=1, how='all')
dataset2_cleaned = dataset2.dropna(axis=1, how='all')

df1 = dataset1_cleaned.drop(dataset1_cleaned.columns[[0]], axis=1)
df2 = dataset2_cleaned.drop(dataset2_cleaned.columns[[0]], axis=1)

print(df1.shape[1])
print(df2.shape[1] -5)

merged_df = pd.merge(df1, df2[['PLAYER'] + df2.columns[5:].tolist()], on='PLAYER', how='left')

display(merged_df)
print(merged_df.shape[1])

dataset3 = pd.read_excel('nbaplayerPoints.xlsx')
dfPoints = dataset3.loc[:, ['Player', 'FG%', '3P%', 'FT%']]
display(dfPoints)

df_final = pd.read_excel('combined_data.xlsx')

dfCombined = df_final.merge(dfPoints, left_on='PLAYER', right_on='Player', how='left')
dfCombinedFinal = dfCombined.drop(columns=['Player_x', 'Player_y'])
display(dfCombinedFinal)


scaler = MinMaxScaler()
normalized_columns = scaler.fit_transform(dfCombinedFinal.iloc[:, 2:])

# Create a new DataFrame with the normalized columns
df_normalized = pd.DataFrame(normalized_columns, columns=dfCombinedFinal.columns[2:])

# Concatenate the unchanged columns (A and B) with the normalized columns
df_final2 = pd.concat([dfCombinedFinal.iloc[:, :2], df_normalized], axis=1)
display(df_final2)

file_path = '/content/drive/My Drive/NBAData/finishedData.xlsx'
df_final2.to_excel(file_path, index=False)


merged_df = pd.read_excel('combined_data.xlsx')
merged_df1 = merged_df.drop(columns=['Player'])
scaler = MinMaxScaler()
display(merged_df1)
normalized_columns = scaler.fit_transform(merged_df1.iloc[:, 2:])

# Create a new DataFrame with the normalized columns
df_normalized = pd.DataFrame(normalized_columns, columns=merged_df1.columns[2:])

# Concatenate the unchanged columns (A and B) with the normalized columns
df_final = pd.concat([merged_df1.iloc[:, :2], df_normalized], axis=1)
display(df_final)

"""
df = pd.read_excel("realStats.xlsx")
df_1 = df.drop(columns=["Rk", "Awards"])

initial_row_count = len(df_1)  # Row count before dropna
merged_df1 = df_1.dropna()  # Drop rows with NaN values
final_row_count = len(merged_df1)  # Row count after dropna


players_to_drop = ["Player"]

# Drop rows where 'Player' is in the list
df_filtered = merged_df1[~merged_df1["Player"].isin(players_to_drop)]

df_new = df_filtered.copy()

# Convert '2P' column to float in the new dataset
df_new["2P"] = pd.to_numeric(df_new["2P"], errors="coerce")


scaler = MinMaxScaler()

columns_to_normalize = df_new.columns.difference(["Player", "Team", "Pos"])

# Apply the scaler to the selected columns
normalized_data = scaler.fit_transform(df_new[columns_to_normalize])

# Create a new DataFrame with the normalized data
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)

# Add the non-normalized columns back to the DataFrame
df_final111 = pd.concat([df[["Player", "Team", "Pos"]], normalized_df], axis=1)


df_final1 = pd.read_excel("./completedDoneData.xlsx")
df_final2 = pd.read_excel("./nbaPlayersAdvancedStats.xlsx")


merged_df2 = df_final2.drop(columns=["Rk", "Awards"]).dropna()
players_to_drop = ["Player"]
print("----------------------------------------------")
initial_row_count = len(merged_df2)  # Row count before dropna
merged_df3 = merged_df2.dropna()  # Drop rows with NaN values
final_row_count = len(merged_df3)  # Row count after dropna


# Drop rows where 'Player' is in the list
df_filtered = merged_df2[~merged_df2["Player"].isin(players_to_drop)]


df_new = df_filtered.copy()
df_new["3PAr"] = pd.to_numeric(df_new["3PAr"], errors="coerce")


scaler = MinMaxScaler()

# Select columns to normalize
columns_to_normalize1 = df_new.columns.difference(["Player", "Team", "Pos"])

# Apply the scaler to the selected columns while preserving the original index
normalized_data1 = scaler.fit_transform(df_new[columns_to_normalize1])

# Create a new DataFrame with the normalized data, ensuring it retains the same index as the original
normalized_df1 = pd.DataFrame(
    normalized_data1, columns=columns_to_normalize1, index=df_new.index
)

# Add the non-normalized columns back to the DataFrame
df_final3 = pd.concat([df_new[["Player", "Team", "Pos"]], normalized_df1], axis=1)

df_final1.columns = df_final1.columns.str.strip()
df_final3.columns = df_final3.columns.str.strip()


# Identify common columns (excluding 'Player' as it's the merge key)
common_columns = df_final1.columns.intersection(df_final3.columns).tolist()
common_columns.remove("Player")
common_columns.remove("G")
print(common_columns)

# Drop the common columns from df_final3 (except 'Player')
df_final3_unique = df_final3.drop(columns=common_columns, errors="ignore")


# Merge the DataFrames on the 'Player' column


def preprocess_duplicates(df):
    """
    Preprocess the dataset to fix duplicates by keeping the player with the fewest games played.

    Args:
        df (pd.DataFrame): The player statistics dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe with no duplicate player names.
    """
    # Sort by Player name and Games played (ascending, so fewest games come first)
    df_sorted = df.sort_values(by=["Player", "G"], ascending=[True, True])

    # Remove duplicates, keeping the player with the fewest games played
    df_cleaned = df_sorted.drop_duplicates(subset="Player", keep="first")

    return df_cleaned


df_finall2345 = preprocess_duplicates(df_final1)
players_to_drop = ["Player", "League Average"]

# Print separator for clarity
print("----------------------------------------------")

# Row count before filtering and dropping NaN values
initial_row_count = len(df_finall2345)
print(f"Initial row count: {initial_row_count}")

# Filter out rows where 'Player' is in the list and drop rows with NaN values
df_final = df_finall2345[~df_finall2345["Player"].isin(players_to_drop)].dropna()

# Row count after filtering and dropping NaN values
final_row_count = len(df_final)
print(f"Final row count: {final_row_count}")


print("Here------------")
import re

# Check if any entry in the 'Player' column contains the word "Player" (case-insensitive)
player_column_check = df_final["Player"].apply(
    lambda x: bool(re.search(r"player", str(x), re.IGNORECASE))
)

# If any entry contains the word "Player" (case insensitive), raise an issue
if player_column_check.any():
    print(
        "There is at least one entry in the 'Player' column that contains 'Player' or a variation of it."
    )
else:
    print("No entry in the 'Player' column contains 'Player' or any variation of it.")


print("Here------------")


# top_players = df_final.groupby('Team').apply(lambda x: x.nlargest(5, 'MP')).reset_index(drop=True)
print(df_final.columns)

import numpy as np
import pandas as pd
import random

# Define the states and allowed transitions
import numpy as np
import pandas as pd
import random

# ----- Setup -----
states = [
    "Start Possession",
    "Pass",
    "SM2",
    "SM3",
    "SMiss2",
    "SMiss3",
    "Rebound",
    "Turnover",
    "Foul",
    "BLK",
    "STL",
    "End Possession",
]

allowed_transitions = {
    "Start Possession": [
        "Pass",
        "SM2",
        "SM3",
        "SMiss2",
        "SMiss3",
        "Turnover",
        "Foul",
        "End Possession",
    ],
    "Pass": ["SM2", "SM3", "SMiss2", "SMiss3", "Turnover", "Foul", "End Possession"],
    "SM2": ["End Possession"],
    "SM3": ["End Possession"],
    "SMiss2": ["Rebound", "Turnover", "Foul", "End Possession"],
    "SMiss3": ["Rebound", "Turnover", "Foul", "End Possession"],
    "Rebound": ["Pass", "SM2", "SM3", "Turnover", "BLK", "STL"],
    "Turnover": ["BLK", "STL", "Foul", "End Possession"],
    "Foul": ["BLK", "STL", "End Possession"],
    "BLK": ["End Possession"],
    "STL": ["End Possession"],
    "End Possession": ["End Possession"],
}


def compute_transition_matrix(player_stats, states, allowed_transitions):
    """
    Compute a stochastic transition matrix for a player based on their stats.
    The probabilities use shooting percentages, turnover rate, passing likelihood, etc.
    """
    num_states = len(states)
    transition_matrix = np.zeros((num_states, num_states))

    # Extract player stats
    shot_2P = player_stats["2P%"]  # Probability of making a 2P shot
    shot_3P = player_stats["3P%"]  # Probability of making a 3P shot
    turnover_rate = player_stats["TOV"] / (
        player_stats["AST"] + player_stats["TOV"] + 1e-6
    )
    pass_prob = player_stats["AST"] / (player_stats["AST"] + player_stats["TOV"] + 1e-6)
    foul_prob = 0.05  # fixed foul probability

    # Compute block and steal base probabilities using a similar denominator as for turnover.
    # Here we use (AST + TOV) to normalize.
    block_base_prob = player_stats["BLK"] / (
        player_stats["AST"] + player_stats["TOV"] + 1e-6
    )
    steal_base_prob = player_stats["STL"] / (
        player_stats["AST"] + player_stats["TOV"] + 1e-6
    )

    # Extremely reduce block and steal probabilities: reduce to 1% of the base value, with a minimum floor.
    block_prob = max(0.0001, block_base_prob * 0.01)
    steal_prob = max(0.0001, steal_base_prob * 0.01)

    # Drastically reduce turnover probability (reduce by a factor of 10, floor at 0.01)
    turnover_rate = max(0.01, turnover_rate * 0.1)

    # For each current state, assign probabilities to allowed next states
    for i, current_state in enumerate(states):
        if current_state not in allowed_transitions:
            continue
        next_states = allowed_transitions[current_state]
        probs = []
        for next_state in next_states:
            if next_state == "Pass":
                probs.append(pass_prob)
            elif next_state == "SM2":
                probs.append(shot_2P * (1 - turnover_rate))
            elif next_state == "SM3":
                probs.append(shot_3P * (1 - turnover_rate))
            elif next_state == "SMiss2":
                probs.append((1 - shot_2P) * (1 - turnover_rate))
            elif next_state == "SMiss3":
                probs.append((1 - shot_3P) * (1 - turnover_rate))
            elif next_state == "Turnover":
                probs.append(turnover_rate)
            elif next_state == "Foul":
                probs.append(foul_prob)
            elif next_state == "Rebound":
                probs.append(0.3)  # assumed constant for now
            elif next_state == "End Possession":
                probs.append(1)  # deterministic end
            elif next_state == "BLK":
                probs.append(block_prob)
            elif next_state == "STL":
                probs.append(steal_prob)

        # For states that are deterministic (SM2 and SM3 always lead to End Possession)
        if current_state in ["SM2", "SM3"]:
            probs = [1 if state == "End Possession" else 0 for state in next_states]
        else:
            probs = np.array(probs, dtype=np.float64)
            if probs.sum() > 0:
                probs /= probs.sum()

        indices = [states.index(s) for s in next_states]
        transition_matrix[i, indices] = probs

    return transition_matrix


# ----- Sample Data (df_final) -----


# Compute each player's transition matrix and store in df_final
df_final["Transition Matrix"] = df_final.apply(
    lambda row: compute_transition_matrix(row, states, allowed_transitions), axis=1
)


# ----- Selecting 5 Players Per Team -----
def select_top_players(df, team, num_players=5):
    """
    Select the top 'num_players' for a given team based on a composite metric combining
    'PTS' (70% weight) and 'eFG%' (30% weight).

    Parameters:
        df (DataFrame): The dataframe containing player statistics.
        team (str): The team for which to select top players.
        num_players (int): The number of top players to select.

    Returns:
        DataFrame: The top players sorted by the composite metric.
    """
    # Filter players by team
    team_df = df[df["Team"] == team].copy()

    # Handle missing values in 'PTS' or 'eFG%' (set to 0 if missing)
    team_df["PTS"] = team_df["PTS"].fillna(0)
    team_df["eFG%"] = team_df["eFG%"].fillna(0)

    # Composite metric: 70% weight to PTS, 30% to eFG%
    team_df["Composite"] = team_df["PTS"] * 0.7 + team_df["eFG%"] * 0.3

    # Sort by composite metric and select the top 'num_players'
    top_players = team_df.sort_values(by="Composite", ascending=False).head(num_players)

    return top_players


# ----- Simulation -----
def initialize_stats(team_players):
    """
    Initialize player and team stats for the simulation.

    Parameters:
        team_players (DataFrame): A dataframe containing player names and their associated teams.

    Returns:
        player_stats (dict): A dictionary containing the initial stats for each player.
        team_stats (dict): A dictionary containing the initial stats for each team.
    """
    # Initialize stats for each player
    player_stats = {
        player: {
            "Team": team,
            "PTS": 0,
            "REB": 0,
            "AST": 0,
            "BLK": 0,
            "STL": 0,
            "TOV": 0,
        }
        for player, team in zip(team_players["Player"], team_players["Team"])
    }

    # Initialize team stats (default to 0 for all categories)
    team_stats = {
        team: {"PTS": 0, "REB": 0, "AST": 0, "BLK": 0, "STL": 0, "TOV": 0}
        for team in team_players["Team"].unique()
    }

    return player_stats, team_stats


# ----- Simulation -----

import random
import numpy as np


def simulate_game(selected_players, df_final, selected_teams):
    """
    Simulate a full basketball game between two teams based on player stats.
    """
    team1, team2 = selected_teams
    team_players = selected_players.copy()

    player_stats, team_stats = initialize_stats(team_players)
    game_log = []

    # Start with a random team
    possession_team = random.choice([team1, team2])
    game_log.append(f"Game starts with {possession_team}")

    last_passer = None
    max_possessions = 800  # Full game simulation

    for pos in range(max_possessions):
        current_state = "Start Possession"
        game_log.append(f"{possession_team} starts possession")

        # Select a random player from the team in possession
        team_roster = team_players[team_players["Team"] == possession_team]
        if team_roster.empty:
            raise ValueError(f"No players available for team {possession_team}.")
        player = random.choice(team_roster["Player"].tolist())

        # Get player's data from df_final (should be present)
        if player not in df_final["Player"].values:
            raise ValueError(f"{player} not found in df_final.")
        player_data = df_final[df_final["Player"] == player].iloc[0]

        two_p_percentage = player_data["2P%"]
        three_p_percentage = player_data["3P%"]

        transition_matrix = player_data["Transition Matrix"]
        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)

        state_index = states.index("Start Possession")

        while current_state != "End Possession":
            try:
                next_state_index = np.random.choice(
                    len(states), p=transition_matrix[state_index]
                )
            except ValueError as e:
                raise ValueError(
                    f"Error for {player} at state '{states[state_index]}': {transition_matrix[state_index]}"
                ) from e

            next_state = states[next_state_index]
            game_log.append(f"{player} transitions to {next_state}")

            if next_state == "Pass":
                last_passer = player
                game_log.append(f"{player} passes the ball.")
                # --- Steal Check ---
                opponent_roster = team_players[team_players["Team"] != possession_team]
                if not opponent_roster.empty:
                    defender = random.choice(opponent_roster["Player"].tolist())
                    defender_data = df_final[df_final["Player"] == defender].iloc[0]
                    # Compute defender steal probability from STL stat divided by games (G)
                    defender_steal_prob = (
                        defender_data["STL"] / defender_data["G"]
                        if defender_data["G"] > 0
                        else 0.05
                    )
                    defender_steal_prob *= 1.1  # Slight boost
                    if random.random() < defender_steal_prob:
                        game_log.append(
                            f"STEAL! {defender} steals the ball from {player}!"
                        )
                        player_stats[defender]["STL"] += 1
                        team_stats[defender_data["Team"]]["STL"] += 1
                        current_state = "End Possession"
                        break
                current_state = "Pass"

            elif next_state in ["SM2", "SM3"]:
                # --- Block Check before shot resolution ---
                opponent_roster = team_players[team_players["Team"] != possession_team]
                if not opponent_roster.empty:
                    defender = random.choice(opponent_roster["Player"].tolist())
                    defender_data = df_final[df_final["Player"] == defender].iloc[0]
                    # Compute defender block probability from BLK stat divided by games (G)
                    defender_block_prob = (
                        defender_data["BLK"] / defender_data["G"]
                        if defender_data["G"] > 0
                        else 0.03
                    )
                    defender_block_prob *= 1.2  # Slight boost
                    if random.random() < defender_block_prob:
                        game_log.append(f"BLOCK! {defender} blocks {player}'s shot.")
                        player_stats[defender]["BLK"] += 1
                        team_stats[defender_data["Team"]]["BLK"] += 1
                        current_state = "End Possession"
                        break
                shot_prob = (
                    two_p_percentage if next_state == "SM2" else three_p_percentage
                )
                if random.random() < shot_prob:
                    pts = 2 if next_state == "SM2" else 3
                    game_log.append(f"{player} scores {pts} points!")
                    player_stats[player]["PTS"] += pts
                    team_stats[possession_team]["PTS"] += pts
                    if last_passer and last_passer != player:
                        game_log.append(f"Assist by {last_passer}")
                        player_stats[last_passer]["AST"] += 1
                    last_passer = None
                else:
                    game_log.append(f"{player} misses the shot.")
                current_state = "End Possession"

            elif next_state == "Turnover":
                game_log.append(f"{player} commits a turnover!")
                player_stats[player]["TOV"] += 1
                team_stats[possession_team]["TOV"] += 1
                current_state = "End Possession"

            elif next_state == "Rebound":
                if random.random() < 0.5:
                    rebounder = random.choice(team_roster["Player"].tolist())
                    game_log.append(f"{rebounder} grabs an offensive rebound!")
                else:
                    opponent_roster = team_players[
                        team_players["Team"] != possession_team
                    ]
                    rebounder = random.choice(opponent_roster["Player"].tolist())
                    game_log.append(f"{rebounder} grabs a defensive rebound!")
                player_stats[rebounder]["REB"] += 1
                team_stats[possession_team]["REB"] += 1
                current_state = "End Possession"

            elif next_state == "Foul":
                game_log.append(f"{player} commits a foul.")
                current_state = "End Possession"

            elif next_state == "End Possession":
                game_log.append(f"{player} ends possession.")
                current_state = "End Possession"

            state_index = next_state_index

        possession_team = team1 if possession_team == team2 else team2

    print("Game simulation complete.")
    return game_log, player_stats, team_stats


# ----- Run Selection and Simulation -----
# Get unique teams from df_final
teams_available = df_final["Team"].unique()
# Randomly select two teams
selected_teams = np.random.choice(teams_available, size=2, replace=False)

# For each selected team, choose the top players (up to 5)
selected_players = pd.concat(
    [select_top_players(df_final, team, num_players=5) for team in selected_teams]
)
print("Selected Teams and their Players:")
for team in selected_teams:
    players = selected_players[selected_players["Team"] == team]["Player"].tolist()
    print(f"{team}: {', '.join(players)}")


def update_team_stats(player_stats, selected_teams):
    """
    This function updates team stats by summing up individual player stats for both teams.
    The stat categories include: 'PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV'.

    Parameters:
        player_stats (dict): A dictionary containing player stats where the keys are player names
                              and the values are dictionaries of stats like 'PTS', 'REB', etc.
        selected_teams (list): A list of two teams involved in the game.

    Returns:
        team_stats (dict): A dictionary containing the updated team stats.
    """
    # Initialize team stats for the selected teams
    team_stats = {
        team: {"PTS": 0, "REB": 0, "AST": 0, "BLK": 0, "STL": 0, "TOV": 0}
        for team in selected_teams
    }

    # Summing stats for both teams
    for player, stats in player_stats.items():
        # Extract the team name from the player's stats and update the right team
        team = stats["Team"]
        if team not in team_stats:
            raise ValueError(f"Unknown team {team} for player {player}.")

        for stat, value in stats.items():
            if stat in team_stats[team]:
                team_stats[team][stat] += value

    return team_stats


def monte_carlo_simulation(selected_teams, df_final, num_simulations=1000):
    """
    Perform Monte Carlo simulations to get the average stats for two selected teams and their players.
    """
    # Initialize total stats for teams
    total_team_stats = {
        selected_teams[0]: {"PTS": 0, "REB": 0, "AST": 0, "BLK": 0, "STL": 0, "TOV": 0},
        selected_teams[1]: {"PTS": 0, "REB": 0, "AST": 0, "BLK": 0, "STL": 0, "TOV": 0},
    }

    # Initialize total stats for players
    total_player_stats = {
        player: {
            "Team": df_final[df_final["Player"] == player]["Team"].values[0],
            "PTS": 0,
            "REB": 0,
            "AST": 0,
            "BLK": 0,
            "STL": 0,
            "TOV": 0,
        }
        for player in df_final["Player"]
        if df_final[df_final["Player"] == player]["Team"].values[0] in selected_teams
    }

    # Run simulations
    for _ in range(num_simulations):
        # Select top players for both teams
        selected_players_team1 = select_top_players(
            df_final, selected_teams[0], num_players=5
        )
        selected_players_team2 = select_top_players(
            df_final, selected_teams[1], num_players=5
        )

        # Simulate the game
        game_log, player_stats, team_stats = simulate_game(
            pd.concat([selected_players_team1, selected_players_team2]),
            df_final,
            selected_teams,
        )

        # Update team stats based on player stats
        updated_team_stats = update_team_stats(player_stats, selected_teams)

        # Add stats to total team stats
        for team in selected_teams:
            for stat in total_team_stats[team]:
                total_team_stats[team][stat] += updated_team_stats[team][stat]

        # Add player stats to total player stats
        for player, stats in player_stats.items():
            if player in total_player_stats:
                for stat in stats:
                    total_player_stats[player][stat] += stats[stat]

    # Calculate average stats for teams
    average_team_stats = {
        team: {
            stat: total_team_stats[team][stat] / num_simulations
            for stat in total_team_stats[team]
        }
        for team in selected_teams
    }

    # Calculate average stats for players
    average_player_stats = {}
    for player, stats in total_player_stats.items():
        average_player_stats[player] = {
            stat: stats[stat] / num_simulations if stat != "Team" else stats[stat]
            for stat in stats
        }

    return average_team_stats, average_player_stats


# Example usage:
num_simulations = 20  # number of Monte Carlo simulations

# Get average stats for the selected teams and players
average_team_stats, average_player_stats = monte_carlo_simulation(
    selected_teams, df_final, num_simulations
)

# Print the average team stats
print("\nAverage Team Stats after Monte Carlo Simulation:")
for team, stats in average_team_stats.items():
    print(f"\n{team} Stats:")
    for stat, avg_value in stats.items():
        print(f"{stat}: {avg_value:.2f}")

# Print the average player stats
players_who_played = selected_players["Player"].tolist()

print("\nAverage Player Stats after Monte Carlo Simulation:")
for player, stats in average_player_stats.items():
    if player in players_who_played:  # Check if the player participated
        print(f"\n{player} Stats:")
        for stat, avg_value in stats.items():
            if isinstance(avg_value, (int, float)):  # Check if it's a number
                print(f"{stat}: {avg_value:.2f}")
            else:
                print(
                    f"{stat}: {avg_value}"
                )  # If it's not a number, just print it as is


def pick_teams_manually(df):
    """
    This function allows the user to manually pick two teams for the simulation.

    Parameters:
        df (DataFrame): The dataframe containing the teams and players data.

    Returns:
        selected_teams (list): A list containing the two selected teams.
    """
    teams = df["Team"].unique()

    print("Available teams:")
    for i, team in enumerate(teams, start=1):
        print(f"{i}. {team}")

    team1_idx = int(input(f"Select the first team (1-{len(teams)}): ")) - 1
    team2_idx = int(input(f"Select the second team (1-{len(teams)}): ")) - 1

    while team1_idx == team2_idx:
        print("You cannot select the same team twice. Please pick different teams.")
        team2_idx = int(input(f"Select the second team (1-{len(teams)}): ")) - 1

    selected_teams = [teams[team1_idx], teams[team2_idx]]

    print(f"You have selected: {selected_teams[0]} vs. {selected_teams[1]}")

    return selected_teams


# Call this function to allow manual selection of teams
selected_teams = pick_teams_manually(df_final)
selected_players = pd.concat(
    [select_top_players(df_final, team, num_players=5) for team in selected_teams]
)
print("Selected Teams and their Players:")
for team in selected_teams:
    players = selected_players[selected_players["Team"] == team]["Player"].tolist()
    print(f"{team}: {', '.join(players)}")

# Run Monte Carlo simulation with the selected teams
average_team_stats, average_player_stats = monte_carlo_simulation(
    selected_teams, df_final, num_simulations
)

# Print the results
print("\nAverage Team Stats after Monte Carlo Simulation:")
for team, stats in average_team_stats.items():
    print(f"\n{team} Stats:")
    for stat, avg_value in stats.items():
        print(f"{stat}: {avg_value:.2f}")

# Print the average player stats
players_who_played = selected_players["Player"].tolist()

print("\nAverage Player Stats after Monte Carlo Simulation:")
for player, stats in average_player_stats.items():
    if player in players_who_played:  # Check if the player participated
        print(f"\n{player} Stats:")
        for stat, avg_value in stats.items():
            if isinstance(avg_value, (int, float)):  # Check if it's a number
                print(f"{stat}: {avg_value:.2f}")
            else:
                print(
                    f"{stat}: {avg_value}"
                )  # If it's not a number, just print it as is
