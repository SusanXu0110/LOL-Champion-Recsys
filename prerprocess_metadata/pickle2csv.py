import pandas as pd
from collections import defaultdict

# Function to read pickle file
def read_pickle_file(file_path):
    data = pd.read_pickle(file_path)
    return data

# Function to process data and extract necessary fields from the pickle file
def process_data(file_path, summoner_ids, champion_ids, win_values, game_counts):
    data = read_pickle_file(file_path)

    if isinstance(data, pd.DataFrame):
        for index, row in data.iterrows():
            participant_identity = row.get('participantIdentities', [])
            participants = row.get('participants', [])

            if isinstance(participant_identity, list) and isinstance(participants, list):
                for identity, participant in zip(participant_identity, participants):
                    player = identity.get('player', {})
                    summoner_id = player.get('summonerId')
                    champion_id = participant.get('championId')
                    stats = participant.get('stats', {})
                    win = stats.get('win')

                    if summoner_id is not None and champion_id is not None and win is not None:
                        summoner_ids.append(summoner_id)
                        champion_ids.append(champion_id)
                        win_values.append(win)
                        game_counts[(summoner_id, champion_id)] += 1  # Increment game count for the summoner-champion pair

# Initialize empty lists to store data extracted from the file
summoner_ids = []
champion_ids = []
win_values = []
game_counts = defaultdict(int)  # Dictionary to store game counts

# Process the first pickle file
file_path1 = 'data/LOL_match/match_data_version1.pickle'
process_data(file_path1, summoner_ids, champion_ids, win_values, game_counts)

# If needed, process the second pickle file
file_path2 = 'data/LOL_match/match_data_version2.pickle'
process_data(file_path2, summoner_ids, champion_ids, win_values, game_counts)

# Print the length of extracted data
print(f"\nLength of Summoner IDs: {len(summoner_ids)}")
print(f"Length of Champion IDs: {len(champion_ids)}")
print(f"Length of Win Values: {len(win_values)}")

# Use defaultdict to create nested dictionaries to count wins and games for each summoner and champion
summoner_champion_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {summoner_id: {champion_id: [win_count, game_count]}}

# Populate the summoner-champion statistics
for summoner_id, champion_id, win in zip(summoner_ids, champion_ids, win_values):
    summoner_champion_stats[summoner_id][champion_id][1] += 1  # Increment game count
    if win:
        summoner_champion_stats[summoner_id][champion_id][0] += 1  # Increment win count

# Create dictionaries to store win rates and play rates
summoner_champion_winrate = {}
summoner_champion_playrate = {}

# Calculate win rates and play rates for each summoner and champion
for summoner_id, champions in summoner_champion_stats.items():
    total_games = sum(counts[1] for counts in champions.values())
    for champion_id, (win_count, game_count) in champions.items():
        win_rate = win_count / game_count if game_count > 0 else 0
        play_rate = game_count / total_games if total_games > 0 else 0
        if summoner_id not in summoner_champion_winrate:
            summoner_champion_winrate[summoner_id] = {}
            summoner_champion_playrate[summoner_id] = {}
        summoner_champion_winrate[summoner_id][champion_id] = win_rate
        summoner_champion_playrate[summoner_id][champion_id] = play_rate

# Create lists to store win rates, play rates, and game counts for each summoner-champion pair
win_rate_list = []
play_rate_list = []
game_count_list = []

# Populate win rate, play rate, and game count lists
for summoner_id, champion_id in zip(summoner_ids, champion_ids):
    win_rate = summoner_champion_winrate.get(summoner_id, {}).get(champion_id, 0)
    play_rate = summoner_champion_playrate.get(summoner_id, {}).get(champion_id, 0)
    game_count = game_counts[(summoner_id, champion_id)]
    win_rate_list.append(win_rate)
    play_rate_list.append(play_rate)
    game_count_list.append(game_count)

# Print the length of win rate and play rate lists
print(f"\nLength of Win Rate List: {len(win_rate_list)}")
print(f"Length of Play Rate List: {len(play_rate_list)}")
print(f"Length of Game Count List: {len(game_count_list)}")

# Create a DataFrame and save it as a CSV file
data = {
    'summoner_ids': summoner_ids,
    'champion_ids': champion_ids,
    'win_rate_list': win_rate_list,
    'play_rate_list': play_rate_list,
    'game_counts': game_count_list
}
df = pd.DataFrame(data)
output_file_path = 'data/LOL_match/summoner_champion_stats_with_game_count.csv'
df.to_csv(output_file_path, index=False)
print(f"Data has been saved to {output_file_path}")
