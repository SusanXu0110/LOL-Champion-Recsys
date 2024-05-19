import pandas as pd

# Define the input file path
input_file_path = 'data/LOL_rank/summoner_champion_stats.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file_path)

# Select the first 50,000 unique summoner IDs
unique_summoner_ids = df['summoner_ids'].unique()[:50000]

# Filter the DataFrame to include only rows with the selected summoner IDs
filtered_df = df[df['summoner_ids'].isin(unique_summoner_ids)]

# Convert the summoner IDs to numeric codes
filtered_df['numeric_summoner_id'] = filtered_df['summoner_ids'].astype('category').cat.codes + 1

# Drop the original summoner ID column
filtered_df.drop(columns=['summoner_ids'], inplace=True)

# Rename the numeric summoner ID column to 'summoner_ids'
filtered_df.rename(columns={'numeric_summoner_id': 'summoner_ids'}, inplace=True)

# Reorder columns to ensure 'summoner_ids' is the first column
cols = ['summoner_ids'] + [col for col in filtered_df if col != 'summoner_ids']
filtered_df = filtered_df[cols]

# Define the output file path
output_file_path = 'summoner_champion_stats_numeric_50000.csv'

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file_path, index=False)

# Print a message indicating where the data has been saved
print(f"Data has been saved to {output_file_path}")
