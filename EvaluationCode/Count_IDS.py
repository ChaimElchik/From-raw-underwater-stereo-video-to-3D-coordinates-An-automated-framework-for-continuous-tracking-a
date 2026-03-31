import pandas as pd

# Load the file into a pandas DataFrame
# Ensure '129_1_clean.txt' is in the same directory as your script
df = pd.read_csv('mots/406_1_clean.txt')

# Count the number of unique values in the 'id' column
num_unique_ids = df['id'].nunique()

# Get the actual list of unique IDs (sorted)
unique_ids_list = sorted(df['id'].unique())

# Print the results
print(f"Number of unique IDs: {num_unique_ids}")
print(f"List of Unique IDs: {unique_ids_list}")