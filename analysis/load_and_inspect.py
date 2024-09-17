import pandas as pd

# Load the CSV file
captions_df = pd.read_csv('./../data/captions.csv')  # Adjust the path if needed

# Inspect the first few rows
print("First 5 rows of data:")
print(captions_df.head())

# Check for missing values
print("\nMissing values in each column:")
print(captions_df.isnull().sum())

# Calculate caption lengths and basic statistics
captions_df['caption_length'] = captions_df['caption'].apply(len)
print("\nAverage caption length:")
print(captions_df['caption_length'].mean())

# Save the processed data for further analysis if needed
captions_df.to_csv('./../data/captions_processed.csv', index=False)
