import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load your captions data
df = pd.read_csv('./../data/captions.csv')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize text
def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)

# Tokenize captions
df['tokenized_caption'] = df['caption'].apply(tokenize_text)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the processed data
train_df.to_csv('./../data/train_data.csv', index=False)
val_df.to_csv('./../data/val_data.csv', index=False)

print("Data preparation complete.")
