import pandas as pd
import spacy
import os

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the processed data, including image paths
captions_df = pd.read_csv('./../data/captions_processed.csv')

# Ensure the 'image_path' column exists and load images if needed (optional for now)
if 'image_path' not in captions_df.columns:
    raise KeyError("The 'image_path' column is missing from captions.csv. Make sure your scraping script saves image paths correctly.")

# Extract named entities from captions
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply to all captions
captions_df['entities'] = captions_df['caption'].apply(extract_entities)

# Show some examples
print("Extracted Named Entities:")
print(captions_df[['caption', 'entities']].head())

# Save the data with extracted entities
captions_df.to_csv('./../data/captions_with_entities.csv', index=False)

print("Entity extraction complete.")
