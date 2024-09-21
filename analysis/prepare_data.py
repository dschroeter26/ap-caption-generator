import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# Define paths using pathlib
captions_file = Path('./../data/captions.csv')  # Path to your captions.csv file

# Load your captions and image paths data
df = pd.read_csv(captions_file)

# Initialize the BLIP processor (handles both tokenization and image preprocessing)
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

# Function to preprocess images (using BLIP's processor)
def preprocess_image(image_name):
    image_path = Path(image_name)  # Convert image_name to Path object
    try:
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        return img
    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process both text and images
def process_row(row):
    image = preprocess_image(row['image_path']) if pd.notna(row['image_path']) else None
    caption = row['caption']
    
    if image is not None:
        inputs = processor(images=image, text=caption, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs
    return None

# Apply processing to each row in the dataframe
df['processed_data'] = df.apply(process_row, axis=1)

# Check if 'processed_data' column was created successfully
if 'processed_data' in df.columns:
    # Filter out rows where processed_data is None
    df = df[df['processed_data'].notnull()]

    # Split the data into training and validation sets (captions and images)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the processed data
    train_df.to_csv('./../data/train_data.csv', index=False)
    val_df.to_csv('./../data/val_data.csv', index=False)

    print("Data preparation with BLIP processor and images complete.")
else:
    print("Error: 'processed_data' column not created.")
