import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import evaluate

# Define image folder path
image_folder = './../data/images/'

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# Load the validation data
val_df = pd.read_csv('./../data/val_data.csv')

# Load the BLEU metric
bleu = evaluate.load("bleu")

# Function to load and preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    return image

# Function to calculate BLEU score
def compute_bleu(preds, refs):
    return bleu.compute(predictions=preds, references=refs)

# Evaluate the model
predictions, references = [], []

for idx, row in val_df.iterrows():
    caption = row['caption']
    image_path = row['image_path']
    
    # Process image
    full_image_path = os.path.join(image_folder, image_path)
    if os.path.exists(full_image_path):
        image = preprocess_image(full_image_path)
    else:
        print(f"Image not found: {full_image_path}")
        continue  # Skip this entry if the image is missing
    
    # Tokenize input text and image together using the processor
    inputs = processor(images=image, text=caption, return_tensors="pt")

    # Generate caption from the model
    outputs = model.generate(**inputs, max_length=50)
    pred = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Append predictions and references (ensure they are in the correct format)
    predictions.append(pred)
    references.append([caption])  # References need to be a list of lists for BLEU score

# Check lengths of predictions and references
if len(predictions) == 0 or len(references) == 0:
    print("Error: No predictions or references generated. Check your data.")
elif len(predictions) != len(references):
    print(f"Error: Mismatch in lengths - Predictions: {len(predictions)}, References: {len(references)}")
else:
    # Calculate BLEU score
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
