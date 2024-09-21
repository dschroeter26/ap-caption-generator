import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from pathlib import Path

# Define image folder path using pathlib
image_folder = Path('./../data/images/')

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# Function to preprocess images
def preprocess_image(image_path):
    image_path = Path(image_path)  # Ensure it's a Path object
    if image_path.exists():
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        return img
    else:
        print(f"Image not found: {image_path}")
        return None

# Function to generate caption using BLIP
def generate_caption(model, processor, input_text, image_path=None, max_length=50):
    # Preprocess the image
    if image_path:
        image = preprocess_image(image_path)
    else:
        image = None

    # Tokenize input text and preprocess image together
    inputs = processor(images=image, text=input_text, return_tensors="pt", padding=True)

    # Generate caption from the model
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# Example testing with new data
test_data = [
    {"text": "A U.S. Army soldier prepares for deployment.", "image_path": "./../data/images/8652204_360x255_q95.jpg"},
    {"text": "The Navy celebrates Fleet Week in New York City.", "image_path": "./../data/images/8652139_360x255_q95.jpg"}
]

# Generate captions for new data
for i, data in enumerate(test_data):
    print(f"Input {i + 1}: {data['text']}")
    generated_caption = generate_caption(model, processor, data['text'], data['image_path'])
    print(f"Generated Caption {i + 1}: {generated_caption}")
