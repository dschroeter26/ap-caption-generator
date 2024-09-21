from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load the custom BLIP processor and model from your trained model directory
processor = BlipProcessor.from_pretrained('./model/output')
model = BlipForConditionalGeneration.from_pretrained('./model/output')

app = Flask(__name__)

def generate_caption(image_path, metadata):
    # Load and process the image
    image = Image.open(image_path).convert('RGB')

    # Combine the metadata into a text format for the model
    metadata_text = (
        f"Date: {metadata['date']}. "
        f"City: {metadata['city']}, State: {metadata['state']}. "
        f"Photo Action: {metadata['photo_action']}, Context: {metadata['photo_context']}. "
        f"Photographer: {metadata['photographer_rank']} {metadata['photographer_name']} from {metadata['photographer_branch']}. "
        + " ".join([f"{member['rank']} {member['firstName']} {member['lastName']} from {member['unit']} ({member['serviceBranch']})"
                    for member in metadata['service_members']])
    )

    # Prepare inputs with both image and metadata text
    inputs = processor(images=image, text=metadata_text, return_tensors="pt")

    # Generate caption using the model
    generated_ids = model.generate(**inputs)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return caption

# Route to handle image and metadata
@app.route('/generate-caption', methods=['POST'])
def generate_caption_with_metadata():
    try:
        # Get image file
        image = request.files.get('image')
        if not image:
            return jsonify({"error": "No image file provided"}), 400
        
        filename = secure_filename(image.filename)
        image_path = os.path.join("/path/to/save", filename)
        image.save(image_path)
        
        # Get form data
        metadata = {
            'date': request.form.get('date'),
            'city': request.form.get('city'),
            'state': request.form.get('state'),
            'photo_action': request.form.get('photoAction'),
            'photo_context': request.form.get('photoContext'),
            'photographer_branch': request.form.get('photographersBranch'),
            'photographer_name': request.form.get('photographersName'),
            'photographer_rank': request.form.get('photographersRank'),
            'service_members': request.get_json().get('serviceMembers', [])
        }

        # Feed the image and metadata to the BLIP model
        caption = generate_caption(image_path, metadata)
        
        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
