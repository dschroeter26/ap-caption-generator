from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./model/output')
model = GPT2LMHeadModel.from_pretrained('./model/output')

app = Flask(__name__)

# Function to generate caption based on input text
def generate_caption(input_text, max_length=50):
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Route to handle image and metadata
@app.route('/generate-caption', methods=['POST'])
def generate_caption_with_metadata():
    try:
        # Parse form data (multipart/form-data)
        image = request.files.get('image')  # Access the image file
        metadata = request.form  # Access form data (contains other fields)

        if not image or not metadata:
            return jsonify({'error': 'Image or metadata not provided'}), 400

        # Extract recognized people, action, context, etc.
        recognized_people = metadata.getlist('recognized_people')
        action = metadata.get('action')
        context = metadata.get('context')
        city = metadata.get('city')
        state = metadata.get('state')
        date = metadata.get('date')

        # Combine input information to form the caption input text
        people_info = "; ".join(recognized_people) if recognized_people else "Unknown people"
        input_text = f"In {city}, {state}, on {date}, {people_info} were involved in {action}. Context: {context}."

        # Generate the caption
        caption = generate_caption(input_text)

        # Return the generated caption as JSON
        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
