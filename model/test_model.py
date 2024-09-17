import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./../output')
model = GPT2LMHeadModel.from_pretrained('./../output')

def generate_caption(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Example testing with new data
test_data = [
    "A U.S. Army soldier prepares for deployment.",
    "The Navy celebrates Fleet Week in New York City."
]

# Generate captions for new data
for i, text in enumerate(test_data):
    print(f"Input {i + 1}: {text}")
    generated_caption = generate_caption(model, tokenizer, text)
    print(f"Generated Caption {i + 1}: {generated_caption}")
