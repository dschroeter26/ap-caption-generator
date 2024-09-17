import pandas as pd  # Import pandas for reading CSV
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch
import evaluate

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('./output')

# Load the validation data
val_df = pd.read_csv('./../data/val_data.csv')

# Load the metric
bleu = evaluate.load("bleu")

# Function to calculate BLEU score
def compute_bleu(preds, refs):
    return bleu.compute(predictions=preds, references=refs)

# Evaluate the model
predictions, references = [], []

for caption in val_df['caption']:
    inputs = tokenizer.encode(caption, return_tensors='pt')
    outputs = model.generate(inputs, max_length=512)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)
    references.append(caption)

# Calculate BLEU score
bleu_score = compute_bleu(predictions, references)
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
