import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class CaptionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_df = pd.read_csv('./../data/train_data.csv')
val_df = pd.read_csv('./../data/val_data.csv')

# Tokenize dataset and add labels for loss calculation
train_encodings = tokenizer(
    list(train_df['caption']),
    truncation=True,
    padding=True,
    max_length=512
)
train_encodings['labels'] = train_encodings['input_ids'].copy()

val_encodings = tokenizer(
    list(val_df['caption']),
    truncation=True,
    padding=True,
    max_length=512
)
val_encodings['labels'] = val_encodings['input_ids'].copy()

# Prepare datasets
train_dataset = CaptionDataset(train_encodings)
val_dataset = CaptionDataset(val_encodings)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./output')
tokenizer.save_pretrained('./output')
