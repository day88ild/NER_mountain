# -*- coding: utf-8 -*-
"""fine_tuning_RoBERTa.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-_48WhiLZcTMQ2ZpmY2GBTiM34HG5qzN
"""

!pip install datasets peft

from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from transformers import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

class CustomRoBERTaDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, max_length):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

DATA_PATH = "/content/drive/MyDrive/data"
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 20
MAX_LENGTH = 32

# Check if a GPU is available and set the device accordingly

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Load the pretrained RoBERTa model and tokenizer

model_name = "distilroberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=2)

# Move model parameters to gpu if available

model.to(device)

# Configure LoRA

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,    # Task type: sequence classification
    inference_mode=False,          # Set to False for training
    r=8,                           # Rank for the LoRA matrices
    lora_alpha=16,                 # Scaling factor
    lora_dropout=0.5,              # Dropout probability
    target_modules="all-linear"
)

# Apply the LoRA configuration to the model

model = get_peft_model(model, lora_config)

test_dataset = torch.load(os.path.join(DATA_PATH, "processed_test_dataset.pt"))
train_dataset = torch.load(os.path.join(DATA_PATH, "processed_train_dataset.pt"))

test_dataload = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
train_dataload = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=LR)
loss_f = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 30]).to(torch.float64).to(device))

history_loss = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in tqdm(train_dataload):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device).to(torch.int64)

        # Forward pass
        outputs = model(input_ids=input_ids).logits.to(torch.float64)
        loss = loss_f(outputs.view(-1, 2), labels.view(-1))
        total_loss += loss.item()
        history_loss.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataload)

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

plt.figure(figsize=(20, 10))

plt.plot(history_loss)
plt.title("Loss per training step")
plt.show()

def acc(model, dataloader):
    model.eval()

    total_acc = 0

    for batch in iter(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device).to(torch.int64)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids)

        outputs = torch.softmax(outputs.logits, dim=2).argmax(dim=2)
        acc = (outputs == labels).to(float).mean()
        total_acc += acc

    total_acc /= len(dataloader)
    return total_acc

acc(model, train_dataload)

acc(model, test_dataload)

def avg_pos_true(model, dataloader):
    model.eval()

    total_pos_true = 0

    for batch in iter(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device).to(torch.int64)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids)

        outputs = torch.softmax(outputs.logits, dim=2).argmax(dim=2)[labels.bool()]
        pos_true = (outputs == 1).to(float).mean()
        total_pos_true += pos_true

    total_pos_true /= len(dataloader)
    return total_pos_true

avg_pos_true(model, train_dataload)

avg_pos_true(model, test_dataload)

torch.save(model.cpu(), os.path.join(DATA_PATH, "roberta_fine_tuned.pt"))

