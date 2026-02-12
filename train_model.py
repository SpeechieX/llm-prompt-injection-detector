import pandas as pd
from datasets import Dataset
from transformers import (
  AutoTokenizer,
  AutoModelForSequenceClassification,
  TrainingArguments,
  Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

#Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Convert to HF Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
 return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Metrics
def compute_metrics(eval_pred):
 logits, labels = eval_pred
 predictions = logits.argmax(axis=-1)
 precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
 acc = accuracy_score(labels, predictions)
 return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training config
args = TrainingArguments(
  output_dir="./model",
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  load_best_model_at_end=True,
  logging_dir="./logs",
)

trainer = Trainer(
  model=model,
  args=args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./model/final")
tokenizer.save_pretrained("./model/final")
print("Model Training Complete!")
