import pandas as pd
import numpy as np
import time
import torch
import wandb
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
import os
import json
import joblib

# Initialize Weights & Biases
wandb.init(project="gpt2-text-classification", name="run_gpt2")

# Ensure results and logs directories exist
os.makedirs('./results', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

# Load and preprocess dataset
df = pd.read_csv("unified_processed_final_dataset.csv")
df = df[['summary_processed', 'label']].dropna()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
joblib.dump(le, './results/label_encoder.pkl')

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=len(set(df['label'])))
model.config.pad_token_id = tokenizer.pad_token_id

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example['summary_processed'],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=2)
test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=2)

# Format datasets for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    probs = softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': recall,  # Same as recall here
        'specificity': recall,  # Placeholder if you want class-wise specificity
        'roc_auc': roc_auc_score(labels, probs, multi_class='ovr') if len(set(labels)) > 2 else roc_auc_score(labels, probs[:, 1])
    }

    wandb.log(metrics)
    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=200,
    per_device_train_batch_size=4,  # Smaller batch for GPT2
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    report_to="wandb"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Training
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

# Save model
trainer.save_model('./results/best_model_gpt2')
trainer.save_state()

# Save training log
log_history = trainer.state.log_history
np.save('./results/log_history_gpt2.npy', log_history)
with open('./results/log_history_gpt2.txt', 'w') as f:
    for log in log_history:
        f.write(json.dumps(log) + '\n')

# Inference
start_infer = time.time()
predictions = trainer.predict(test_dataset)
inference_time = time.time() - start_infer

# Save predictions and labels
np.save('./results/predictions_gpt2.npy', predictions.predictions)
np.save('./results/labels_gpt2.npy', predictions.label_ids)

# Final metrics
probs = softmax(predictions.predictions, axis=1)
predicted_classes = np.argmax(predictions.predictions, axis=1)
metrics = {
    'accuracy': accuracy_score(predictions.label_ids, predicted_classes),
    'precision': precision_score(predictions.label_ids, predicted_classes, average='weighted'),
    'recall': recall_score(predictions.label_ids, predicted_classes, average='weighted'),
    'f1': f1_score(predictions.label_ids, predicted_classes, average='weighted'),
    'roc_auc': roc_auc_score(predictions.label_ids, probs, multi_class='ovr') if model.config.num_labels > 2 else roc_auc_score(predictions.label_ids, probs[:, 1])
}

with open('./results/metrics_gpt2.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save times
with open('./results/times_gpt2.txt', 'w') as f:
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Inference time: {inference_time:.2f} seconds\n")

# Log to wandb
wandb.log({"training_time_sec": training_time, "inference_time_sec": inference_time})
wandb.finish()

print(f"Training time: {training_time:.2f} sec")
print(f"Inference time: {inference_time:.2f} sec")
