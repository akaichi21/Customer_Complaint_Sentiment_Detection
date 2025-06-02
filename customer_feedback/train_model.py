import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from preprocess import preprocess_for_phobert
import joblib

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Load dataset
df = pd.read_csv("data/feedback_dataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Encode labels
label_encoder_complaint = LabelEncoder()
df["complaint_label"] = label_encoder_complaint.fit_transform(df["complaint_type"])
joblib.dump(label_encoder_complaint, "models/complaint_model/label_encoder_complaint.pkl")

label_encoder_sentiment = LabelEncoder()
df["sentiment_label"] = label_encoder_sentiment.fit_transform(df["sentiment"])
joblib.dump(label_encoder_sentiment, "models/sentiment_model/label_encoder_sentiment.pkl")

print("Saving Label Encoder Successfully !")

# Split train test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["complaint_label"])

train_text = train_df["text"]
test_text = test_df["text"]
train_complaint = train_df["complaint_label"]
test_complaint = test_df["complaint_label"]
train_sentiment = train_df["sentiment_label"]
test_sentiment = test_df["sentiment_label"]

# Dataset class function
class CustomerFeedbackMultiDataset(Dataset):
    def __init__(self, text, complaint, sentiment):
        self.text = text.reset_index(drop=True)
        self.complaint = complaint.reset_index(drop=True)
        self.sentiment = sentiment.reset_index(drop=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        encoding = preprocess_for_phobert(self.text[index])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "complaint_label": torch.tensor(self.complaint[index]),
            "sentiment_label": torch.tensor(self.sentiment[index])
        }

# Create dataloader
train_dataset = CustomerFeedbackMultiDataset(train_text, train_complaint, train_sentiment)
test_dataset = CustomerFeedbackMultiDataset(test_text, test_complaint, test_sentiment)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Create models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

complaint_model = BertForSequenceClassification.from_pretrained(
    "vinai/phobert-base", num_labels=len(label_encoder_complaint.classes_)).to(device)

sentiment_model = BertForSequenceClassification.from_pretrained(
    "vinai/phobert-base", num_labels=len(label_encoder_sentiment.classes_)).to(device)

optimizer_complaint = AdamW(complaint_model.parameters(), lr=2e-5)
optimizer_sentiment = AdamW(sentiment_model.parameters(), lr=2e-5)

# Train models
EPOCHS = 10
for epoch in range(EPOCHS):
    complaint_model.train()
    sentiment_model.train()
    total_loss_complaint = 0
    total_loss_sentiment = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Train complaint
        label_complaint = batch["complaint_label"].to(device)
        output_complaint = complaint_model(input_ids=input_ids, attention_mask=attention_mask, labels=label_complaint)
        loss_complaint = output_complaint.loss
        total_loss_complaint += loss_complaint.item()
        loss_complaint.backward()
        optimizer_complaint.step()
        optimizer_complaint.zero_grad()

        # Train sentiment
        label_sentiment = batch["sentiment_label"].to(device)
        output_sentiment = sentiment_model(input_ids=input_ids, attention_mask=attention_mask, labels=label_sentiment)
        loss_sentiment = output_sentiment.loss
        total_loss_sentiment += loss_sentiment.item()
        loss_sentiment.backward()
        optimizer_sentiment.step()
        optimizer_sentiment.zero_grad()

    print(f"Epoch {epoch + 1} | Complaint Loss: {total_loss_complaint:.4f} | Sentiment Loss: {total_loss_sentiment:.4f}")

# Save models
os.makedirs("models/complaint_model", exist_ok=True)
os.makedirs("models/sentiment_model", exist_ok=True)
os.makedirs("models/tokenizer", exist_ok=True)

complaint_model.save_pretrained("models/complaint_model")
sentiment_model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/tokenizer")
print("Saving Models & Tokenizer Successfully !")