import re
import unicodedata
from underthesea import word_tokenize
from transformers import AutoTokenizer
import pandas as pd

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Preprocess
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Delete emoji
    emoji_patterns = re.compile("["
                                u"\U0001F600-\U0001F64F"  # Emotion Icons
                                u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                                u"\U0001F1E0-\U0001F1FF"  # Flags
                                "]+", flags=re.UNICODE)
    text = emoji_patterns.sub(r'', text)

    # Delete special characters and number
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Delete extra space
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize_vietnamese(text):
    return word_tokenize(text, format="text")

def preprocess_for_phobert(text):
    # 1. Clean text
    cleaned_text = clean_text(text)

    # 2. Tokenize vietnamese words
    tokenized_text = tokenize_vietnamese(cleaned_text)

    # 3. Tokenize PhoBERT
    encoding = tokenizer(
            tokenized_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    return encoding

# Get Data & Information
df = pd.read_csv("data/feedback_dataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

encoding_sample = preprocess_for_phobert(df.loc[0, "text"])

print(encoding_sample["input_ids"])
print(encoding_sample["attention_mask"])