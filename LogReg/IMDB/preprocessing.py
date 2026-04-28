import re
import os
import json
import random
import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets using: pip install datasets")
    exit(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

SEED=42
set_seed(SEED)

DATASET = "IMDB"

def fix_spaced_contractions(text: str) -> str:
    text = re.sub(r"\b(\w+)\s+'\s*s\b", r"\1's", text)
    text = re.sub(r"\b(\w+)\s+'\s*re\b", r"\1're", text)
    text = re.sub(r"\b(\w+)\s+'\s*ve\b", r"\1've", text)
    text = re.sub(r"\b(\w+)\s+'\s*ll\b", r"\1'll", text)
    text = re.sub(r"\b(\w+)\s+'\s*d\b", r"\1'd", text)
    text = re.sub(r"\b(\w+)\s+'\s*m\b", r"\1'm", text)
    text = re.sub(r"\b(\w+)\s+n\s*'\s*t\b", r"\1n't", text)
    text = re.sub(r"\bca\s+n\s*'\s*t\b", "can't", text)
    text = re.sub(r"\bwo\s+n\s*'\s*t\b", "won't", text)
    text = re.sub(r"\bsha\s+n\s*'\s*t\b", "shan't", text)
    text = re.sub(r"\b(\w+)\s+'\s*em\b", r"\1 'em", text)
    return text

CONTRACTION_MAP = {
    "'em": "them", "'ve": " have", "aren't": "are not", "can't": "can not",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what's": "what is",
    "where's": "where is", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have",
}

_contraction_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTION_MAP, key=len, reverse=True)) + r")\b"
)

def expand_contractions(text: str) -> str:
    return _contraction_pattern.sub(lambda m: CONTRACTION_MAP[m.group(0)], text)

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text) # Remove HTML tags specifically for IMDB
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = fix_spaced_contractions(text)
    text = expand_contractions(text)
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    text = re.sub(r"\.{2,}|--|/|;|[()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if __name__ == "__main__":
    print(f"Loading {DATASET} dataset...")
    dataset = load_dataset("imdb")

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        sentences = split_data["text"]
        labels    = split_data["label"]

        records = [
            {
                "sentence_original":     sent,
                "sentence_preprocessed": preprocess(sent),
                "label":                 label,
            }
            for sent, label in zip(sentences, labels)
        ]

        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"{split_name:>10}: {len(records):>6} samples -> {path}")

    print("\nDone! Preprocessing for IMDB dataset is complete.")
