import pandas as pd
import re
from pathlib import Path

# Define paths for input and output datasets
DATA_PATH = Path(__file__).parent / "merged_train_essays.csv"
OUTPUT_PATH = Path(__file__).parent / "cleaned_train_essays.csv"

def clean_text(text):
    """Perform enhanced text cleaning:
       - Remove LaTeX commands (e.g., \textit{...})
       - Remove markdown symbols (e.g., #, *, `_`)
       - Remove URLs and email addresses
       - Normalize whitespace (remove excessive spaces/newlines)
    """
    # Remove LaTeX commands (e.g., \textit{word})
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)

    # Remove markdown headings and artifacts (e.g., #, **, *, `)
    text = re.sub(r'[#*_`]', '', text)

    # Remove URLs (http://, https://, www.)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Optionally, remove non-ASCII characters (uncomment if needed)
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Normalize spacing: collapse multiple spaces/newlines to a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally convert to lower case if your analysis is case-insensitive
    # text = text.lower()

    return text

def preprocess_dataset():
    """Load, clean, and save the dataset."""
    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    # Apply enhanced text cleaning to each essay
    df["text"] = df["text"].apply(clean_text)

    # Save the cleaned dataset to a new CSV file
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cleaned dataset saved: {len(df)} essays.")

if __name__ == "__main__":
    preprocess_dataset()