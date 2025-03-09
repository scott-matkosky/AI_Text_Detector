import pandas as pd
import numpy as np
import re
import nltk
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import textstat
from tqdm import tqdm

# Download necessary NLTK packages (if not already downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# Define input and output paths
DATA_PATH = Path(__file__).parent / "cleaned_train_essays.csv"
OUTPUT_PATH = Path(__file__).parent / "features_train_essays.csv"

# Initialize the progress bar for pandas
tqdm.pandas()

# Load a pre-trained language model for perplexity calculation (using GPT-2)
MODEL_NAME = "gpt2"
perplexity_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
perplexity_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# GPT-2 doesn't have a pad token by default; add one for padding purposes.
if perplexity_tokenizer.pad_token is None:
    perplexity_tokenizer.add_special_tokens({'pad_token': perplexity_tokenizer.eos_token})
    perplexity_model.resize_token_embeddings(len(perplexity_tokenizer))

# Set device (MPS for macOS if available)
device = "mps" if torch.backends.mps.is_available() else "cpu"
perplexity_model.to(device)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize TF-IDF vectorizer for cosine similarity
vectorizer = TfidfVectorizer()


### Feature 1: Perplexity (Lower perplexity = more predictable/AI-like) ###
def compute_perplexity(text):
    # Truncate/pad to 256 tokens for efficiency
    inputs = perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return np.exp(loss)


### Feature 2: Burstiness (Standard deviation of sentence lengths) ###
def compute_burstiness(text):
    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    return np.std(sent_lengths) if sent_lengths else 0


### Feature 3: Function Word Ratio ###
FUNCTION_WORDS = {"the", "is", "in", "and", "to", "a", "of", "that", "it", "on", "for", "with"}
def function_word_ratio(text):
    words = word_tokenize(text.lower())
    if not words:
        return 0
    return sum(1 for word in words if word in FUNCTION_WORDS) / len(words)


### Feature 4: Levenshtein Edit Distance (Placeholder) ###
# (Requires a rewritten version of the text; here we set a placeholder value)
def levenshtein_distance(text1, text2):
    return fuzz.ratio(text1, text2) / 100  # Normalized between 0 and 1


### Feature 5: Cosine Similarity (Placeholder) ###
# (Requires two versions of text; here we set a placeholder value)
def cosine_similarity(text1, text2):
    vectors = vectorizer.fit_transform([text1, text2])
    return (vectors * vectors.T).A[0, 1]


### Feature 6: Sentiment Score (Compound score using VADER) ###
def sentiment_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment["compound"]


### Feature 7: Readability (Flesch Reading Ease) ###
def readability_score(text):
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return np.nan


### Feature 8: POS Ratios (Noun, Verb, Adjective ratios) ###
def pos_ratios(text):
    words = word_tokenize(text)
    if not words:
        return 0, 0, 0
    pos_tags = nltk.pos_tag(words)
    total = len(words)
    noun_count = sum(1 for word, pos in pos_tags if pos.startswith("NN"))
    verb_count = sum(1 for word, pos in pos_tags if pos.startswith("VB"))
    adj_count = sum(1 for word, pos in pos_tags if pos.startswith("JJ"))
    return noun_count / total, verb_count / total, adj_count / total


### Feature 9: Average Sentence Length ###
def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    lengths = [len(word_tokenize(s)) for s in sentences]
    return np.mean(lengths)


### Main Feature Extraction Function ###
def extract_features():
    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset with {len(df)} essays.")

    # Compute features with progress bars
    tqdm.pandas(desc="Perplexity")
    df["perplexity"] = df["text"].progress_apply(compute_perplexity)

    tqdm.pandas(desc="Burstiness")
    df["burstiness"] = df["text"].progress_apply(compute_burstiness)

    tqdm.pandas(desc="Function Word Ratio")
    df["function_word_ratio"] = df["text"].progress_apply(function_word_ratio)

    # Placeholders: you can later compute these if you have a rewritten version
    df["levenshtein_distance"] = 0.0
    df["cosine_similarity"] = 0.0

    tqdm.pandas(desc="Sentiment")
    df["sentiment"] = df["text"].progress_apply(sentiment_score)

    tqdm.pandas(desc="Readability")
    df["readability"] = df["text"].progress_apply(readability_score)

    tqdm.pandas(desc="POS Ratios")
    pos_df = df["text"].progress_apply(lambda x: pd.Series(pos_ratios(x), index=["noun_ratio", "verb_ratio", "adj_ratio"]))
    df = pd.concat([df, pos_df], axis=1)

    tqdm.pandas(desc="Avg Sentence Length")
    df["avg_sentence_length"] = df["text"].progress_apply(avg_sentence_length)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Extracted features for {len(df)} essays saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_features()