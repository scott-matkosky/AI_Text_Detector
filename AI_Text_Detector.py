import torch
import argparse
import numpy as np
import PyPDF2
import nltk
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import textstat

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the trained model from the saved folder
MODEL_PATH = "./saved_model"
print("🔄 Loading AI Text Detector...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Use Mac GPU (MPS) if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"🚀 Using device: {device}")

def compute_stats(text):
    """Compute advanced linguistic markers for AI-generated text detection."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_words = len(words)

    avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
    burstiness = np.std([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
    function_words = set(["the", "is", "in", "and", "to", "a", "of", "that", "it", "on", "for", "with"])
    function_word_ratio = sum(1 for word in words if word.lower() in function_words) / num_words if num_words else 0
    edit_distance = sum(len(word) for word in words) / num_words if num_words else 0
    vectorizer = CountVectorizer().fit_transform([" ".join(words)])
    cosine_sim = cosine(vectorizer.toarray()[0], vectorizer.toarray()[0]) if vectorizer.shape[0] > 1 else 0
    stop_words = set(stopwords.words("english"))
    stopword_ratio = sum(1 for word in words if word.lower() in stop_words) / num_words if num_words else 0
    lexical_diversity = len(set(words)) / num_words if num_words else 0
    readability = textstat.flesch_kincaid_grade(text)
    pos_tags = nltk.pos_tag(words)
    noun_ratio = sum(1 for _, pos in pos_tags if pos.startswith("NN")) / num_words if num_words else 0
    verb_ratio = sum(1 for _, pos in pos_tags if pos.startswith("VB")) / num_words if num_words else 0
    adj_ratio = sum(1 for _, pos in pos_tags if pos.startswith("JJ")) / num_words if num_words else 0

    return {
        "avg_sentence_length": avg_sentence_length,
        "burstiness": burstiness,
        "function_word_ratio": function_word_ratio,
        "edit_distance": edit_distance,
        "cosine_similarity": cosine_sim,
        "stopword_ratio": stopword_ratio,
        "lexical_diversity": lexical_diversity,
        "readability": readability,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio
    }

def format_output(result, confidence, stats):
    """Format output for clear readability in CLI."""
    formatted_output = f"""
📝 **AI Text Detector Results**
──────────────────────────────────────
🔹 **Prediction:** {result} (Confidence: {confidence:.2f}%)

📊 **Linguistic Stats**
──────────────────────────────────────
🔹 **Sentence Structure**
   - **Avg Sentence Length:** {stats['avg_sentence_length']:.2f} words
   - **Burstiness (Variance):** {stats['burstiness']:.2f} (Higher = More Human-like)
   - **Edit Distance Approximation:** {stats['edit_distance']:.2f}

🔹 **Word Usage & Diversity**
   - **Function Word Ratio:** {stats['function_word_ratio']:.2f} (Higher = More AI-like)
   - **Stopword Ratio:** {stats['stopword_ratio']:.2f}
   - **Lexical Diversity:** {stats['lexical_diversity']:.2f} (Higher = More Human-like)

🔹 **AI Similarity & Readability**
   - **Cosine Similarity to AI Datasets:** {stats['cosine_similarity']:.2f} (Higher = More AI-like)
   - **Readability Score (Flesch-Kincaid):** {stats['readability']:.2f} (Lower = More AI-like)

🔹 **Part-of-Speech Ratios**
   - **Noun Ratio:** {stats['noun_ratio']:.2f}
   - **Verb Ratio:** {stats['verb_ratio']:.2f}
   - **Adjective Ratio:** {stats['adj_ratio']:.2f}

──────────────────────────────────────
"""
    return formatted_output

def predict(text, explain=False):
    """Analyze text, predict AI vs. Human, and output linguistic stats."""
    # Tokenize text for model prediction (using truncation)
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    ai_prob = probs[0][1]
    human_prob = probs[0][0]
    result = "🟢 Human-Written" if human_prob > ai_prob else "🔴 AI-Generated"
    confidence = max(ai_prob, human_prob) * 100

    # Compute linguistic stats on the full text
    stats = compute_stats(text)

    return format_output(result, confidence, stats)

def read_file(file_path):
    """Read text from .txt, .tex, or .pdf files."""
    file_path = Path(file_path)
    if file_path.suffix in [".txt", ".tex"]:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_path.suffix == ".pdf":
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise ValueError("❌ Unsupported file format. Please provide a .txt, .tex, or .pdf file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect AI-generated text with advanced linguistic analysis."
    )
    parser.add_argument("input", type=str, help="Text or file path to analyze (use quotes for multi-word input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        print(f"📂 Processing file: {args.input}")
        text = read_file(args.input)
    else:
        text = args.input

    print(predict(text, explain=False))