import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk
import PyPDF2
import docx

# Setup device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔹 Device: {device}")

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIRS = [f"./saved_model_fold_{i}" for i in range(1, 6)]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load all 5 fine-tuned models
models = [AutoModelForSequenceClassification.from_pretrained(path).to(device) for path in MODEL_DIRS]

def load_file(file_path):
    """Load text content from supported file formats."""
    file_path = Path(file_path)
    if file_path.suffix in [".txt", ".tex"]:
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix == ".pdf":
        with open(file_path, "rb") as f:
            import PyPDF2
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages)
    elif file_path.suffix == ".docx":
        import docx
        doc = docx.Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def ensemble_predict(text):
    """Predict text classification using ensemble soft voting."""
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = [model(**inputs).logits.softmax(dim=-1).cpu().numpy() for model in models]

    # Average probabilities across models
    avg_probs = np.mean(logits, axis=0).flatten()
    pred_label = np.argmax(avg_probs)
    confidence = avg_probs.max()

    return ("AI-Generated" if avg_probs[1] > avg_probs[0] else "Human-Written", confidence)

def analyze_text(text):
    """Analyze text and return prediction with ensemble confidence."""
    sentences = sent_tokenize(text)
    sentence_analysis = []

    for sentence in sentences:
        result, confidence = ensemble_predict(sentence)
        sentence_analysis.append({"sentence": sentence, "prediction": result, "confidence": confidence})

    overall_result, overall_confidence = ensemble_predict(text)

    return overall_result, overall_confidence, sentence_analysis

def format_output(overall_result, overall_confidence, sentence_analysis):
    print("\n📝 AI Text Detector Ensemble Results")
    print("──────────────────────────────────────")
    print(f"Overall Prediction: {'🔴 AI-Generated' if overall_result == 1 else '🟢 Human-Written'} ({overall_confidence * 100:.2f}%)")
    print("\nSentence-level Analysis:\n" + "-" * 40)
    for idx, sa in enumerate(sentence_analysis, 1):
        print(f"{idx}. {sa['sentence']}\n   → {sa['prediction']} ({sa['confidence'] * 100:.2f}%)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble AI Text Detector (DistilBERT)")
    parser.add_argument("input", type=str, help="Input text or path to text file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        text = load_file(input_path)
    else:
        text = args.input

    overall_result, overall_confidence, sentence_analysis = analyze_text(text)

    format_output(overall_result, overall_confidence, sentence_analysis)