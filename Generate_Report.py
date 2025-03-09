import torch
import argparse
import numpy as np
import PyPDF2
import nltk
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Ensure NLTK packages are downloaded
for pkg in ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'vader_lexicon']:
    nltk.download(pkg)

# Paths to the ensemble models
MODEL_DIRS = [f"./saved_model_fold_{i}" for i in range(1, 6)]
MODEL_NAME = "distilbert-base-uncased"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔹 Device: {device}")

# Load ensemble models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
models = [AutoModelForSequenceClassification.from_pretrained(path).to(device) for path in MODEL_DIRS]

# GPT-2 for perplexity
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

sia = SentimentIntensityAnalyzer()

def load_text(file_path):
    path = Path(file_path)
    if path.suffix in [".txt", ".tex"]:
        return path.read_text(encoding="utf-8")
    elif path.suffix == ".pdf":
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def compute_stats(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_words = len(words)

    pos_tags = nltk.pos_tag(words)
    stop_words = set(stopwords.words('english'))

    return {
        "avg_sentence_length": np.mean([len(word_tokenize(s)) for s in sentences]),
        "burstiness": np.std([len(word_tokenize(s)) for s in sentences]),
        "function_word_ratio": sum(w.lower() in stop_words for w in words) / num_words,
        "lexical_diversity": len(set(words)) / num_words,
        "readability": textstat.flesch_kincaid_grade(text),
        "noun_ratio": sum(1 for _, p in pos_tags if p.startswith("NN")) / num_words,
        "verb_ratio": sum(1 for _, p in pos_tags if p.startswith("VB")) / num_words,
        "adj_ratio": sum(1 for _, p in pos_tags if p.startswith("JJ")) / num_words,
    }

def perplexity(sentence):
    inputs = gpt2_tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs.input_ids).loss
    return torch.exp(loss).item()

def ensemble_predict(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = np.mean([model(**inputs).logits.softmax(dim=-1).cpu().numpy() for model in models], axis=0).flatten()
    label = "AI-Generated" if probs[1] > probs[0] else "Human-Written"
    return label, probs.max()

def sentence_analysis(text):
    analysis = []
    for sent in sent_tokenize(text):
        pred, conf = ensemble_predict(sent)
        analysis.append({
            "sentence": sent,
            "prediction": pred,
            "confidence": conf * 100,
            "perplexity": perplexity(sent),
            "sentiment": sia.polarity_scores(sent)["compound"]
        })
    return analysis

def overall_impression(overall_result, confidence, stats, sentences):
    counts = {"High":0, "Medium":0, "Low":0}
    for s in sentences:
        cat = "High" if s["confidence"]>70 else "Medium" if s["confidence"]>40 else "Low"
        counts[cat] += 1
    impression = (
        f"{overall_result} (Confidence: {confidence:.2f}%)\n"
        f"Sentence Breakdown: High:{counts['High']} Medium:{counts['Medium']} Low:{counts['Low']}\n"
        f"Avg Sentiment: {np.mean([s['sentiment'] for s in sentences]):.2f}, "
        f"Avg Perplexity: {np.mean([s['perplexity'] for s in sentences]):.2f}\n"
    )
    return impression

def format_report(overall, confidence, stats, sentences):
    report = f"AI Text Detector Report\n{'='*50}\n"
    report += f"Overall: {overall} ({confidence:.2f}%)\n\nStats:\n"
    for k,v in stats.items():
        report += f"  {k.replace('_',' ').title()}: {v:.2f}\n"
    report += f"\n{overall_impression(overall, confidence, stats, sentences)}\n"
    report += "Sentence Analysis:\n"
    for idx, s in enumerate(sentences, 1):
        report += (f"{idx}. {s['sentence']}\n   → {s['prediction']} "
                   f"({s['confidence']:.2f}%), Perplexity: {s['perplexity']:.2f}, "
                   f"Sentiment: {s['sentiment']:.2f}\n")
    return report

def create_pdf(report_text, output):
    c = canvas.Canvas(output, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 10)
    for line in report_text.split("\n"):
        text.textLine(line)
        if text.getY() < 40:
            c.drawText(text)
            c.showPage()
            text = c.beginText(40, 750)
            text.setFont("Helvetica", 10)
    c.drawText(text)
    c.save()
    print(f"Saved PDF as '{output}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Text Analysis PDF Report")
    parser.add_argument("input", help="File path or text input")
    parser.add_argument("--output", default="AI_Text_Detector_Report.pdf")
    args = parser.parse_args()

    input_path = Path(args.input)
    text = load_text(input_path) if input_path.exists() else args.input

    overall, conf = ensemble_predict(text)
    stats = compute_stats(text)
    sentences = sentence_analysis(text)
    report_text = format_report(overall, conf * 100, stats, sentences)

    print(report_text)
    create_pdf(report_text, args.output)