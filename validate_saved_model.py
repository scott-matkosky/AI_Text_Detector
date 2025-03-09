import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# ✅ Load the trained model from `saved_model/`
MODEL_PATH = Path("./saved_model")
DATA_PATH = Path("./cleaned_train_essays.csv")

print("🔄 Loading saved model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ✅ Use Mac GPU (MPS) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"🚀 Using device: {device}")

# ✅ Load dataset
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Use the same test split as before
    _, test_texts, _, test_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )

    return test_texts, test_labels

# ✅ Tokenize dataset
def tokenize_function(examples):
    """ Ensure tokenization works correctly for single and batch inputs """
    if isinstance(examples["text"], list):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    else:
        return tokenizer([examples["text"]], truncation=True, padding="max_length", max_length=512)

# ✅ Prepare dataset for evaluation
def create_dataset():
    test_texts, test_labels = load_data()
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    return test_dataset, test_labels

# ✅ Compute accuracy manually
def compute_metrics(predictions, labels):
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, digits=4)
    return accuracy, report

# ✅ Run evaluation on test set
def evaluate_model():
    print("🔍 Evaluating model on test dataset...")
    test_dataset, test_labels = create_dataset()

    trainer = Trainer(model=model)
    raw_preds = trainer.predict(test_dataset)
    predictions = raw_preds.predictions

    # Compute accuracy & classification metrics
    accuracy, report = compute_metrics(predictions, test_labels)

    print("\n📊 **Model Evaluation Results:**")
    print(f"🔹 Validation Loss: {raw_preds.metrics['test_loss']:.4f}")
    print(f"🔹 Validation Accuracy: {accuracy:.4f}")
    print(f"\n🔹 Classification Report:\n{report}")

if __name__ == "__main__":
    evaluate_model()