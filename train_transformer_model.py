import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
from tqdm import tqdm

def load_data(data_path, sample_size=None):
    df = pd.read_csv(data_path)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df

def tokenize_function(examples, tokenizer, max_length=512):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["labels"] = examples["label"]
    return tokenized

def create_dataset_from_df(df, tokenizer):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="features_train_essays.csv")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    df = pd.read_csv(args.data_path)
    print(f"📊 Loaded dataset with {len(df)} rows.")

    df = df.rename(columns={"generated": "label"})

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n📌 Starting Fold {fold + 1}/{args.k_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = create_dataset_from_df(train_df, tokenizer)
        val_ds = create_dataset_from_df(val_df, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold+1}",
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=0.01,
            logging_dir=f"./logs/fold_{fold+1}",
            report_to=["none"]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=create_dataset_from_df(train_df, tokenizer),
            eval_dataset=create_dataset_from_df(val_df, tokenizer),
        )

        trainer.train()

        eval_results = trainer.evaluate()
        predictions = trainer.predict(create_dataset_from_df(val_df, tokenizer)).predictions
        predictions = predictions.argmax(axis=1)

        accuracy = accuracy_score(val_df["label"], predictions)
        report = classification_report(val_df["label"], predictions, digits=4)
        conf_matrix = confusion_matrix(val_df["label"], predictions)

        print(f"✅ Fold {fold+1} Results:")
        print(f"🔸 Accuracy: {accuracy:.4f}")
        print("🔸 Confusion Matrix:\n", conf_matrix)
        print("🔹 Classification Report:\n", report)

        # Save models after each fold
        fold_save_path = Path(f"./saved_model_fold_{fold + 1}")
        fold_save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fold_save_path)
        tokenizer.save_pretrained(fold_save_path)

        print(f"✅ Fold {fold + 1} model saved to {fold_save_path}")

    print("🚩 Cross-validation completed!")

if __name__ == "__main__":
    main()

'''
### Quick Summary of the Updated Script:
- **K-Fold Cross-Validation**: Clearly defined folds with metrics.
- **Model**: Fine-tuned DistilBERT for sequence classification.
- **Metrics**: Confusion matrix, accuracy, precision, recall, and F1-score for each fold.
- **Saving**: Models and tokenizer saved separately per fold for inspection and deployment.
- **Progress**: TQDM progress bar explicitly available through HuggingFace Trainer logs during training.

### Next Steps:
#Run the script with command line arguments like:


python train_transformer_model.py --data_path features_train_essays.csv --num_epochs 3 --batch_size 8 --k_folds 5
'''