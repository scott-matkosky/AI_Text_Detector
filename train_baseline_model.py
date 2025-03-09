import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

DATA_PATH = Path(__file__).parent / "features_train_essays.csv"


def load_data():
    """ Load dataset and prepare features/labels. """
    df = pd.read_csv(DATA_PATH)

    # Select features (excluding 'text' column)
    feature_cols = ["perplexity", "burstiness", "function_word_ratio", "levenshtein_distance", "cosine_similarity"]
    X = df[feature_cols]
    y = df["generated"]  # Target variable (AI = 1, Human = 0)

    return X, y


def train_models():
    """ Train multiple classifiers and evaluate performance. """
    X, y = load_data()

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n🔍 {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_models()