import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

DATA_PATH = Path(__file__).parent / "features_train_essays.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select features
feature_cols = ["perplexity", "burstiness", "function_word_ratio", "levenshtein_distance", "cosine_similarity"]
X = df[feature_cols]
y = df["label"]  # AI = 1, Human = 0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print results
print(f"\n🔍 Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))