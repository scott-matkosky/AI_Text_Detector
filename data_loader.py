import json
import pandas as pd
from pathlib import Path

# Define directories for data
L2R_DIR = Path(__file__).parent / "L2R_dataset"
ARCHIVE_DIR = Path(__file__).parent / "archive"
KAGGLE_PATH = Path(__file__).parent / "llm-detect-ai-generated-text" / "train_essays.csv"

# List of filenames that indicate AI-generated texts (modify if needed)
AI_FILENAMES = {"GPT-4o.json", "Gemini-1.5-Pro.json", "Llama-3-70B.json", "GPT-3-Turbo.json"}
HUMAN_FILENAME = "human.json"

def load_json(file_path):
    """Load a JSON file and return its content as a list of texts."""
    if not file_path.exists():
        print(f"⚠️ Skipping missing file: {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Assume the JSON is a list of text samples
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def load_l2r_data():
    """Recursively load all JSON files in the L2R_dataset and assign labels."""
    data_samples = []
    # Use rglob to find all JSON files under L2R_DIR
    for json_file in L2R_DIR.rglob("*.json"):
        # Determine label based on file name
        if json_file.name == HUMAN_FILENAME:
            label = 0  # human
        elif json_file.name in AI_FILENAMES:
            label = 1  # ai
        else:
            # If filename is not recognized, skip it.
            print(f"⚠️ Unrecognized JSON file (skipping): {json_file}")
            continue

        texts = load_json(json_file)
        # Append each text along with its label
        for text in texts:
            data_samples.append((text, label))
    return pd.DataFrame(data_samples, columns=["text", "generated"])

def load_archive_data():
    """Load additional data from the archive directory (e.g. AIDE_train_essays.csv)."""
    archive_samples = pd.DataFrame()
    # Check for a CSV file named AIDE_train_essays.csv (adjust filename as needed)
    aide_csv = ARCHIVE_DIR / "AIDE_train_essays.csv"
    if aide_csv.exists():
        archive_samples = pd.read_csv(aide_csv)
        # Ensure that the label column is standardized (assume "generated" column with human=0, ai=1)
        if "generated" not in archive_samples.columns and "label" in archive_samples.columns:
            archive_samples = archive_samples.rename(columns={"label": "generated"})
    else:
        print(f"⚠️ AIDE_train_essays.csv not found in {ARCHIVE_DIR}")
    return archive_samples

def merge_datasets():
    """Merge L2R data, Archive data, and Kaggle essays."""
    print("Loading L2R dataset...")
    l2r_df = load_l2r_data()
    print(f"✅ Loaded {len(l2r_df)} essays from L2R_dataset.")

    print("Loading Archive dataset...")
    archive_df = load_archive_data()
    print(f"✅ Loaded {len(archive_df)} essays from Archive.")

    print("Loading Kaggle dataset...")
    if KAGGLE_PATH.exists():
        kaggle_df = pd.read_csv(KAGGLE_PATH)
        # Standardize column name if needed (assuming 'generated' or 'label')
        if "label" in kaggle_df.columns and "generated" not in kaggle_df.columns:
            kaggle_df = kaggle_df.rename(columns={"label": "generated"})
        print(f"✅ Loaded {len(kaggle_df)} essays from Kaggle dataset.")
    else:
        print(f"⚠️ Kaggle file not found: {KAGGLE_PATH}")
        kaggle_df = pd.DataFrame(columns=["text", "generated"])

    # Merge all datasets
    merged_df = pd.concat([kaggle_df, l2r_df, archive_df], ignore_index=True)
    OUTPUT_PATH = Path(__file__).parent / "merged_train_essays.csv"
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Merged dataset: {len(merged_df)} essays (AI & human) saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_datasets()