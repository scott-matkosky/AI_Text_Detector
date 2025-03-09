from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model from the last checkpoint
MODEL_NAME = "./results/checkpoint-last"  # Change if needed

print("🔄 Loading trained model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Save the model for CLI use
SAVE_PATH = "./saved_model"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"✅ Model saved to `{SAVE_PATH}` for CLI usage.")