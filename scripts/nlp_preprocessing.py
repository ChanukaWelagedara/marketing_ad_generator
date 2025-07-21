import pandas as pd
import spacy
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Text preprocessing function
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# 1. Load dataset
input_path = os.path.join("data", "amazon_ads_dataset.csv")
df = pd.read_csv(input_path)

# 2. Apply preprocessing to text fields
print("Preprocessing 'features' and 'ad_text' columns...")
df["clean_features"] = df["features"].apply(preprocess_text)
df["clean_ad_text"] = df["ad_text"].apply(preprocess_text)

# 3. (Optional) Save just the cleaned fields for training
output_path = os.path.join("data", "cleaned_amazon_ads.csv")
df.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to '{output_path}'")
