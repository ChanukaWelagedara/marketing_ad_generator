import pandas as pd
import random
import os
import spacy

# Step 0: Setup
AGE_RANGE = list(range(18, 65))
GENDERS = ["Male", "Female"]

INTEREST_MAP = {
    "Arts & Crafts": "art, DIY",
    "Electronics": "tech, gadgets",
    "Books": "reading, literature",
    "Toys & Games": "gaming, kids",
    "Beauty": "fashion, skincare",
    "Sports": "fitness, outdoors",
    "Home & Kitchen": "home decor, cooking",
    "default": "shopping, deals"
}

# Load spaCy English model (small model for speed)
nlp = spacy.load("en_core_web_sm")

# Step 1: Load dataset
from datasets import load_dataset
print("Loading dataset...") 
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm")
df = pd.DataFrame(dataset["train"])
print(f"Dataset loaded with {len(df)} records")

# Step 2: Simulate demographics
def simulate_demographics(row):
    # Fixed here: access with row["Category"], handle NaN safely
    category = str(row["Category"]).strip() if pd.notna(row["Category"]) else ""
    interests = INTEREST_MAP.get(category, INTEREST_MAP["default"])
    return pd.Series({
        "age": random.choice(AGE_RANGE),
        "gender": random.choice(GENDERS),
        "interests": interests
    })

print("Simulating demographics...")
demo_df = df.apply(simulate_demographics, axis=1)
df = pd.concat([df, demo_df], axis=1)

# Step 3: Feature extraction from ad text (product description)
def extract_features(text):
    doc = nlp(str(text))
    # Extract noun chunks (noun phrases)
    features = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1]
    return ", ".join(features[:5])  # top 5 features

print("Extracting features...")
df['features'] = df['description'].apply(extract_features)

# Step 4: Normalize ad text
df['ad_text'] = df['description'].astype(str).str.strip()

# Step 5: Data cleaning
df.dropna(subset=["Product Name", "ad_text"], inplace=True)
df.drop_duplicates(subset=["Product Name", "ad_text"], inplace=True)
text_cols = ["Product Name", "features", "ad_text"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Step 6: Ad text augmentation - tone-based variations
def augment_ad_text(row, tone=None):
    base = row['ad_text']
    product = row['Product Name']
    tone_phrases = {
        "persuasive": [
            "Limited time offer! Don't miss out on {product}. Buy now!",
            "Grab your {product} today and enjoy exclusive benefits!"
        ],
        "informative": [
            "Discover the features of {product}. High quality guaranteed.",
            "{product} comes with outstanding specs to meet your needs."
        ],
        "casual": [
            "Check out this cool {product} that everyone is talking about!",
            "Looking for something fun? Try our {product} today."
        ],
        "neutral": [
            "{product} is now available. Learn more about its features."
        ]
    }
    if tone not in tone_phrases:
        tone = "neutral"
    phrase = random.choice(tone_phrases[tone])
    return f"{phrase.format(product=product)} {base}"

print("Augmenting ad texts with different tones...")
df['augmented_ad_text_persuasive'] = df.apply(lambda r: augment_ad_text(r, tone="persuasive"), axis=1)
df['augmented_ad_text_informative'] = df.apply(lambda r: augment_ad_text(r, tone="informative"), axis=1)
df['augmented_ad_text_casual'] = df.apply(lambda r: augment_ad_text(r, tone="casual"), axis=1)

# Step 7: Select final columns
final_cols = ["age", "gender", "interests", "Product Name", "features", "ad_text",
              "augmented_ad_text_persuasive", "augmented_ad_text_informative", "augmented_ad_text_casual"]
final_df = df[final_cols].copy()

# Step 8: Save final dataset
os.makedirs("data", exist_ok=True)
final_df.to_csv("data/amazon_ads_dataset_enhanced.csv", index=False)
print("Enhanced dataset saved as 'data/amazon_ads_dataset_enhanced.csv'")
print(f"Total records: {len(final_df)}")
