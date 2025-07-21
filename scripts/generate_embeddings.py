import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Load the cleaned data
df = pd.read_csv("data/cleaned_amazon_ads.csv")

# Replace NaN with empty string and convert to string type
df['clean_features'] = df['clean_features'].fillna("").astype(str)

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
df['features_emb'] = df['clean_features'].apply(lambda x: model.encode(x).tolist())

# Save embeddings
os.makedirs("data", exist_ok=True)
df.to_json("data/amazon_ads_dataset_with_embeddings.json", orient="records", lines=True)

print("Embeddings saved to data/amazon_ads_dataset_with_embeddings.json")
