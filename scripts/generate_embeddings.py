import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

# Load cleaned data
df = pd.read_csv("data/amazon_ads_after_pre_processing.csv")

# Fill NaN with empty strings for safe processing
df['clean_features'] = df['clean_features'].fillna("").astype(str)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for product features
print("Generating embeddings...")
embeddings = model.encode(df['clean_features'].tolist(), show_progress_bar=True)

# Add embeddings to dataframe
df['features_emb'] = embeddings.tolist()

# Perform clustering to segment products/users
NUM_CLUSTERS = 10  # You can adjust this
print(f"Clustering into {NUM_CLUSTERS} segments...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# Save dataframe with clusters and embeddings (embeddings saved as JSON array strings)
os.makedirs("data", exist_ok=True)
df.to_json("data/amazon_ads_with_clusters.json", orient="records", lines=True)

print("Embeddings and clusters saved to data/amazon_ads_with_clusters.json")
