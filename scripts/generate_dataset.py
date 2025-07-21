from datasets import load_dataset
import pandas as pd
import random
import os

# 1. Load the dataset
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm")
data = dataset["train"]

# 2. Convert to pandas for easier manipulation
df = pd.DataFrame(data)

# Optional: check columns and unique categories
print("Columns:", df.columns.tolist())
print("Unique Categories:", df["Category"].unique())

# 3. Simulate age, gender, and interests
ages = list(range(18, 65))
genders = ["Male", "Female"]
interests_by_category = {
    "Arts & Crafts": "art, DIY",
    "Electronics": "tech, gadgets",
    "Books": "reading, literature",
    "Toys & Games": "gaming, kids",
    "Beauty": "fashion, skincare",
    "Sports": "fitness, outdoors",
    "Home & Kitchen": "home decor, cooking",
}

def random_interest(cat):
    return interests_by_category.get(cat, "general, shopping")

# 4. Add new columns
df["age"] = df.apply(lambda x: random.choice(ages), axis=1)
df["gender"] = df.apply(lambda x: random.choice(genders), axis=1)
df["interests"] = df.apply(lambda x: random_interest(x["Category"]), axis=1)

# The dataset has no 'feature_bullets' column, use an empty string or another column if suitable
df["features"] = ""  # or df["Product Specification"] if you want to summarize features

df["ad_text"] = df["description"]

# 5. Keep only needed columns (adjust column names as per dataset)
final_df = df[["age", "gender", "interests", "Product Name", "features", "ad_text"]]

# 6. Create data directory if not exists
os.makedirs("data", exist_ok=True)

# 7. Save to CSV
final_df.to_csv("data/amazon_ads_dataset.csv", index=False)

print("Dataset saved as 'data/amazon_ads_dataset.csv'")
