import pandas as pd

def create_prompt(row):
    # You can modify prompt style based on cluster number to control tone/style
    cluster = row['cluster']
    tone_map = {
        0: "persuasive",
        1: "informative",
        2: "casual",
        3: "neutral",
        4: "persuasive",
        5: "informative",
        6: "casual",
        7: "neutral",
        8: "persuasive",
        9: "informative"
    }
    tone = tone_map.get(cluster, "neutral")

    prompt = (
        f"Write a {tone} Facebook ad for a {row['age']}-year-old {row['gender']} "
        f"interested in {row['interests']}. The product is '{row['Product Name']}' "
        f"with features: {row['features']}."
    )
    return prompt

if __name__ == "__main__":
    df = pd.read_json("data/amazon_ads_with_clusters.json", lines=True)
    df['prompt'] = df.apply(create_prompt, axis=1)
    df['target'] = df['ad_text']
    df[['prompt', 'target']].to_json('data/train_data_clustered_prompts.jsonl', orient='records', lines=True)
    print("Prompts with cluster-based tones created for fine-tuning.")
