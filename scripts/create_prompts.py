import pandas as pd

def create_prompt(row):
    return (
        f"Write a personalized Facebook ad for a {row['age']}-year-old {row['gender']} "
        f"interested in {row['interests']}. The product is '{row['Product Name']}' "
        f"with features: {row['features']}."
    )

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_amazon_ads.csv")
    df['prompt'] = df.apply(create_prompt, axis=1)
    df['target'] = df['ad_text']
    df[['prompt', 'target']].to_json('data/train_data.jsonl', orient='records', lines=True)
    print("Prompts created for fine-tuning.")
