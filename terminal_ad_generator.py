import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
from textblob import TextBlob
from langdetect import detect
import textstat
import contractions
import re

# Load models
print("Loading models...")
tokenizer = GPT2Tokenizer.from_pretrained("./models/fine_tuned_gpt2_clustered")
model = GPT2LMHeadModel.from_pretrained("./models/fine_tuned_gpt2_clustered")
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = contractions.fix(text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def generate(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("=== Personalized Facebook Ad Generator ===")

    # Take user input
    age = input("Enter target age: ")
    gender = input("Enter gender (Male/Female): ")
    interest = input("Enter user's interest (e.g., tech, fitness): ")
    product_name = input("Enter product name: ")
    features = input("Enter product features (comma separated): ")

    # Preprocess
    age = clean_text(age)
    gender = clean_text(gender)
    interest = clean_text(interest)
    product_name = clean_text(product_name)
    features = clean_text(features)
    
    # Construct prompt
    prompt = (
        f"Write a persuasive Facebook ad for a {age}-year-old {gender} "
        f"interested in {interest}. The product is '{product_name}' "
        f"with features: {features}."
    )

    print("\nGenerating ad...\n")
    ad = generate(prompt)
    print("=== Generated Ad ===")
    print(ad)
    print("====================")

if __name__ == "__main__":
    main()
