import pandas as pd
import spacy
import re
import os
import contractions
import textstat
from textblob import TextBlob
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

# === Load NLP models ===
nlp = spacy.load("en_core_web_trf")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT()
tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Config ===
brand_list = ["Nike", "Apple", "Samsung", "Sony"]
cta_phrases = ['buy now', 'shop today', 'don’t miss', 'grab it now', 'limited time', 'order today', 'get yours']
tone_labels = ["persuasive", "informative", "casual", "neutral"]

# === Preprocessing Functions ===

def clean_contractions(text):
    if pd.isnull(text): return ""
    text = contractions.fix(text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def extract_pos_keywords(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and token.is_alpha])

def detect_tone_ml(text):
    result = tone_classifier(text, tone_labels)
    return result['labels'][0]  # Most likely tone

def has_cta(text):
    return any(phrase in text.lower() for phrase in cta_phrases)

def get_sentiment(text):
    return round(TextBlob(text).sentiment.polarity, 3)

def extract_named_entities(text):
    doc = nlp(text)
    return ", ".join(set(ent.text for ent in doc.ents))

def anonymize_brands(text, brands):
    for brand in brands:
        text = re.sub(rf"\b{brand}\b", "[BRAND]", text, flags=re.IGNORECASE)
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def keyword_density(row):
    keywords = row["keywords"].split()
    words = row["clean_ad_text"].split()
    return round(len([w for w in words if w in keywords]) / len(words), 2) if words else 0

# === Main Pipeline ===

# 1. Load input
input_path = os.path.join("data", "amazon_ads_dataset_enhanced.csv")
df = pd.read_csv(input_path)

# 2. Text Cleaning
df["ad_text"] = df["ad_text"].astype(str).apply(clean_contractions)
df["features"] = df["features"].astype(str).apply(clean_contractions)

# 3. Basic NLP Preprocessing
print("Applying text preprocessing...")
df["clean_ad_text"] = df["ad_text"].apply(preprocess_text)
df["clean_features"] = df["features"].apply(preprocess_text)
df["keywords"] = df["ad_text"].apply(extract_pos_keywords)

# 4. Text Enhancements
print("Extracting tone, CTA, entities, etc...")
df["tone"] = df["ad_text"].apply(detect_tone_ml)
df["has_cta"] = df["ad_text"].apply(has_cta)
df["sentiment"] = df["ad_text"].apply(get_sentiment)
df["named_entities"] = df["ad_text"].apply(extract_named_entities)
df["anonymized_text"] = df["ad_text"].apply(lambda t: anonymize_brands(t, brand_list))
df["readability_score"] = df["ad_text"].apply(textstat.flesch_reading_ease)
df["language"] = df["ad_text"].apply(detect_language)
df = df[df["language"] == "en"]
df["keyword_density"] = df.apply(keyword_density, axis=1)

# 5. KeyBERT Semantic Keywords
print("Extracting semantic keywords...")
df["semantic_keywords"] = df["ad_text"].apply(lambda t: ", ".join([kw[0] for kw in kw_model.extract_keywords(t, top_n=5)]))

# 6. Sentence Embeddings (batched)
print("Generating sentence embeddings (batched)...")
texts = df["clean_ad_text"].tolist()
embeddings = embedder.encode(texts, convert_to_tensor=True, batch_size=64)
df["embedding"] = embeddings.tolist()

# 7. Optional memory optimization
df["gender"] = df["gender"].astype("category")
df["tone"] = df["tone"].astype("category")
df["interests"] = df["interests"].astype("category")

# 8. Save output
output_path = os.path.join("data", "amazon_ads_after_pre_processing.csv")
df.to_csv(output_path, index=False)
print(f"Enhanced dataset saved to '{output_path}'")
