import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Use the recommended caching decorator
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('./models/fine_tuned_gpt2')
    model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned_gpt2')
    return tokenizer, model

def generate_ad(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,          # Add randomness to generation
        top_p=0.9,               # Nucleus sampling for better quality
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load model/tokenizer once
tokenizer, model = load_model()

# Streamlit UI layout
st.set_page_config(page_title="Ad Generator", layout="centered")
st.title("📢 AI-Powered Personalized Marketing Ad Generator")

# User inputs
age = st.number_input("👤 Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("🚻 Gender", ["Male", "Female"])
interests = st.text_input("🎯 Interests", "skincare, fashion")
product = st.text_input("🛍️ Product Name", "Lipstick")
features = st.text_input("✨ Product Features", "long-lasting, moisturizing")

if st.button("🚀 Generate Ad"):
    # Handle possible 'nan' inputs
    clean_features = features if features and features.lower() != "nan" else ""
    
    prompt = (
        f"Write a personalized Facebook ad for a {age}-year-old {gender} "
        f"interested in {interests}. The product is '{product}' with features: {clean_features}."
    )
    
    # Generate the ad text from the model
    ad = generate_ad(prompt, tokenizer, model)
    
    # Remove the prompt from the generated output to keep only new text
    clean_ad = ad.replace(prompt, "").strip()
    
    st.subheader("📄 Generated Ad:")
    if clean_ad:
        st.success(clean_ad)
        st.download_button("📥 Download Ad", clean_ad, file_name="generated_ad.txt")
    else:
        st.warning("⚠️ The model did not generate any content. Try again with different inputs.")
