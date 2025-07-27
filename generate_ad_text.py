from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import OrderedDict
import os

# Optional post-processing to reduce repeated words
def remove_repeats(text):
    words = text.split()
    seen = OrderedDict()
    for word in words:
        if word not in seen:
            seen[word] = True
    return " ".join(seen.keys())

def generate_ad(prompt, model_path='./models/fine_tuned_gpt2_clustered'):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            use_safetensors=True  # 
        )
        print(f"[SUCCESS] Loaded fine-tuned model from: {model_path}")
    except Exception as e:
        print(f"[WARNING] Failed to load fine-tuned model from: {model_path}")
        print(f"         Reason: {e}")
        print("[INFO] Falling back to base GPT-2 model...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate text with improved parameters
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=80,                 # Limit to 80 tokens
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2         
    )

    # Decode output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt if it's included in the output
    ad_text = full_text.replace(prompt, "").strip()

    # Optional: truncate at stop characters
    for stop_char in ['{', '"', '\n']:
        if stop_char in ad_text:
            ad_text = ad_text.split(stop_char)[0].strip()

    # Optional: remove repeated words (mild cleanup)
    ad_text = remove_repeats(ad_text)

    return ad_text

# ✅ CLI test
if __name__ == "__main__":
    prompt = "Write a compelling ad for eco-friendly water bottles targeting 18–25-year-olds."
    ad = generate_ad(prompt)
    print("\n--- Generated Ad ---")
    print(ad)
