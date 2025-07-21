from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_ad(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def clean_generated_ad(full_text, prompt):
    # Remove the prompt from the generated text
    text = full_text.replace(prompt, "").strip()
    # Remove trailing JSON or weird characters after first suspicious char
    for stop_char in ['{', '"', '\n']:
        if stop_char in text:
            text = text.split(stop_char)[0].strip()
    return text

if __name__ == "__main__":
    # Load tokenizer and model from your fine-tuned model directory
    tokenizer = GPT2Tokenizer.from_pretrained('./models/fine_tuned_gpt2')
    model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned_gpt2')

    # Fix padding token issue for GPT2 (if not done already)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Define one prompt to test
    prompt = "Write a personalized Facebook ad for a 30-year-old Female interested in skincare, fashion. The product is 'Lipstick' with features: long-lasting, moisturizing."

    # Generate ad
    full_generated_text = generate_ad(prompt, tokenizer, model)

    # Clean generated ad output
    ad_only = clean_generated_ad(full_generated_text, prompt)

    print("\n=== Generated Ad ===")
    print(ad_only if ad_only else "No ad generated. Try again or change the prompt.")
