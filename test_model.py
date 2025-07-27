from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model

def generate_ad_text(model, tokenizer, prompt, max_length=120, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove prompt prefix if present
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text

def main():
    model_path = "./models/fine_tuned_gpt2"  # change to your model folder
    print("Loading model...")
    tokenizer, model = load_model(model_path)
    print("Model loaded! Enter your prompt or 'quit' to exit.")

    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() in ('quit', 'exit'):
            print("Exiting. Goodbye!")
            break

        print("\nGenerating ad...")
        ad = generate_ad_text(model, tokenizer, prompt)
        print("\n--- Generated Ad ---")
        print(ad)
        print("--------------------")

if __name__ == "__main__":
    main()
