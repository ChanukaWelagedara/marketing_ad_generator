# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# def generate_ad(prompt, model_path='./models/fine_tuned_gpt2'):
#     try:
#         tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#         model = GPT2LMHeadModel.from_pretrained(model_path)
#         print(f"Loaded fine-tuned model from {model_path}")
#     except Exception as e:
#         print(f"Failed to load fine-tuned model. Falling back to base GPT-2. Error: {e}")
#         tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         model = GPT2LMHeadModel.from_pretrained("gpt2")

#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.eos_token_id

#     inputs = tokenizer(prompt, return_tensors="pt", padding=True)
#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_length=100,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         top_p=0.9,
#         temperature=0.8
#     )
#     full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove prompt from output
#     ad_text = full_text.replace(prompt, "").strip()
#     for stop_char in ['{', '"', '\n']:
#         if stop_char in ad_text:
#             ad_text = ad_text.split(stop_char)[0].strip()
#     return ad_text
