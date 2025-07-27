from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def tokenize_function(examples):
    # Combine prompt and target into a single training text
    texts = [p + " " + t for p, t in zip(examples['prompt'], examples['target'])]
    return tokenizer(texts, truncation=True, max_length=128)

def main():
    global tokenizer  # So it can be used in tokenize_function

    # === Load tokenizer and model ===
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos (GPT-2 requirement)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # === Load dataset ===
    dataset = load_dataset("json", data_files="data/train_data_clustered_prompts.jsonl", split="train")

    # === Tokenize dataset ===
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "target"])

    # === Prepare data collator ===
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Define training arguments ===
    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_gpt2_clustered",   # Save everything here
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no"
    )

    # === Initialize Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # === Start training ===
    trainer.train()

    # === Save tokenizer (important for inference consistency) ===
    tokenizer.save_pretrained("./models/fine_tuned_gpt2_clustered")

    print("✅ Model and tokenizer saved to './models/fine_tuned_gpt2_clustered'")

if __name__ == "__main__":
    main()
