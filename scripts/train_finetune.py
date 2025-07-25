from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def tokenize_function(examples):
    # When batched=True, examples['prompt'] and examples['target'] are lists of strings
    texts = [p + " " + t for p, t in zip(examples['prompt'], examples['target'])]
    return tokenizer(texts, truncation=True, max_length=128)

def main():
    global tokenizer  # so tokenize_function can use it

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token, set to eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load dataset from jsonl file
    dataset = load_dataset("json", data_files="data/train_data_clustered_prompts.jsonl", split="train")

    # Tokenize dataset with the fixed tokenize_function
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "target"])

    # Data collator for causal LM (no masking)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./models/fine_tuned_gpt2_clustered")
    print("Model fine-tuning complete and saved to './models/fine_tuned_gpt2_clustered'")

if __name__ == "__main__":
    main()
