from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokenized_train_dataset = load_from_disk("data/tokenized_train_dataset")
tokenized_val_dataset = load_from_disk("data/tokenized_val_dataset")

training_args = TrainingArguments(
    output_dir="./mediguide_gpt2",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    max_steps=3282,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

trainer.train()

model.save_pretrained("./mediguide_gpt2_finetuned")
tokenizer.save_pretrained("./mediguide_gpt2_finetuned")

print("Basic fine-tuning completed!")
print("Basic fine-tuned model and tokenizer saved!")