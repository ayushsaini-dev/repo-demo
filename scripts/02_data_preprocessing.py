import json
from datasets import Dataset
from transformers import GPT2Tokenizer

with open("data/test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open("data/train_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("data/val_data.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

train_texts = [pair["question"] + " Answer: " + pair["answer"] for pair in train_data]
val_texts = [pair["question"] + " Answer: " + pair["answer"] for pair in val_data]
test_texts = [pair["question"] + " Answer: " + pair["answer"] for pair in test_data[:100]]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})
test_dataset = Dataset.from_dict({"text": test_texts})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_train_dataset.save_to_disk("data/tokenized_train_dataset")
tokenized_val_dataset.save_to_disk("data/tokenized_val_dataset")
tokenized_test_dataset.save_to_disk("data/tokenized_test_dataset")

print("Data preprocessing completed!")