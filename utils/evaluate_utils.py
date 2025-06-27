import time
import os
from rouge_score import rouge_scorer
import torch

def evaluate_model(model, tokenizer, test_prompt, reference_answer):
    model.to("cuda")
    model.eval()

    start_time = time.time()
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=False,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    latency = time.time() - start_time
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, generated_response)

    model_dir = model.config._name_or_path if not hasattr(model, 'base_model') else model.base_model.config._name_or_path
    model_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))) / (1024 ** 2)

    return {
        "generated_response": generated_response,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "latency": latency,
        "model_size_mb": model_size,
        "perplexity": "TBD",
    }

def compute_perplexity(model, dataset, tokenizer):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataset:
            input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to("cuda")
            labels = torch.tensor(batch["labels"]).unsqueeze(0).to("cuda")
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += (labels != tokenizer.pad_token_id).sum().item()

    avg_loss = total_loss / len(dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity