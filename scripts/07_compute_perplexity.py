from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from datasets import load_from_disk
from utils.evaluate_utils import compute_perplexity

tokenizer_basic = GPT2Tokenizer.from_pretrained("./mediguide_gpt2")
tokenizer_basic.pad_token = tokenizer_basic.eos_token

tokenized_test_dataset = load_from_disk("data/tokenized_test_dataset")

basic_model = GPT2LMHeadModel.from_pretrained("./mediguide_gpt2")
prompt_model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2"), "./mediguide_gpt2_prompt_tuned_v2")
lora_model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2"), "./mediguide_gpt2_lora")

basic_model.to("cuda")
prompt_model.to("cuda")
lora_model.to("cuda")

basic_perplexity = compute_perplexity(basic_model, tokenized_test_dataset, tokenizer_basic)
print("Basic Fine-Tuned Model Perplexity:", basic_perplexity)

prompt_perplexity = compute_perplexity(prompt_model, tokenized_test_dataset, tokenizer_basic)
print("Prompt-Tuned Model Perplexity:", prompt_perplexity)

lora_perplexity = compute_perplexity(lora_model, tokenized_test_dataset, tokenizer_basic)
print("LoRA Fine-Tuned Model Perplexity:", lora_perplexity)