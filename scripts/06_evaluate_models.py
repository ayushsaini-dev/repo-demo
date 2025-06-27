from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from utils.evaluate_utils import evaluate_model

reference_answer = "The symptoms of diabetes include being very thirsty, frequent urination, feeling very hungry or tired, losing weight without trying, having sores that heal slowly, having dry, itchy skin, loss of feeling or tingling in the feet, and having blurry eyesight. Consult a healthcare provider for diagnosis."

test_prompt = "List the symptoms of diabetes in this format: 'The symptoms of diabetes include [list symptoms]. Consult a healthcare provider for diagnosis.'"

base_model = GPT2LMHeadModel.from_pretrained("./mediguide_gpt2")
tokenizer_basic = GPT2Tokenizer.from_pretrained("./mediguide_gpt2")
basic_results = evaluate_model(base_model, tokenizer_basic, test_prompt, reference_answer)
print("Basic Fine-Tuned Model Results:", basic_results)

base_model = GPT2LMHeadModel.from_pretrained("./mediguide_gpt2")
tokenizer_prompt = GPT2Tokenizer.from_pretrained("./mediguide_gpt2_prompt_tuned_v2")
prompt_model = PeftModel.from_pretrained(base_model, "./mediguide_gpt2_prompt_tuned_v2")
prompt_results = evaluate_model(prompt_model, tokenizer_prompt, test_prompt, reference_answer)
print("Prompt-Tuned Model Results:", prompt_results)

base_model = GPT2LMHeadModel.from_pretrained("./mediguide_gpt2")
tokenizer_lora = GPT2Tokenizer.from_pretrained("./mediguide_gpt2_lora")
lora_model = PeftModel.from_pretrained(base_model, "./mediguide_gpt2_lora")
lora_results = evaluate_model(lora_model, tokenizer_lora, test_prompt, reference_answer)
print("LoRA Fine-Tuned Model Results:", lora_results)