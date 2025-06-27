from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

base_model_name = "gpt2"
base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
tokenizer = GPT2Tokenizer.from_pretrained("./mediguide_gpt2_lora")
base_model.to("cuda")

model = PeftModel.from_pretrained(base_model, "./mediguide_gpt2_lora")
model.to("cuda")
model.eval()

test_prompt = "List the symptoms of diabetes in this format: 'The symptoms of diabetes include [list symptoms]. Consult a healthcare provider for diagnosis.'"
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
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated response:", response)