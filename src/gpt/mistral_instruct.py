from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

login(token='')


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model.eval()

prompt = 'where can I find some fresh air in Seattle ?'

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=512,
        temperature=0.2,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
