from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

prompt = 'what is 2+2 ?'

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=50,
        temperature=0.2,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
