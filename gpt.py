import os
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_path = "models/gpt2-chinese-cluecorpussmall"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 生成文本
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
prompt = "我走进了那扇从未打开过的门"
print(generate_text(prompt))