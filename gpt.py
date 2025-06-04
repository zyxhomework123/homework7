import os
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_path = "models/gpt2-chinese-cluecorpussmall"

# 加载模型并显式配置pad_token
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 如果tokenizer没有pad_token，设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 改进的生成函数
def generate_text(prompt, max_length=50):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # 确保填充
        return_attention_mask=True  # 返回注意力掩码
    )

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "当人类第一次踏上火星"
print(generate_text(prompt))