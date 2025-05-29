import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
tokenizer = BertTokenizer.from_pretrained("models/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("models/bert-base-chinese", num_labels=2)


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "正面" if prediction == 1 else "负面"

review = "剧情老套，充满套路和硬凹的感动。"
delivery = "汤汁洒得到处都是，包装太随便了。"


print(f"影评分类结果: {predict_sentiment(review)}")
print(f"外卖分类结果: {predict_sentiment(delivery)}")





