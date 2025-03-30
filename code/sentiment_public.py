import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

df = pd.read_csv('../datasets/cleaned_comment_public.csv')
df = df.dropna()

model_path = './tuned_indobert_for-sentiment/checkpoint-615'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


id2labels = {0: "Positif", 1: "Netral", 2: "Negatif"}


def predict(data):
    input_data = tokenizer(data, return_tensors='pt', truncation=True, max_length=512)
    output_data = model(**input_data)

    logits = output_data.logits.detach().numpy()
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    predict_id = np.argmax(probabilities, axis=1)[0]
    predictions = id2labels[predict_id]
    confidence = probabilities[0][predict_id]



    return predictions, confidence


df['predict_label'] = df['comment_cleansing'].apply(lambda x: predict(x)[0])

df.to_csv('./predicted_datasets/predict_public_comment.csv', index=False)