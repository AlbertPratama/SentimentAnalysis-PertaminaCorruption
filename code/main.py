import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load models
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=3)

# # input text
# text = 'Saya sangat bahagia karena sepertinya dia telah dengan yang lain'
text = 'korupsi aja semua terus....sampai indonesia bangkrut'

# Input tokenizations
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    logits = model(**inputs).logits
    print(logits)

# act funct
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# probs class
predicted_class_id = torch.argmax(probabilities, dim=-1).item()

# showed outoput
label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
predicted_class = label_map[predicted_class_id]

print(f'Teks: "{text}"')
print(f'Prediksi Sentimen: {predicted_class}')
