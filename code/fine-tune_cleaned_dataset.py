# Load important libary
from datasets import load_dataset, ClassLabel, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import evaluate
import numpy as np
import pandas as pd

# Load dataset 
data_txt = pd.read_csv('../datasets/cleaned_train_dataset.csv')
print(data_txt)

#  EDA 
print(data_txt.info())
print(data_txt.describe(include='all'))
print(data_txt.isna().sum())
print(data_txt[data_txt['text'].isna()] )

data_txt = data_txt.dropna()
print(data_txt.head())
print('\nNull values: ',data_txt.isna().sum())



print('\nData Information: ', data_txt.info())
print('Describe: ', data_txt.describe(include='all'))

# Convert label to ClassLabel 
label_names = ['Negatif', 'Netral', 'Positif']
data_txt = Dataset.from_pandas(data_txt)

data_txt = data_txt.cast_column('label', ClassLabel(names=label_names))
print(data_txt)

5. 
train_val_split = data_txt.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column='label'
)
train_data = train_val_split['train']
val_data = train_val_split['test']


print(train_data['text'][0])
print(val_data)

print(train_data['text'])
print(train_data[0]["text"]) 
print(type(train_data))  

# save val_data to csv 
val_data.to_csv('val_cleaned_data.csv',index=False)


# Tokenization
model_name = "indolem/indobert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # make sure the data is String
    texts = examples['text']
    if isinstance(texts, list):
        texts = [str(text) for text in texts]
    else:
        texts = str(texts)
    
    return tokenizer(texts, truncation=True, max_length=512)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)


tokenized_train = tokenized_train.rename_column('label', 'labels')
tokenized_val = tokenized_val.rename_column('label', 'labels')


tokenized_train = tokenized_train.remove_columns(['text'])
tokenized_val = tokenized_val.remove_columns(['text'])


tokenized_train.set_format("torch")
tokenized_val.set_format("torch")


print("Kolom tokenized_train:", tokenized_train.column_names)
print("Kolom tokenized_val:", tokenized_val.column_names)
print()
print(f"Jumlah data training: {len(tokenized_train)}")
print(f"Jumlah data validasi: {len(tokenized_val)}")


print("Contoh data training:")
print(tokenized_train[0]) 
print("Contoh data validasi:")
print(tokenized_val[0])


# Model configuration
id_to_labels = {0: 'Positif', 1: 'Netral', 2: 'Negatif'}
label_to_id = {'Positif': 0, 'Netral': 1, 'Negatif': 2}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id_to_labels,
    label2id=label_to_id
)


# Sample out
sample = val_data[0]
print(f"Contoh teks: {sample['text']}")
print(f"Label asli (numerik): {sample['label']}")
print(f"Label asli (nama): {id_to_labels[sample['label']]}")


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = np.sum(preds==labels) / len(labels)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    conf_matx = confusion_matrix(labels, preds)
    
    return {"accuracy": acc, 
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion matrix": conf_matx.tolist()}



# Training hyperparameters
training_args = TrainingArguments(
    output_dir="tuned_indobert_for-sentiment",  # Output dir
    learning_rate=2e-5,  # Learning rates
    per_device_train_batch_size=16,  # batch size
    per_device_eval_batch_size=16,  # batch size
    num_train_epochs=5,  # Epoch learning
    weight_decay=0.01,  # L2 Regularization
    evaluation_strategy="epoch",  # Evaluation every step an epoch 
    save_strategy="epoch",  # Save model after epoch
    load_best_model_at_end=True,  # Load the model 
    metric_for_best_model="f1",  # F1-score as the best model metric
    save_total_limit=1,  # Limit checkpoint 
    logging_dir="./tuned_indobert_for-sentiment/logs",  # Output dir log
    logging_strategy="epoch",  # Saving log each Epoch
)



#  Trainer inisiation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



# Running model train
trainer.train()

# 12. Evaluation metrics
print("Evaluasi pada data validasi:")
results = trainer.evaluate()
print(f"Hasil Evaluasi: {results}")



# Testing model predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = np.exp(outputs.logits.detach().numpy())
    probs = probs / probs.sum(axis=-1, keepdims=True)
    pred_id = np.argmax(probs)
    return id_to_labels[pred_id], probs[0][pred_id]


# Test contoh yang bermasalah
# text = "oi ngerti bahasa indonesia kan pokoknya sampe jam belon beres gua gratis jam mokad gratis dst dst lu gila mati service melulu jam berkali &amp"
# text = 'saya rasa itu hal yang tidak perlu di besar-besarkan'
text = 'biasa saja sih'
prediction, confidence = predict(text)
print(f"\nText: {text}")
print(f"Prediksi: {prediction} (Confidence: {confidence:.2f})")