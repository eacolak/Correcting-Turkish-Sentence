
#!pip install huggingface_hub
#!pip install datasets
#!pip install transformers
#!pip install sentencepiece

#!pip install torch
# Upgrade datasets library to fix the streaming error
#!pip install --upgrade datasets.

#TODO kendim için not: Bu kod pcde çalışabilicek durumda. Colab için t5-basedFinetuneE.ipynb
#TODO de yazılı.

#? Bu kodda google t5-base modelini allenai c4 ile fine tune etmeye çalışıyorum
#? çıktılar 10 epoch içerisinde 36 dan 1 e düşse de çıktı vermedi.



from datasets import load_dataset
import os


# Load the recommended 'allenai/c4' dataset (replaces deprecated 'mc4')
dataset = load_dataset('allenai/c4', 'tr', split='train', streaming=True)

# Create an iterator for efficient processing
iterator = iter(dataset)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

sentences = []
max_sentences = 10000  # Limit dataset size

# Extract sentences
while len(sentences) < max_sentences:
    try:
        example = next(iterator)
        text = example['text']
        file_sentences = sent_tokenize(text)
        # Filter sentences: at least 5 words and non-empty
        sentences.extend([s for s in file_sentences if len(s.split()) > 5 and s.strip()])
        if len(sentences) >= max_sentences:
            sentences = sentences[:max_sentences]
            break
    except StopIteration:
        break

print(f"Toplanan cümle sayısı: {len(sentences)}")
print(f"Örnek cümle: {sentences[0]}")

import random

turkish_chars = 'abcçdefgğhıijklmnoöprsştuüvyz'

# Function to introduce errors
def introduce_errors(sentence, error_rate=0.2):
    words = sentence.split()
    corrupted_words = []
    for word in words:
        if random.random() > error_rate or len(word) < 3:
            corrupted_words.append(word)
            continue
        error_type = random.choice(["delete", "insert", "substitute", "transpose"])
        pos = random.randint(0, len(word)-1)
        if error_type == 'delete' and len(word) > 1:
            corrupted = word[:pos] + word[pos+1:]
        elif error_type == "insert":
            char = random.choice(turkish_chars)
            corrupted = word[:pos] + char + word[pos:]
        elif error_type == "substitute":
            char = random.choice(turkish_chars)
            corrupted = word[:pos] + char + word[pos+1:]
        elif error_type == "transpose" and pos < len(word) - 1:
            corrupted = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        else:
            corrupted = word
        corrupted_words.append(corrupted)
    return ' '.join(corrupted_words)

print("-" * 50)
# Create corrupted sentences
corrupted_sentences = [introduce_errors(s) for s in sentences]

print(f"Normal cümle: {sentences[0]}")
print(f"Bozuk cümle: {corrupted_sentences[0]}")

from datasets import Dataset

data = {
    'input_text': [f"düzelt: {cor}" for cor in corrupted_sentences],
    'target_text': sentences
}

dataset = Dataset.from_dict(data)

dataset = dataset.train_test_split(test_size=0.2)

# Shuffle and select subsets
tokenized_dataset = dataset.shuffle(seed=42)
tokenized_dataset['train'] = tokenized_dataset['train'].select(range(8000))
tokenized_dataset['test'] = tokenized_dataset['test'].select(range(2000))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenization function (fixed typo)
def preprocess_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding='max_length')
    targets = tokenizer(examples['target_text'], max_length=128, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']  # Fixed: inputs['labels']
    return inputs

tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)


# Trainer ayarlama
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import get_scheduler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_steps=500,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,  # Enable if GPU available; set False if CPU-only
    max_grad_norm=1.0,
    gradient_accumulation_steps=8,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_checkpointing=True,
)

#! pip3 install torch
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate,)

num_training_steps = len(tokenized_dataset['train']) * training_args.num_train_epochs // training_args.per_device_train_batch_size
scheduler = get_scheduler(name=training_args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)

# belleği sıfırlamak icin ufak tefek şeylerde dolayı cuda hatası almayalım diye
torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    optimizers = (optimizer, scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# Print training logs
print(trainer.state.log_history)

# Save the model
trainer.save_model("./fine_tuned_model")

def correct_text(input_text):
    input_ids = tokenizer(f"düzelt: {input_text}", return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentence = "Bu bir tst cümle hatlarla dolu."
corrected = correct_text(test_sentence)
print(f"Bozuk: {test_sentence}")
print(f"Düzeltilmiş: {corrected}")