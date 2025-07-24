# Türkçe mc4 veri setini kullanarak google-smallt5 i fine tune etmeyi öğreniyorum

from datasets import load_dataset
import os

# kayıt dosya yolu kullanımı hatalı olabilir. emin değilim
kullanıcı_ana_dizini = os.path.expanduser('~')
kayıt_dosya_yolu = os.path.join(kullanıcı_ana_dizini, 'Desktop', 'Eren', 'Yazılım', 'Dialog-emotions')

#datasetin yüklenmesi, streaming=true belleğe yüklemeden yüklemeyi ifade eder
dataset = load_dataset('allenai/c4', 'tr', split='train', streaming=True)

# Büyük veri setini parça parça işleyip belleği verimli kullanmak için iterator oluştururuz.
iterator = iter(dataset)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

sentences = []
max_sentences = 5000 # Veri setinin büyüklüğü

# Cümleleri çekmek için kod kısmı
while len(sentences) < max_sentences:
    try:
        example = next(iterator)
        text = example['text']
        file_sentences = sent_tokenize(text)
        # 5 kelimeden az olan cümleleri ve boş olanları filtrelemek için bu kod kullanıldı.
        sentences.extend([s for s in file_sentences if len(s.split()) > 5 and s.strip()])
        if len(sentences) >= max_sentences:
            sentences = sentences[: max_sentences]
            break
    except StopIteration:
        break

print(f"Toplanan cümle sayısı: {len(sentences)}")
print(f"Örnek cümle: {sentences[0]}")

import random

turkish_chars = 'abcçdefgğhıijklmnoöprsştuüvyz'

# Şimdi de aldığımız cümleleri biraz bozarak doğru, yanlış şeklinde bir veri elde etmek için
# Cümleleri bozuyoruz ve farklı bir listeye kaydediyoruz.
def introduce_errors(sentence, error_rate=0.2):
    words = sentence.split()
    corrupted_words = []
    for word in words:
        if random.random() > error_rate or len(word) < 3:
            corrupted_words.append(word)
            continue
        error_type = random.choice(["delete", "insert", "substitute", "transpose"])
        pos = random.randint(0, len(word)-1)
        if error_type == 'delete' and len(word) > 1 :
            corrupted = word[: pos] + word[pos+1 :]
        elif error_type == "insert":
            char = random.choice(turkish_chars)
            corrupted = word[:pos] + char + word[pos:]
        elif error_type == "substitute": 
            char = random.choice(turkish_chars)
            corrupted = word[:pos] + char + word[pos+1:]
        elif error_type == "transpose" and pos < len(word) -1: 
            corrupted = word[:pos] + word[pos+1] + word[pos] + word[pos+2]
        else:
            corrupted = word
        
        corrupted_words.append(corrupted)
    return ' '.join(corrupted_words)

print("-" * 50)
# Bozuk cümle oluştur.
corrupted_sentences = [introduce_errors(s) for s in sentences]

print(f"Normal cümle: {sentences[0]}" )
print(f"Bozuk cümle: {corrupted_sentences[0]}" )

from datasets import Dataset

data = {
    'input_text': [f"düzelt: {cor}" for cor in corrupted_sentences],
    'target_text': sentences
}

dataset = Dataset.from_dict(data)

dataset = dataset.train_test_split(test_size=0.2)

subset = 5000
tokenized_dataset = dataset.shuffle(seed=42)
tokenized_dataset['train'] = tokenized_dataset['train'].select(range(4000))
tokenized_dataset['test'] = tokenized_dataset['test'].select(range(1000))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize fonksiyonu
def preprocess_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=64, truncation=True, padding='max_length')
    targets = tokenizer(examples['target_text'], max_length=64, truncation=True, padding='max_length')
    input['labels'] = targets['input_ids']
    return inputs

tokenized_dataset = tokenized_dataset.map(preprocess_function, batched = True)

# Fine-Tune için Trainer'i kurun
# Eğitim parametrelerini ayarlayalım.

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", 
    logging_strategy="steps", # Loggingi adımlara göre yap
    logging_steps = 10, # Her 10 adımda logla
    learning_rate= 2e-5,
    per_device_train_batch_size=4, #Batch_size = 4 OOM önlemek için küçültülür
    per_device_eval_batch_size=4,
    num_train_epochs = 5,
    weight_decay = 0.01, # modelin ağırlıkları çok büyümesin diye ufak bi düzeltme
    save_steps = 500,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,

    # Her batchte 4 örnek vardır ve 8 batchte bir kere gradyanlar güncellenir
    # Yani 32 cümlede bir gradyanlar güncellenir.
    gradient_accumulation_steps=8,
    warmup_steps=100, # Burada learning rateye ulaşmak için atılacak adım belirtiliyor
    # yani 0 dan başlayıp 100 adımda 2e-5 olacak şekilde artıyor.

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    # Erken durdurma callback method
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Eğitim sonrası logların print edilmesi. (loss ilerlemesi için)
print(trainer.state.log_history)


# Modeli kaydet
trainer.save_model(f"{kayıt_dosya_yolu}")

def correct_text(input_text):
    input_ids = tokenizer(f"düzelt: {input_text}", return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_length = 64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentence = "Bu bir tst cümle hatlarla dolu."
corrected = correct_text(test_sentence)
print(f"Bozuk: {test_sentence}")
print(f"Düzeltilmiş: {corrected}")
