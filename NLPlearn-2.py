import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from gensim.models import FastText

try:
    kullanıcı_ana_dizini = os.path.expanduser('~')
    proje_klasoru = os.path.join(kullanıcı_ana_dizini, 'Desktop', 'Eren', 'Yazılım', 'Correcting-Turkish-Sentence')
except FileNotFoundError:
    print("Dosya bulunamadı.")
except Exception as e:
    print(f"Hata: Dosyaya ulaşılırken hata oluştu: {e}")

#dosya yolları
json_dosya_yolu = os.path.join(proje_klasoru, 'yazim_yanlislari_dataset.json')
print("Json dosyasın okunuyor...")
try:
    df = pd.read_json(json_dosya_yolu)
    hatali_metinler = df['hatali_metin'].tolist()
    dogru_metinler = df['dogru_metin'].tolist()

except FileNotFoundError:
    print("Dosya bulunamadı")
    exit()
except Exception as e:
    print(f"Hata:{e}")
    exit()


# Karakter bazlı tokenizasyona geçerken Metin_Temizle fonksiyponunu da devredisi bıraktık cunku bu durumu zaten
# TextVectorization fonksiyonu da yapabiliyor gereksiz kod yazmak yerine işi ona bıraktık.

#metinlerin temizlenmesi gürültü azaltmak amacıyla 
# def metin_temizle(metin):
#     metin = metin.lower()  
#     metin = re.sub(r"i", "İ", metin)
#     metin = re.sub(r"ı", "I", metin)
#     metin = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ0-9\s\n.,!?]', '', metin)
#     metin = metin.strip()
#     metin = "<start>" + metin + "<end>"
#     return metin





# Metinleri temizleme işlemi. Fakat hatalı metinlerin temizlenmesi elimizdeki
# verileri bir nevi kaybetmek olurdu bu yüzden hatalı metinler için uygulanmaz.
# Karakter bazlı tokenizasyona geçtiğimizde temizleme yapmıyoruz.
# Cümlenin tamamı için başına start sonuna end konur
input_texts = ["<start>" + metin.lower().strip() + "<end>" for metin in hatali_metinler]
target_texts = ["<start> " + metin.lower().strip() + " <end>" for metin in dogru_metinler]


# MAX_VOCAB_SIZE = 10000 # Kelime dağarcığı
MAX_SEQ_LENGTH = 150 # Cümle maksimum uzunluğu
embedding_dim = 128 # Kelimeler embedding uygularken yani anlam vektörü haline getirirken hangi uzunlukta olmalı onu seçiyoruz.

#Vectorize etmek için vectorizerler hazırlandı
# Karakter tabanlı tokenizasyona geçince her karakter ayrı ayrı idlendirilip bir sözlük haline getirliyor.
input_vectorizer = tf.keras.layers.TextVectorization(
    #max_tokens = MAX_VOCAB_SIZE,
    output_sequence_length = MAX_SEQ_LENGTH,
    standardize = None,
    split = 'character',
)

target_vectorizer = tf.keras.layers.TextVectorization(
    #max_tokens = MAX_VOCAB_SIZE,
    output_sequence_length = MAX_SEQ_LENGTH + 1, #decoder target için yaygın kullanılırmış
    standardize = None,
    split = 'character',
)

# Bu kısımda vectorize etmek için kelimeler tarandı ve kelime : ID sözlüğü oluşturuldu
input_vectorizer.adapt(input_texts)
target_vectorizer.adapt(target_texts)

#Kelimelere ID verildi padding ve truncation uygulandı
#Bu kısım cümle vektörleridir. Henüz model için gerekli olan
#anlam vektörleri uygulanmadı. Bunun için embedding gerekli.
input_sequence = input_vectorizer(np.array(input_texts))
target_sequence = target_vectorizer(np.array(target_texts))

#print(input_vectorizer.get_vocabulary())
#print(input_sequence.numpy())
#print("SEQUENCE DÜMDÜZ YAZDIRMA:#########################:#########################:#########################:#########################")
#print(input_sequence)

print("FastText vocab yükleniyor... İşlem biraz sürebilir.")
try:
    fasttext_model_yolu =  "~/Desktop/eren/Yazılım/Correcting-Turkish-Sentence/cc.tr.300.bin"
    fasttext_model = FastText.load_fasttext_format(fasttext_model_yolu)

except FileNotFoundError:
    print(f"{fasttext_model_yolu} adresinde dosya bulunamadı.")
    exit()

char_vocab = input_vectorizer.get_vocabulary()
vocab_size = len(char_vocab)
embedding_dim = fasttext_model.wv.vector_size

print(f"Karakter sözlüğü boyutu: {vocab_size}")
print(f"FastText Embedding boyutu {embedding_dim}")

#ağırlık matrisi oluşturma
embedding_matrix = np.zeros((vocab_size, embedding_dim))
hits = 0
misses = 0

for i, char in enumerate(char_vocab):
    try:
        #Karakterin vektörünü fasttext modelinden al
        embedding_vector = fasttext_model.wv[char]
        embedding_matrix[i] = embedding_vector
        hits +=1
    except KeyError:
        misses += 1


print(f"FastTextte bulunan karakter sayısı: {hits}")
print(f"FastTextte bulunmayan karakter sayısı: {misses}")

embedding_layer_block = tf.keras.layers.Embedding(
    input_dim = vocab_size,
    output_dim = embedding_dim,
    weights = [embedding_matrix],
    trainable = True, # Backpropagation ile embedding katmanlarında düzeltme olabilmesi için bu durum true yapılır. 
    name = "FastText_Destekli_Embedding"
)

print("\nFastText entegrasyonu tamamlandı!")
print(f"Oluşturulan Embedding katmanının şekli (Shape): {embedding_layer_block.get_weights()[0].shape}")

print(f"Satırlar kelimeler için ({embedding_layer_block.get_weights()[0].shape[0]}), \nSütunlar ise özellikler için ayrılmıştır({embedding_layer_block.get_weights()[0].shape[1]})")
print("vocab_size: ", vocab_size)



# --- Eğitim için girdi ve hedefleri hazırlama ---
# Encodere verilecek olan hatalı metinlerin ID'leri
encoder_input_data = input_sequence

#Decoder'a girdi olarak, hedef cümlenin son token'ı hariç olan kısmı verilir.
# Modelin, bir sonraki karakteri tahmin etmesi istenir
decoder_input_data = target_sequence[:, :-1]

# Loss fonksiyonunda karşılaştırılacak olan gerçek hedef ise, cümlenin ilk token'ı hariç
# olan kısımdır.
decoder_target_data = target_sequence[:, 1:]

# --- tf.data.Dataset Nesnesini oluşturma ---
dataset = tf.data.Dataset.from_tensor_slices(
    ((encoder_input_data, decoder_input_data), decoder_target_data)
)
# --- Veri setinin eğitim için yapılandırılması ---
BUFFER_SIZE =len(input_sequence)
dataset = dataset.shuffle(BUFFER_SIZE)

# Veriyi batchlere (gruplara) ayırıyoruz. Model her adımda 64'lük gruplar halinde veri işleyecek
BATCH_SIZE = 64
dataset = dataset.batch(BATCH_SIZE)

# Performans optimizasyonu: GPU bir batch üzerinde çalışırken, CPU'nun
# sonraki batch'i hazırlamasını sağlar. Bu, bekleme sürelerini ortadan kaldırır
dataset = dataset.prefetch(tf.data.AUTOTUNE)
print("\nEğitime hazır 'dataset' nesnesi oluşturuldu!")
print("Dataset bir batch'te şu yapıda veri döndürecek:")
for (inputs, targets) in dataset.take(1):
    enc_in, dec_in = inputs
    print(f"  - Encoder Girdisi Şekli: {enc_in.shape}")
    print(f"  - Decoder Girdisi Şekli: {dec_in.shape}")
    print(f"  - Hedef (Loss için) Şekli: {targets.shape}")



# Embedding katmanından gelen Anlam vektörünün encodere girmeden hemen önce hangi karakterin nerede olduğunu encoderin anlayabilmesi için bir
# Konum damgalama gibi çalışır
# Çalışma prensibi basit olarak şöyle:
# Embedding katmanından gelen veri önce aynı uzunlukta yani max_seq_length ve embedding_dim uzunlukları kullanılarak bir boş matris oluşturulur.
# Bu matris daha sonra tek olanalar sin çift olanlar cos fonskiyonu ile doldurulur.
# Daha sonra bu oluşturulan matrisle bizim embedding katmanından gelen matris toplanarak karakterler için konum bilgisinin de anlaşılabileceği
# vektör elde edilir.
class PositionalEncoding(tf.keras.layer.Layer):
    def __init__(self, position, d_model):
        #position(int):sekansın maksimum uzunluğu(max_seq_length)
        #d_model(int) : embedding boyutu(embedding_dim,yani 300)

        super(PositionalEncoding ,self).__init__()
        
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    #açıların hesaplandığı formül
    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates


    # burası hakkında en ufak bi fikrim yok nasıl çalıştığını anlamadım.
    
    def positional_encoding(self, position, d_model):
        
        #(position, d_model) şeklinde bir matris
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis], # Satırlar: pozisyonlar (0,1, ..., 149)
            np.arange(d_model)[np.newaxis, :], # Sütunlar: embedding boyutları (0, 1, ..., 299)
            d_model
        )

        #Çift indeksli sütunlara sinüs uygula
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        #Tek indeksli sütunlara cosinüs uygula
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]

        return inputs + self.pos_encoding[:, :seq_len, :]



# --- Encoder kısmı ---

# Multi-Head Self-Attention için kendi sarmalayıcı katmanımız

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        # kerasın hazır multiheadattention fonksiyonu ile kaç başlıkolacağını: num_heads
        # ve başlık başına ne kadar özellik düşeceğini belirliyoruz.
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, # Okuyucu başlık sayısı
            key_dim = d_model // num_heads # özellik sayısı // num_heads
        )
        
        # multiheadattention katmanının çıktısını tekrar orijinal 
        # boyuta getirmek için d_model ile sinir ağına sokarız
        self.dense_layer = tf.keras.layers.Dense(d_model)


    def call(self, inputs):
        attention_output = self.attention(
            query = inputs,
            value = inputs,
            key = inputs
        )

        return self.dense_layer(attention_output)


# Her bir pozisyon için ayrı ayrı uygulanan Feed-Forward ağı
# pozisyonların diğer pozisyonlardan izole şekilde öğrendiklerini işlemesidir
# Toplantıdan sonra ofisine çekilip toplantı raporlarını inceleyen bir çalışan gibi.
# burada bir nevi önce genişletip daha büyük bir alanda öğrenim yapıyor
# sonra daraltırlıyor.
class PositionWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        # dff = dimension of feed-forward
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.dense1 = tf.keras.layers.Dense(dff, activation = 'relu')

        # bir sonraki encoder bloğuna girebilmesi için orijinal boyutuna çekiyoruz
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


# Encoder katmanında önce mutli head fonksiyonuna sonra dropouta sonra bir normalizasyon fonksiyonuna sokulur.
# normalizasyon fonksiyonu içeriği ilk katman girdisi + dropout çıktısıdır 
# Bu katmandan sonra feed forward katmanı gelir ve bu katman da ilk feed forwarda sokulur sonra dropouta
# sonrasında da forward ilk girdisi + dropout çıktısı haliyle normalize edilir.


# Tam bir encoder katmanı
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        # Multi-Head Self-Attention katmanımız.
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        # Position-Wise Feed-Forward Network katmanımız.
        self.ffn = PositionWiseFeedForwardNetwork(d_model, dff=dff)


        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon =1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon =1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)


    def call(self, inputs, training):
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate = 0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Encoder'e ait olan Embedding ve Positional Encoding katmanları
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model)

        # Belirtilen sayıda encoderlayer'ı bir liste içinde oluşturuyoruz
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)] 

        # Embedding'lere uygulanacak olan ana dropout katmanı.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        def call(self, inputs, training):
            seq_len = tf.shape(inputs)[1]

            # 1. Girdi ID'lerini embedding vektörlere çevir
            x = self.embedding(inputs)
            # 2. Embedding'lere konumsal bilgi ekle
            x = self.pos_encoding(x)
            # 3. dropout uygula
            x = self.dropout(x, training = training)

            # 4. Veriyi sırayla tüm encoder katmanlarından geçir.
            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training)

            return x

# Decoder için kısım
class CausalSelfAttention (tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(CausalSelfAttention, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = d_model // num_heads,
            use_causal_mask = True
        )
        self.dense_layer = tf.keras.layers.Dense(d_model)
    
    def call(self, inputs):
        attention_output = self.attention(
            query = inputs,
            value = inputs,
            key = inputs
        )
        return self.dense_layer(attention_output)

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = d_model // num_heads
        )
        self.dense_layer = tf.keras.layers.Dense(d_model)

    def call(self, inputs, context):

        attention_output = self.attention(
            query = inputs,
            key = context,
            value = context
        )

        return self.dense_layer(attention_output)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate = 0.1):
        super(DecoderLayer, self).__init__()

        # --- Alt katmanlar ---
        # 1. katman: Maskeli Öz-dikkat katmanı
        self.causal_self_attention = CausalSelfAttention(d_model, num_heads)

        # 2. katman: Çarpraz dikkat katmanı (encoder-decoder arasında ilk haberleşilen katmandır)
        self.cross_attention = CrossAttention(d_model, num_heads)

        # 3. katman: Feed Forward katmanı
        self.ffn = PositionWiseFeedForwardNetwork(d_model, dff)
        
        #her katmanın arasına ve en son çıkışa normalizasyon ve dropout konur
        self.layernorm1 = tf.keras.layers.Layerormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.Layerormalization(epsilon = 1e-6)
        self.layernorm3 = tf.keras.layers.Layerormalization(epsilon = 1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, context, training):

        # Maskeli öz dikkat katmanı
        attn1 = self.causal_self_attention(inputs)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + inputs)

        # Cross-attention katmanı
        attn2 = self.cross_attention(out1)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-Forward katmanı
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers

        # Decoder'ın hedef metin için kendi embedding ve positional encoding katmanları var
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        #num_layers kadar decoderLayeri bir liste içerisinde oluşturuyoruz
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, context, training):
        seq_len = tf.shape(inputs)[1]

        # embedding, positional encoding ve dropout katmanlarına sokup sonrasında
        # decodere sokuyoruz

        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context, training)

        
        return x


# --- TRANSFORMER ---
# Şu ana kadar oluşturduğumuz encoder ve decoderleri birleştirerek
# amaçladığımız transformer katmanını oluşturduk.

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_pos_input, max_pos_target, dropout_rate = 0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_pos_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_pos_target, dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self,inputs,training):

        # Bu elimizdeki inputs değişkeni bir tupledir ve bu kodun anlamı
        # tupleyi aç ve ilk değişkeni enc_inputa,
        # ikinci değişkeni dec_inputa ver.
        enc_input, dec_input = inputs  
        # hatalı metinleri encoderden geçiriyoruz
        enc_output = self.encoder(enc_input, training)
        # doğru metinleri ve encoderden gelen bağlamı decoderden geçiriyoruz
        dec_output = self.decoder(dec_input, enc_output, training)
        # decoderin çıktısını son katmandan geçiriyoruz
        final_output = self.final_layer(dec_output)

        # olasılığa çevirilmemiş nihai çıktı
        return final_output  

# HIPERPARAMETRELER
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# Veri Seti Bilgileri
input_vocab_size = len(input_vectorizer.get_vocabulary())
target_vocab_size = len(target_vectorizer.get_vocabulary())


# Model
transformer = Transformer(
    num_layers= num_layers,
    d_model = d_model,
    num_heads= num_heads,
    dff= dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_pos_input=MAX_SEQ_LENGTH, # encoder için max pozisyon bilgisi
    max_pos_target=MAX_SEQ_LENGTH + 1, # decoder için max pozisyon bilgisi
)

# Optimzer "Adam" fakat bunu biraz değiştirip önce yavaşca artan bir learning_rate
# ardından yavaşca düşen bir learning_rate haline getiriyoruz.

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model)*tf.main.minimum(arg1, arg2)
    

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta2=0.98, epsilon=1e-9)

# Kayıp fonksiyonları
# elimizdeki veride padding uygulanmış olduğu yani boş olan yerleri
# 0 ile doldurarak işlem yapmış olduğumuz için eğer model bun 0 ları
# loss func içine sokarsa loss büyür ve bu yüzden model cezalandırılır.
# bunun olmaması için özel bir fonksiyon tanımlıyoruz.

# normalde loss sonucu ortalama verir
# fakat biz  reduction = none kullanarak bunu engelliyoruz ve bize
# ortalama alınmamış çıktı veriyor ki paddingleri silebilelim




loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits =True, reduction = 'none'
)

#paddingi yok sayan özel fonksiyon
def loss_function(real, pred):
    # real içerisinde ki o veri 0 a eşit mi değil mi onu true false olarak döndürür
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    # bütün realin lossunu hesaplarız
    loss_ = loss_object(real, pred)

    # Maskeyi True false durumundan 1 0 durumuna getirir
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Loss'u mask ile çarparak padding pozisyonlarını sıfırlar
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# Normal şartlarda kerasda model.fit() komutu ile eğitim yapılabilir fakat
# bizim gibi karmaşık transformer kullanımlarında bu süreci
# tf ile yaparak daha ayrıntılı ve kontrol edilebilir şekilde yazmak
# bu konuyu yeni öğrenen benim için daha iyi olacaktır

# Bu fonksiyon tekrar tekrar çağırılacak ana işlevdir. 
# Bir batch veri ile modelin nasıl öğrenim yapacağını tanımlayan tek bir adımdır
# @tf.function, bu fonksiyonu hızlı bir TensorFlow grafiğine derler.
@tf.function
def train_step(encoder_input, target):
    # encoder_input: hatalı metinlerin ID'lerinden oluşan tensör.
    # target: Doğru metinlerin ID'lerinden oluşan tensör.

    # Decodere girdi olarak, hedef cümlenin son token'i hariç her şeyi veririz (<end>)
    dec_input = target[:, :-1]

    # Loss hesaplanırken karşılaştıracağımız gerçek hedef ise, ilk tokensiz halidir. (<start>)
    real_target = target[:, 1:]

    # tf.GradientTape(), içine yazılan tüm işlemleri kaydeder ve gradyan hesaplamak için kullanır.

    with tf.GradientTape() as tape:
        # 1. İleri besleme (Forward Pass) : Modeli çalıştırıp tahminlerin alınması
        # Girdiyi (hatalı metin, doğru metin) modele veriyoruz.
        predictions = transformer((encoder_input, dec_input), training = True)

        # 2. Kayıp (loss) hesabı: tahminler ile gerçek değerler arasındaki hata ölçülür
        # daha önce yazdığımız özel loss fonksiyonu kullanılır.
        loss = loss_function(real_target, predictions)



    # --- Geri yayılım (Backward Pass) ---
    # 3. Gradyanları hesapla: Kaydedilen işlemleri kullanarak, loss'u minimalize etmek için
    # modelin ağırlıklarının hangi yönde ne kadar değişmesi gerektiğini hesaplar.
    gradients = tape.gradient(loss, transformer.trainable_veriables)

    # 4. Ağırlıkları Güncelle: Optimizer'a bu gradyanları vererek modelin ağırlıklarını
    # öğrenme adımını gerçekleştirmesini söyleriz.
    # öğrenme kısmı burada gerçekleşir

    optimizer.apply_gradients(zip(gradients, transformer.trainable_veriables))

    #Bilgilendirme için o anki batch'in loss'unu ve doğruluğunu döndür.
    return loss

# Epochs normal bir veri seti üzerinden kaç kere geçileceğini belirler
EPOCHS = 20

# Eğitim döngüsünü başlat
for epoch in range(EPOCHS):
    start = time.time()

    #epoch başında toplam lossu sıfırla
    total_loss = 0

    # Veri setindeki her bir batch için döngüye gir.
    # dataset değişkeni daha önce tf.dataset.Dataset.from_tensor_slices
    # ile hazırlanan veri setidir diyor ama anlamadım.

    for(batch,( enc_input, target)) in enumerate(dataset):
        batch_loss = train_step(enc_input, target)
        total_loss += batch_loss


        # her 50 batchte bir ilerleme durumunu yazdır
        if batch%50 == 0:
            print(f'Epoch {epoch + 1} batch {batch} Loss {batch_loss.numpy():.4f}')

    # Epoch tamamlandıpında özet yazdırılması
    print(f'Epoch {epoch + 1} Loss {total_loss/(batch + 1):.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')









