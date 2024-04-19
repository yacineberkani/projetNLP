# Import des bibliothèques
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
import emoji
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import sys
from unidecode import unidecode
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.metrics import Precision, Recall

# Initialisez le tokenizer
tknzr = TweetTokenizer()
# Initialisez le lemmatizer
lemmatizer = WordNetLemmatizer()
# Définissez les mots vides en anglais à supprimer
stop_words = set(stopwords.words('english')) - set(('not', 'no', 'never', 'nor'))

# Fonction pour standardiser le tweet
def standardization(tweet):
    tweet = re.sub(r"\u2019", "'", tweet)
    tweet = re.sub(r"\u002C", ",", tweet)
    tweet = ' '.join(str2emoji(unidecode(tweet).lower().split()))
    tweet = re.sub(r"(http|https)?:\/\/[a-zA-Z0-9\-.]+\.[a-zA-Z]{2,4}(/\S*)?", " ", tweet)
    tweet = re.sub(r"\bve\b", "have", tweet)  # Correction: remove extra 're' in 've'
    tweet = re.sub(r"\bcan't\b", "cannot", tweet)
    tweet = re.sub(r"\b'n't\b", " not", tweet)
    tweet = re.sub(r"\bre\b", "are", tweet)  # Correction: remove extra 're' in 'are'
    tweet = re.sub(r"\bd\b", "would", tweet)  # Correction: remove extra 'd' in 'would'
    tweet = re.sub(r"\b'll\b", "will", tweet)
    tweet = re.sub(r"\b's\b", " ", tweet)
    tweet = re.sub(r"\bn\b", " ", tweet)
    tweet = re.sub(r"\bm\b", "am", tweet)
    tweet = re.sub(r"@\w+", " ", tweet)
    tweet = re.sub(r"#\w+", " ", tweet)
    tweet = re.sub(r"[0-9]+", " ", tweet)
    tweet = tknzr.tokenize(tweet)
    tweet = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i, j in pos_tag(tweet)]
    tweet = [i for i in tweet if (i.lower() not in stop_words) and (i.lower() not in punctuation)]
    tweet = ' '.join(tweet)
    return tweet

# Fonction pour convertir les émojis en texte
def str2emoji(tweet):
    for pos,ej in enumerate(tweet):
        if ej in emojis:
            tweet[pos]=emojis[ej]
    return tweet

# Dictionnaire des émojis avec leurs équivalents textuels
emojis = {
    u":)": "🙂",
    u":-)": "🙂",
    u":]": "🙂",
    u":-3": "🙂",
    u":3": "🙂",
    u":->": "🙂",
    u":>": "🙂",
    u"8-)": "🙂",
    u":o)": "🙂",
    u":-}": "🙂",
    u":}": "🙂",
    u":-)": "🙂",
    u":c)": "🙂",
    u":^)": "🙂",
    u"=]": "🙂",
    u"=)": "🙂",
    u":(": "🙁",
    u":-(": "🙁",
    u":[": "🙁",
    u">:[": "🙁",
    u":{": "🙁",
    u">:(": "🙁",
    u":'(": "😢",
    u":'-(": "😢",
    u"D:": "😨",
    u":o": "😮",
    u":-o": "😮",
    u"o_O": "😮",
    u"o.O": "😮",
    u"o_o": "😮",
    u"0_o": "😮",
    u"T_T": "😭",
    u">_<": "😣",
    u"O_O": "😳",
    u"^_^": "😊",
    u"(^_^)": "😊",
    u"(-_-)": "😑",
    u"(._.)": "😐",
    u"(>_<)": "😣",
    u"(^o^)": "😃",
    u"(^O^)": "😃",
    u"(^0^)": "😃",
    u"(@_@)": "😵",
    u"(+o+)": "😵",
    u"(*_*)": "😲",
    u"(^_^)/": "😊",
    u"(>o<)": "😣",
    u"(-_\\": "😑",
    u"(~_~)": "😑",
    u"(-.-)zzz": "😴",
    u"(^_-)": "😉",
    u"((+_+))": "😵",
    u"(+o+)": "😵",
    u"(-)__(-)": "😑",
}

# Fonction pour prétraiter les données
def data_preprocessing(path_tweets):
    data = pd.read_csv(path_tweets, encoding='utf-8', sep='\t')
    data['tweet'] = data['Tweet'].apply(lambda x: standardization(x))
    data['label'] = data['Intensity Class'].str.split(":").str[0]
    return data['tweet'], data['label']

# Lecture des données d'entraînement et prétraitement
df_3 = pd.read_csv("twitter_training.csv", delimiter=',', names=['ID', 'Game', 'label', 'tweet'])
df_7 = "2018-Valence-oc-En-train.txt"

df_3.drop(columns=['ID', 'Game'], inplace=True)
df_3.dropna(inplace=True)
df_3 = df_3[df_3['label'] != 'Irrelevant']

label_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df_3.loc[:, 'label'] = df_3['label'].replace(label_mapping)

tweets_train_3 = df_3['tweet'].apply(lambda x: standardization(x))
sentiments_train_3 = df_3['label']

tweets_train_7, sentiments_train_7 = data_preprocessing(df_7)
all_tweets = pd.concat([tweets_train_3, tweets_train_7], ignore_index=True)

# Initialisez le tokenizer et ajustez-le sur l'ensemble des tweets
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(all_tweets)

# Convertissez les tweets en séquences numériques
sequences_train_3 = tokenizer.texts_to_sequences(tweets_train_3)
sequences_train_7 = tokenizer.texts_to_sequences(tweets_train_7)

# Déterminez la longueur maximale d'une séquence
MAX_SEQUENCE_LENGTH = max(map(len, sequences_train_3 + sequences_train_7))

# Remplissez les séquences pour qu'elles aient toutes la même longueur
data_train_3 = pad_sequences(sequences_train_3, maxlen=MAX_SEQUENCE_LENGTH)
data_train_7 = pad_sequences(sequences_train_7, maxlen=MAX_SEQUENCE_LENGTH)

# Mélangez et divisez les données en ensembles d'entraînement et de validation
indices_train_3 = np.arange(data_train_3.shape[0])
np.random.shuffle(indices_train_3)
data_train_3 = data_train_3[indices_train_3]

indices_train_7 = np.arange(data_train_7.shape[0])
np.random.shuffle(indices_train_7)
data_train_7 = data_train_7[indices_train_7]

# Convertissez les étiquettes en catégories
labels_train_3 = to_categorical(np.asarray(sentiments_train_3), 3)
labels_train_3 = labels_train_3[indices_train_3]

labels_train_7 = to_categorical(np.asarray(sentiments_train_7), 7)
labels_train_7 = labels_train_7[indices_train_7]

# Séparez les ensembles d'entraînement et de validation
split_idx_7 = int(len(data_train_7) * 0.85)
split_idx_3 = int(len(data_train_3) * 0.85)
x_train_3, x_val_3 = data_train_3[:split_idx_3], data_train_3[split_idx_3:]
y_train_3, y_val_3 = labels_train_3[:split_idx_3], labels_train_3[split_idx_3:]

x_train_7, x_val_7 = data_train_7[:split_idx_7], data_train_7[split_idx_7:]
y_train_7, y_val_7 = labels_train_7[:split_idx_7], labels_train_7[split_idx_7:]

# Affichez l'index des mots
word_index = tokenizer.word_index
print(word_index)

# Initialisez la dimension de l'incorporation et le fichier d'incorporation pré-entraîné
nb_words = len(word_index) + 1
EMBEDDING_DIM = 300
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"

# Initialisez la matrice d'incorporation
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

# Chargez les vecteurs de mots pré-entraînés
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Remplissez la matrice d'incorporation avec les vecteurs de mots pré-entraînés
for word, i in word_index.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]
    else:
        # Remplacez les mots non trouvés par des vecteurs aléatoires
        embedding_matrix[i] = np.random.rand(EMBEDDING_DIM) * 2.0 - 1.0

# Définissez la structure du modèle
model = Sequential()
model.add(Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(3, activation='softmax'))

# Compilez le modèle
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Entraînez le modèle
history = model.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3), batch_size=64, epochs=25)

# Affichez un résumé du modèle
model.summary()

# Évaluer le modèle sur l'ensemble de test

accuracy= model.evaluate(data_test_3, labels_test_3 , verbose=0)

print('Accuracy  : {:.4f}'.format(accuracy[1]))

# Enregistrez le modèle
model.save("model_3C.h5")

# Chargez le modèle entraîné sur 3 classes
model2 = load_model('model_3C.h5')
# Supprimez les couches supplémentaires
model2.layers.pop()
model2.layers.pop()
# Ajoutez de nouvelles couches pour prédire 7 classes
model2.add(Dense(150, activation='relu', name='dense11'))
model2.add(Dense(64, activation='relu', name='dense21'))
model2.add(Dense(7, activation='softmax', name='dense31'))
model2.summary()
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
# Entraînez le modèle sur 7 classes
history = model2.fit(x_train_7, y_train_7, validation_data=(x_val_7, y_val_7), epochs=30, batch_size=50)

# Enregistrez le modèle pour 7 classes
model2.save("model_7C.h5")

# Fonction pour prédire la classe d'un texte
def predict_class(text):
    sentiment_classes = sentiments_test
    max_len = 50
    # Transformez le texte en une séquence d'entiers à l'aide du tokenizer
    xt = tokenizer.texts_to_sequences(text)
    # Remplissez les séquences à la même longueur
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Prédisez la classe en utilisant le modèle chargé
    yt = model_7.predict(xt).argmax(axis=1)
    # Affichez la classe prédite
    print('The predicted sentiment is', sentiment_classes[yt[0]])

# Exemple de prédiction de classe pour un tweet
print(tweets_test[0])
print(sentiments_test[0])
predict_class(tweets_test[0])

# Exemple de prédiction de classe pour un autre tweet
print(tweets_test[1])
print(sentiments_test[1])
predict_class(tweets_test[1])

# Répétez le processus pour d'autres tweets
print(tweets_test[2])
print(sentiments_test[2])
predict_class(tweets_test[2])

print(tweets_test[3])
print(sentiments_test[3])
predict_class(tweets_test[3])

print(tweets_test[4])
print(sentiments_test[4])
predict_class(tweets_test[4])

