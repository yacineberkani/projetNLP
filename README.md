# Projet: Classification de tweets basée sur le transfert learning

Ce projet vise à développer un modèle de classification de sentiments à partir de tweets en utilisant des techniques de transfert learning avec des embeddings de mots pré-entraînés.

## Objectif

L'objectif principal de ce projet est de créer un modèle capable de prédire le sentiment associé à un tweet donné, en le classant dans l'une des catégories suivantes : positif, neutre ou négatif. Le modèle sera développé en utilisant des embeddings de mots pré-entraînés, obtenus à partir de Google News Word2Vec, pour améliorer les performances de classification.

## Dépendances

Ce projet nécessite les dépendances suivantes :

- `numpy`
- `tensorflow`
- `pandas`
- `emoji`
- `nltk`
- `unidecode`
- `gensim`
- `keras`

Pour installer ces dépendances, vous pouvez utiliser le fichier `requirements.txt` :


pip install -r requirements.txt




Une fois le modèle entraîné, vous pouvez l'utiliser pour prédire le sentiment d'un tweet en utilisant le script predict_sentiment.py.

## Références

- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation NLTK](https://www.nltk.org/)
- [Documentation Gensim](https://radimrehurek.com/gensim/)
- [Documentation Keras](https://keras.io/)