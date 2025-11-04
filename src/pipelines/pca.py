import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



# Caminho do encoder pré-treinado
encoder_path = '/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder.keras'
weights_path = '/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder-CNR.weights.h5'

# Carrega o encoder
from models.skip_autoencoder.skip_autoencoder import SkipAutoencoder
from utils import preprocess_dataset
skip = SkipAutoencoder()
skip.load(model_path=encoder_path)
encoder = skip.return_encoder()
encoder.load_weights(weights_path, skip_mismatch=True)

# Isola o vetor latente (primeiro tensor)
encoder_latent = tf.keras.Model(encoder.input, encoder.output[0])

# Carrega dataset de teste (exemplo com PUC)
df_test = pd.read_csv('/home/lucas/DeepLearning/CSV/PUC/PUC_test.csv')
train_ds = preprocess_dataset(df_test[:1024], batch_size=32, autoencoder=True)
y = df_test['class'][:1024].values

# Extrai embeddings
embeddings = encoder_latent.predict(train_ds, batch_size=32)

# Reduz dimensionalidade
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)

# (ou t-SNE, se quiser algo mais visual)
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
# tsne_result = tsne.fit_transform(embeddings)

# Plot PCA
plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0], pca_result[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.title('Distribuição dos embeddings (PCA)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(label='Classe (0=Vazio, 1=Ocupado)')
plt.show()
plt.savefig("teste.png")
