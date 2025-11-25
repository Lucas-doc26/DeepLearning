import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import h5py

from sklearn.metrics import accuracy_score
from models import *
from utils.preprocess import *

data1 = np.load('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/1-CNR-PUC-latent.npy')
data2 = np.load('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/2-CNR-PUC-latent.npy')
data3 = np.load('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/3-CNR-PUC-latent.npy')
data4 = np.load('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/4-CNR-PUC-latent.npy')

stacked = np.stack([data1, data2, data3, data4], axis=0)
x = np.mean(stacked, axis=0)

data = pd.read_csv('/home/lucas/DeepLearning/CSV/PUC/PUC_train.csv')

y = data['class'].values


model = Sequential([
            Input(shape=(100,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax') 
        ], name='SkipAutoencoderFullyConnected')


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(x, y, epochs=10)

with h5py.File('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/1-CNR-PUC-latent-test.h5', 'r') as f:
    test1 = f['latents'][:]   # carrega o dataset inteiro

with h5py.File('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/2-CNR-PUC-latent-test.h5', 'r') as f:
    test2 = f['latents'][:]

with h5py.File('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/3-CNR-PUC-latent-test.h5', 'r') as f:
    test3 = f['latents'][:]

with h5py.File('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds/4-CNR-PUC-latent-test.h5', 'r') as f:
    test4 = f['latents'][:]

stacked_test = np.stack([test1, test2, test3, test4], axis=0)
x_test = np.mean(stacked_test, axis=0)

preds = model.predict(x_test)

test = pd.read_csv('/home/lucas/DeepLearning/CSV/PUC/PUC_test.csv')
mapping = {'0': 0, '1': 1}

y_test = test['class'].astype(str).map(mapping).astype(float).values

y_pred = np.argmax(preds, axis=1)

print(accuracy_score(y_test, y_pred))
