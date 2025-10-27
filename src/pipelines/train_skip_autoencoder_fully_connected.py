import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from models import SkipAutoencoderFullyConnected
from utils import preprocess_dataset, salt_and_pepper

model = SkipAutoencoderFullyConnected('/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder.keras',
          '/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder-cnr.weights.h5')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

df_train = pd.read_csv('/home/lucas/DeepLearning/CSV/CNR/train.csv')
train = preprocess_dataset(df_train[:1000], batch_size=32, autoencoder=False)

df_valid = pd.read_csv('/home/lucas/DeepLearning/CSV/CNR/val.csv')
valid = preprocess_dataset(df_valid[:64], batch_size=32, autoencoder=False)

model.fit(train, epochs=20, validation_data=valid, batch_size=32)

model.save('/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder_fully_connected.keras')
model.save_weights('/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder_fully_connected-cnr.weights.h5')

df_test = pd.read_csv('/home/lucas/DeepLearning/CSV/CNR/test.csv')
test = preprocess_dataset(df_test[1064:], batch_size=32, autoencoder=False)

preds = model.predict(test)
y_true = df_test['class'][1064:].values
y_pred = preds.argmax(axis=1)

accuracy = (y_true == y_pred).mean()
print(f'Test Accuracy: {accuracy * 100:.2f}%')      

