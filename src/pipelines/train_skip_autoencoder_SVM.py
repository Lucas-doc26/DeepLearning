import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from utils import preprocess_dataset, dataset_generator
from models import SkipAutoencoderSVM

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Train a Skip Autoencoder SVM model.')
    parser.add_argument('skip', type=str, help='Dataset do skip.')
    parser.add_argument('--train', type=str, default='camera1', help='Dataset de treino.')
    parser.add_argument('--test', type=str, default='camera2', help='Dataset de teste.')
    args = parser.parse_args()

    model = SkipAutoencoderSVM(
        '/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder.keras',
        f'/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder-{args.skip}.weights.h5'
    )   

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}.csv')
    train_ds = preprocess_dataset(df_train[:1000], batch_size=32, autoencoder=False)

    for x_batch, y_batch in dataset_generator(train_ds):
        model.train(x_batch, y_batch)

    df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.test}.csv')
    test_ds = preprocess_dataset(df_test[:1000], batch_size=32, autoencoder=False)

    preds = []
    y_true = []

    for x_batch, y_batch in dataset_generator(test_ds):
        batch_pred = model.test(x_batch)
        preds.extend(batch_pred)
        y_true.extend(y_batch)

    y_true = np.array(y_true)
    preds = np.array(preds)
    accuracy = (y_true == preds).mean()

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
