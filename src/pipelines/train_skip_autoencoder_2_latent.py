import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models import SkipAutoencoder2Latent, save_autoencoder_log

if __name__ == ("__main__"):
    import argparse
    parser = argparse.ArgumentParser(description='Train a Skip Autoencoder SVM model.')
    parser.add_argument('--train', type=str,default='CNR', help='Dataset do skip.')
    parser.add_argument('--test', type=str,default='PUC', help='Dataset do skip.')
    parser.add_argument('--epochs', type=int, default=1, help='Número de épocas para o treinamento.')
    parser.add_argument('--latent', type=int, default=-1, help='Tamanho do vetor latente')
    parser.add_argument('--id', type=int, default=1, help='Id')
    args = parser.parse_args()

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train = preprocess_dataset(df_train, batch_size=64, autoencoder=True, transform=salt_and_pepper(), prob=1)

    df_valid = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_valid.csv')
    valid = preprocess_dataset(df_valid, batch_size=64, autoencoder=True)

    df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_test.csv')
    test = preprocess_dataset(df_test[:32], batch_size=64, autoencoder=True)

    print(len(train), len(valid), len(test))
    print(type(train), type(valid), type(test))

    model_base = SkipAutoencoder2Latent(id=args.id)
    model = model_base.model
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train, epochs=args.epochs, validation_data=valid, batch_size=64)

    os.makedirs('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights', exist_ok=True)
    model_base.save_model('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent', name='skip_autoencoder_2_latent')
    model_base.save_weights('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights', name='skip_autoencoder_2_latent', train=args.train)
    plot_autoencoder_with_ssim(test, model,
                               save_path=f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/plots/autoencoder_reconstruction/skip_autoencoder_2_latent-id-{args.id}_{args.train}.png')
