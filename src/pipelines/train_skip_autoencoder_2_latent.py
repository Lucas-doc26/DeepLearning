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
    parser.add_argument('--epochs', type=int, default=1, help='Número de épocas para o treinamento.')
    args = parser.parse_args()

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train = preprocess_dataset(df_train, batch_size=32, autoencoder=True, transform=salt_and_pepper(), prob=1)

    df_valid = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_valid.csv')
    valid = preprocess_dataset(df_valid, batch_size=32, autoencoder=True)

    df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_test.csv')
    test = preprocess_dataset(df_test[:32], batch_size=32, autoencoder=True)

    print(len(train), len(valid), len(test))
    print(type(train), type(valid), type(test))

    model_base = SkipAutoencoder2Latent()
    model = model_base.model
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train, epochs=args.epochs, validation_data=valid)

    os.makedirs('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights', exist_ok=True)
    model.save('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/skip_autoencoder_2_latent.keras')
    model.save_weights(f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights/skip_autoencoder_2_latent-{args.train}.weights.h5')
    encoder = model_base.encoder 
    encoder.save_weights(f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights/skip_autoencoder_2_latent-encoder-{args.train}.weights.h5')

    plot_autoencoder_with_ssim(test, model,
                               save_path=f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/plots/autoencoder_reconstruction/skip_autoencoder_2_latent_{args.train}.png')
    
