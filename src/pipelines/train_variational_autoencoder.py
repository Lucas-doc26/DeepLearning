import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models import VariationalAutoencoder, save_autoencoder_log

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Train a Variational Autoencoder.')
    parser.add_argument('--train', type=str,default='CNR', help='Dataset do skip.')
    parser.add_argument('--epochs', type=int, default=1, help='Número de épocas para o treinamento.')
    args = parser.parse_args()

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train = preprocess_dataset(df_train, batch_size=32, autoencoder=True, transform=salt_and_pepper(), prob=0.4)

    df_valid = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_valid.csv')
    valid = preprocess_dataset(df_valid, batch_size=32, autoencoder=True)

    df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_test.csv')
    test = preprocess_dataset(df_test[:32], batch_size=32, autoencoder=True)

    print(len(train), len(valid), len(test))

    model = VariationalAutoencoder(latent_dim=128)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train, epochs=args.epochs, validation_data=valid, batch_size=32)
    model.save_model('/home/lucas/DeepLearning/models/variational_autoencoder/', 'variational_autoencoder')
    model.save_weights('/home/lucas/DeepLearning/models/variational_autoencoder/weights', 'variational_autoencoder', f'{args.train}') 

    plot_autoencoder_with_ssim(test, model,
                               save_path=f'/home/lucas/DeepLearning/models/variational_autoencoder/plots/autoencoder_reconstruction/variational_autoencoder_{args.train}.png')
    

    pred = model.predict(test)
    metrics_image = calculate_ssim(next(iter(test))[0], next(iter(pred))[0])
    print(metrics_image)

    metrics_test = {
        "batch_size": 32,
        "base": args.train,
        "epochs": args.epochs,
        "metrics_image": metrics_image
    }

    save_autoencoder_log(
        log_dir='/home/lucas/DeepLearning/models/variational_autoencoder/logs',
        model_name="Skip Autoencoder",
        input_shape=(128, 128, 3),
        latent_dim=128,
        optimizer="Adam",
        loss_fn="Mean Squared Error",
        metrics=["accuracy"],
        train_info=metrics_test
    )
