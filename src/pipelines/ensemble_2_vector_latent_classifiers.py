import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import*
from utils import preprocess_dataset, salt_and_pepper, plot_confusion_matrix, create_kyoto, randon_rain
from utils.config import clear_session
import random
import time

tf.random.set_seed(42)
random.seed(42)

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Train a Skip Autoencoder SVM model.')
    parser.add_argument('--autoencoder', type=str,default='CNR', help='Dataset do skip.')
    parser.add_argument('--train', type=str,default='UFPR05', help='Dataset do skip.')
    parser.add_argument('--test', nargs='+', default=['PUC'], help='Dataset do skip.')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para o treinamento.')
    parser.add_argument('--labels', nargs='+' ,default=['Vazio', 'Ocupado'], help='Labels de classificação')
    parser.add_argument('--id', type=int)
    args = parser.parse_args()

    model = tf.keras.models.load_model(f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/skip_autoencoder_2_latent_fully_connected-id-{id}.keras')
    model.load_weights('/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/skip_autoencoder_2_latent_fully_connected-id{id}.keras')
    model.summary()

