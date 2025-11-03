import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import VariationalAutoencoderFullyConnected, classifier_log
from utils import preprocess_dataset, salt_and_pepper, plot_confusion_matrix
import random
import time

tf.random.set_seed(42)
random.seed(42)

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Treinando uma svm encima do VAE.')
    parser.add_argument('--autoencoder', type=str,default='CNR', help='Dataset do autoencoder.')
    parser.add_argument('--train', type=str,default='UFPR05', help='Dataset do autoencoder.')
    parser.add_argument('--test', type=str,nargs='+', default=['PUC'], help='Dataset do autoencoder.')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para o treinamento.')
    parser.add_argument('--labels', type=list, default=['Vazio', 'Ocupado'], help='Labels de classificação')
    args = parser.parse_args()

    model = VariationalAutoencoderFullyConnected(model_path='/home/lucas/DeepLearning/models/variational_autoencoder/variational_autoencoder.keras',
            model_weights=f'/home/lucas/DeepLearning/models/variational_autoencoder/weights/variational_autoencoder-{args.autoencoder}.weights.h5')

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train = preprocess_dataset(df_train[:1024], batch_size=32, autoencoder=False, transform=salt_and_pepper())

    df_valid = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_valid.csv')
    valid = preprocess_dataset(df_valid[:64], batch_size=32, autoencoder=False)

    inicial = time.time()
    model.fit(train, epochs=args.epochs, validation_data=valid, batch_size=32)
    final = time.time()

    model.save('/home/lucas/DeepLearning/models/variational_autoencoder/variational_autoencoder_fully_connected.keras')
    model.save_weights(f'/home/lucas/DeepLearning/models/variational_autoencoder/weights/variational_autoencoder_fully_connected-{args.test}.weights.h5')

    results = []
    for dataset_test in args.test:
        df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{dataset_test}/{dataset_test}_test.csv')
        test = preprocess_dataset(df_test, batch_size=32, autoencoder=False)

        preds = model.predict(test)
        y_true = df_test['class'].values
        y_pred = preds.argmax(axis=1)


        cm_path = f'/home/lucas/DeepLearning/models/variational_autoencoder/plots/confusion_matrix/{args.autoencoder}/{args.train}/FC/{args.autoencoder}-{args.train}-{dataset_test}.png'
        plot_confusion_matrix(y_true, y_pred, labels=args.labels,
                            legend=f"Autoencoder: {args.autoencoder} - Treino: {args.train} x Teste:{dataset_test}",
                            save_path=cm_path)
        

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([{"Accuracy": accuracy}, {"Precision":precision}, {"Recall":recall}, {"F1":f1}])

    log_dir = f'/home/lucas/DeepLearning/models/variational_autoencoder/logs'
    classifier_log(log_dir=log_dir, model_name='variational_autoencoder_fully_connected', 
                    input_shape=128, autoencoder_description=128,
                    optimizer='adam', loss_fn='sparse_categorical_crossentropy',
                    train_info={"Train": args.train,
                                "Time to train": final - inicial,
                                "Epochs":args.epochs,
                                "Test": {dataset: result for dataset, result in zip(args.test, results)}}
                )