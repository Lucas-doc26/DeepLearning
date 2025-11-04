import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import SkipAutoencoderFullyConnected2, classifier_log
from utils import preprocess_dataset, salt_and_pepper, plot_confusion_matrix, create_kyoto, randon_rain
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
    args = parser.parse_args()

    model = SkipAutoencoderFullyConnected2('/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder.keras',
            f'/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder-encoder-{args.autoencoder}.weights.h5')
    
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train = preprocess_dataset(df_train, batch_size=32, autoencoder=False, transform=randon_rain())

    df_valid = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_valid.csv')
    valid = preprocess_dataset(df_valid, batch_size=32, autoencoder=False)

    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=15,          
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',    
        factor=0.5,            
        patience=3,
        min_lr=1e-7
    )

    model_checkpoint = ModelCheckpoint(
        filepath='/home/lucas/DeepLearning/models/skip_autoencoder2/weights/best_fc_model.h5',
        monitor='val_accuracy',    # ou 'val_accuracy'
        save_best_only=True
    )
    os.makedirs('/home/lucas/DeepLearning/models/skip_autoencoder2/',exist_ok=True )

    inicial = time.time()
    model.train_model(train, valid, args.epochs, callbacks=[model_checkpoint, reduce_lr, early_stop])
    final = time.time()

    model.model.save('/home/lucas/DeepLearning/models/skip_autoencoder2/skip_autoencoder2_fully_connected.keras')
    model.model.save_weights(f'/home/lucas/DeepLearning/models/skip_autoencoder2/weights/skip_autoencoder2_fully_connected-{args.train}.weights.h5')

    results = []
    for dataset_test in args.test:
        df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{dataset_test}/{dataset_test}_test.csv')
        test = preprocess_dataset(df_test[:1000], batch_size=32, autoencoder=False)


        preds = model.test_model(test)
        y_true = df_test['class'][:1000].values
        y_pred = preds.argmax(axis=1)

        cm_path = f'/home/lucas/DeepLearning/models/skip_autoencoder2/plots/confusion_matrix/{args.autoencoder}/{args.train}/FC/{args.autoencoder}-{args.train}-{dataset_test}.png'
        plot_confusion_matrix(y_true, y_pred, labels=args.labels,
                            legend=f"Autoencoder: {args.autoencoder} - Treino: {args.train} x Teste:{dataset_test}",
                            save_path=cm_path)
        

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([{"Accuracy": accuracy}, {"Precision":precision}, {"Recall":recall}, {"F1":f1}])

    log_dir = f'/home/lucas/DeepLearning/models/skip_autoencoder2/logs'
    classifier_log(log_dir=log_dir, model_name=f'skip_autoencoder2_fully_connected-{args.train}', 
                    input_shape=128, autoencoder_description=128,
                    optimizer='adam', loss_fn='sparse_categorical_crossentropy',
                    train_info={"Train": args.train,
                                "Time to train": final - inicial,
                                "Epochs":args.epochs,
                                "Test": {dataset: result for dataset, result in zip(args.test, results)}}
                )