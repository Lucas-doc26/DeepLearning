import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import preprocess_dataset, dataset_generator
from models import SkipAutoencoderSVM, classifier_log
from utils import preprocess_dataset, salt_and_pepper, plot_confusion_matrix, create_kyoto
import random
import time

random.seed(42)

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Train a Skip Autoencoder SVM model.')
    parser.add_argument('--autoencoder', type=str,default='CNR', help='Dataset do skip.')
    parser.add_argument('--train', type=str,default='UFPR05', help='Dataset do skip.')
    parser.add_argument('--test', type=str,nargs='+', default=['PUC'], help='Dataset do skip.')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para o treinamento.')
    parser.add_argument('--labels', type=list, default=['Vazio', 'Ocupado'], help='Labels de classificação')
    args = parser.parse_args()

    model = SkipAutoencoderSVM(
        '/home/lucas/DeepLearning/models/skip_autoencoder/skip_autoencoder.keras',
        f'/home/lucas/DeepLearning/models/skip_autoencoder/weights/skip_autoencoder-encoder-{args.autoencoder}.weights.h5'
    )   

    df_train = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.train}/{args.train}_train.csv')
    train_ds = preprocess_dataset(df_train[:1024], batch_size=32, autoencoder=False, transform=salt_and_pepper())

    inicial = time.time()
    for x_batch, y_batch in dataset_generator(train_ds):
        print(x_batch.shape)
        model.train(x_batch, y_batch)
    final = time.time()

    results = []
    for dataset_test in args.test:
        df_test = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{dataset_test}/{dataset_test}_test.csv')
        test = preprocess_dataset(df_test, batch_size=32, autoencoder=False)

        y_pred = []
        y_true = []

        for x_batch, y_batch in dataset_generator(test):
            batch_pred = model.test(x_batch)
            y_pred.extend(batch_pred)
            y_true.extend(y_batch)

        cm_path = f'/home/lucas/DeepLearning/models/skip_autoencoder/plots/confusion_matrix/{args.autoencoder}/{args.train}/SVM/{args.autoencoder}-{args.train}-{dataset_test}.png'
        plot_confusion_matrix(y_true, y_pred, labels=args.labels,
                            legend=f"Autoencoder: {args.autoencoder} - Treino: {args.train} x Teste:{dataset_test}",
                            save_path=cm_path)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([{"Accuracy": accuracy}, {"Precision":precision}, {"Recall":recall}, {"F1":f1}])


    log_dir = f'/home/lucas/DeepLearning/models/skip_autoencoder/logs'
    classifier_log(log_dir=log_dir, model_name=f'skip_autoencoder_SVM-{args.train}', 
                    input_shape=128, autoencoder_description=128,
                    optimizer='adam', loss_fn='sparse_categorical_crossentropy',
                    train_info={"Base Autoencoder":args.autoencoder,
                                "Train": args.train,
                                "Time to train": final - inicial,
                                "Epochs":args.epochs,
                                "Test": {dataset: result for dataset, result in zip(args.test, results)}}
                )