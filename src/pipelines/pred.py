import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import h5py   # precisa estar instalado: pip install h5py
from models import SkipAutoencoder2Latent
from utils import preprocess_dataset

tf.random.set_seed(42)
random.seed(42)

def get_inputs_from_batch(batch):
    """
    Retorna os inputs que o modelo espera a partir do elemento do dataset.
    Trata batches que são:
      - tensor/np.array (somente X)
      - tuple/list (X, ...), devolve o primeiro elemento
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Predict latent vectors using Skip Encoder.')
    parser.add_argument('--train', type=str, default='PUC')
    parser.add_argument('--test', type=str, default='PUC')
    parser.add_argument('--id', type=int, default=1)
    args = parser.parse_args()

    train_df = pd.read_csv(f'/home/lucas/DeepLearning/CSV/{args.test}/{args.test}_test.csv')

    pred_ds = preprocess_dataset(train_df, batch_size=64, autoencoder=False)

    model = tf.keras.models.load_model(
        f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/skip_autoencoder_2_latent-{args.id}-encoder.keras'
    )
    model.load_weights(
        f'/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/weights/skip_autoencoder_2_latent-{args.id}-{args.train}-encoder.keras'
    )

    model.summary()

    outdir = '/home/lucas/DeepLearning/models/skip_autoencoder_2_latent/preds'
    os.makedirs(outdir, exist_ok=True)

    # vou salvar em HDF5 (extensível)
    file_pred = f'{outdir}/{args.id}-{args.train}-{args.test}-latent-test.h5'

    # -----------------------------------------------------
    # Previsão batch-a-batch e salvamento incremental (HDF5)
    # -----------------------------------------------------
    total_rows = 0
    latent_dim = None

    # abre arquivo HDF5 em modo append (cria se não existir)
    with h5py.File(file_pred, 'a') as h5f:
        # se já existe dataset "latents", recupera info
        if 'latents' in h5f:
            dset = h5f['latents']
            total_rows = dset.shape[0]
            latent_dim = dset.shape[1]
            print(f"Arquivo existente detectado. Já tem {total_rows} latents, dim = {latent_dim}")
        else:
            dset = None

        with tf.device('/GPU:0'):
            for i, raw_batch in enumerate(pred_ds):
                # extrai somente os inputs
                inputs = get_inputs_from_batch(raw_batch)

                # roda o modelo (garante que recebemos Tensor/np.array ou lista)
                preds = model(inputs, training=False)

                # Se o modelo retorna múltiplos outputs (lista/tuple),
                # assumimos que o latent é o segundo (index 1) como no seu exemplo.
                # Caso contrário, pega o próprio tensor.
                if isinstance(preds, (list, tuple)):
                    if len(preds) > 1:
                        latent_batch = preds[1]
                    else:
                        latent_batch = preds[0]
                else:
                    latent_batch = preds

                # converte para numpy
                if hasattr(latent_batch, 'numpy'):
                    latent_np = latent_batch.numpy()
                else:
                    latent_np = np.asarray(latent_batch)

                # primeira vez: cria dataset extensível
                if dset is None:
                    latent_dim = latent_np.shape[1]
                    # cria dataset com maxshape permitindo expandir na dimensão 0
                    maxshape = (None, latent_dim)
                    dset = h5f.create_dataset(
                        'latents',
                        data=latent_np,
                        maxshape=maxshape,
                        dtype='float32',
                        chunks=True  # chunks ajudam I/O incremental
                    )
                    total_rows = latent_np.shape[0]
                else:
                    # verifica dimensão
                    if latent_np.shape[1] != latent_dim:
                        raise RuntimeError(f"Dimensionalidade do latent mudou: antes {latent_dim}, agora {latent_np.shape[1]}")

                    # expande o dataset e escreve no final
                    new_total = total_rows + latent_np.shape[0]
                    dset.resize((new_total, latent_dim))
                    dset[total_rows:new_total, :] = latent_np
                    total_rows = new_total

                # força flush para disco
                h5f.flush()

                if (i + 1) % 10 == 0:
                    print(f"[batch {i+1}] Salvou. Total latents: {total_rows}")

    print(f"Predições salvas em: {file_pred}")
    print(f"Total de vetores latentes: {total_rows}")
