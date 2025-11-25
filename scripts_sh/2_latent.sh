#!/bin/bash

# -------- ID 1 --------
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train PKLot  --epochs 100 --id 1 --id 1
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train kyoto  --epochs 100 --id 1 --id 1
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train CNR    --epochs 100 --id 1 --id 1

# -------- ID 2 --------
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train PKLot  --epochs 10 --id 2 --id 2
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train kyoto  --epochs 10 --id 2 --id 2
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train CNR    --epochs 10 --id 2 --id 2

# -------- ID 3 --------
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train PKLot  --epochs 10 --id 3 --id 3
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train kyoto  --epochs 10 --id 3 --id 3
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train CNR    --epochs 10 --id 3 --id 3

# -------- ID 4 --------
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train PKLot  --epochs 10 --id 4 --id 4
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train kyoto  --epochs 10 --id 4 --id 4
python /home/lucas/DeepLearning/src/pipelines/train_skip_autoencoder_2_latent.py --train CNR    --epochs 10 --id 4 --id 4

