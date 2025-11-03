import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================================
# 1. Carrega o dataset de áudio
# ==========================================================
ds, info = tfds.load('speech_commands', split='train', with_info=True, as_supervised=True)
print("Total de amostras:", info.splits['train'].num_examples)

# ==========================================================
# 2. Função para converter áudio em espectrograma (imagem)
# ==========================================================
def audio_to_spectrogram(audio, label):
    # converte para mono
    audio = tf.squeeze(audio)
    # gera o espectrograma STFT
    stft = tf.signal.stft(audio, frame_length=256, frame_step=128)
    spectrogram = tf.abs(stft)
    # normaliza para [0,1]
    spectrogram = tf.math.log(spectrogram + 1e-10)
    spectrogram = tf.expand_dims(spectrogram, -1)
    # redimensiona para 128x128
    spectrogram = tf.image.resize(spectrogram, [128, 128])
    return spectrogram, label

# Aplica transformação
ds = ds.map(audio_to_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)

# ==========================================================
# 3. Converte para numpy e seleciona subset fixo
# ==========================================================
np_ds = list(tfds.as_numpy(ds))
X = np.array([x for x, _ in np_ds])
y = np.array([y for _, y in np_ds])

# Seleciona subconjuntos fixos
x_train, y_train = X[:1000], y[:1000]
x_val, y_val     = X[1000:1128], y[1000:1128]
x_test, y_test   = X[1128:1328], y[1128:1328]

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ==========================================================
# 4. Salva como imagens (opcional)
# ==========================================================
output_dir = "/home/lucas/DeepLearning/datasets/speech_spectrograms"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(x_train[:10]):  # salva só 10 exemplos como teste
    plt.imsave(f"{output_dir}/train_{i}.png", img.squeeze(), cmap='magma')

print("✅ Espectrogramas prontos!")
