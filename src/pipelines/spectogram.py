import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================================
# 1. Carrega apenas parte do dataset (ex: 2000 amostras)
# ==========================================================
subset_size = 2000
ds, info = tfds.load('speech_commands', split='train[:{}]'.format(subset_size), with_info=True, as_supervised=True)
print("Amostras carregadas:", subset_size)

# ==========================================================
# 2. Função: converte áudio -> espectrograma
# ==========================================================
def audio_to_spectrogram(audio, label):
    audio = tf.cast(audio, tf.float32) / 32768.0
    stft = tf.signal.stft(audio, frame_length=256, frame_step=128)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    return spectrogram, label

# ==========================================================
# 3. Aplica transformação e converte em batches pequenos
# ==========================================================
ds = (
    ds.map(audio_to_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(64)                              # processa em pequenos lotes
      .prefetch(tf.data.AUTOTUNE)
)

# ==========================================================
# 4. Coleta subconjunto em numpy (com segurança de memória)
# ==========================================================
X_list, y_list = [], []
for batch_x, batch_y in tfds.as_numpy(ds.take(32)):  # 32 * 64 = ~2048 amostras
    X_list.append(batch_x)
    y_list.append(batch_y)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print("Dataset carregado em memória:", X.shape, y.shape)

# Split fixo
x_train, y_train = X[:1000], y[:1000]
x_val, y_val     = X[1000:1128], y[1000:1128]
x_test, y_test   = X[1128:1328], y[1128:1328]

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ==========================================================
# 5. (Opcional) Salvar algumas imagens
# ==========================================================
output_dir = "/home/lucas/DeepLearning/datasets/speech_spectrograms"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(x_train[:10]):
    plt.imsave(f"{output_dir}/train_{i}.png", img.squeeze(), cmap='magma')

print("✅ Espectrogramas prontos (sem travar o processo)!")
