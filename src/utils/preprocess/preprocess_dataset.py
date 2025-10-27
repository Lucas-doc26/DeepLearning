import tensorflow as tf
from utils.data_augmentation.transforms import albumentations_tf
import numpy as np

def preprocess_image_tf(path, label, img_size=(128,128), autoencoder=False):
    # lê arquivo direto em TF
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # ou decode_png se for png
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # normalização

    if autoencoder: 
        label = img 

    return img, label

def preprocess_dataset(df, batch_size, autoencoder=True, img_size=(128,128), transform=None, prob=0.7):
    labels = df['class'].values if not autoencoder else df['path_image'].values

    #transforma em tensores
    dataset = tf.data.Dataset.from_tensor_slices((df['path_image'].values, labels))

    def _process_transform(x, y):
        img, label = preprocess_image_tf(x, y, img_size, autoencoder)
        if transform:
            img = albumentations_tf(img, transform, prob)
        return img, label

    dataset = dataset.map(_process_transform, num_parallel_calls=tf.data.AUTOTUNE)    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def dataset_generator(dataset: tf.data.Dataset) -> np.ndarray:
    """Itera sobre o tf.data.Dataset e entrega lotes numpy."""
    for x_batch, y_batch in dataset:
        yield x_batch.numpy(), y_batch.numpy()