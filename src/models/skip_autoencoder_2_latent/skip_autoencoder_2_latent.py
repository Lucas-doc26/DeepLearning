import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
import random
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
import random

@tf.keras.utils.register_keras_serializable()
class SkipAutoencoder2Latent(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_img = Input(shape=(128, 128, 3))
        self.latent_dim = random.randint(256, 512)
        self.latent_comum = 100
        self.encoder = None
        self.decoder = None
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        inputs = self.input_img
        e1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        p1 = MaxPooling2D((2, 2))(e1)

        e2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        p2 = MaxPooling2D((2, 2))(e2)

        # Latent individual
        x = Conv2D(self.latent_dim, (3, 3), activation='relu', padding='same')(p2)
        x = Flatten()(x)
        latent = Dense(self.latent_dim, activation='relu')(x)

        # Latent comum
        x_comum = Conv2D(self.latent_comum, (3, 3), activation='relu', padding='same')(p2)
        x_comum = Flatten()(x_comum)
        latent_comum = Dense(self.latent_comum, activation='relu')(x_comum)

        encoder = Model(inputs, [latent, latent_comum, e1, e2], name='encoder')
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_in = Input(shape=(self.latent_dim,))
        latent_comum_in = Input(shape=(self.latent_comum,))
        e1_in = Input(shape=(128, 128, 32))
        e2_in = Input(shape=(64, 64, 64))

        # Expande latents para shape espacial
        x = Dense(32*32*128, activation='relu')(latent_in)
        x = Reshape((32, 32, 128))(x)

        x_comum = Dense(32*32*128, activation='relu')(latent_comum_in)
        x_comum = Reshape((32, 32, 128))(x_comum)

        # Combina latents
        x_combined = Concatenate()([x, x_comum])  # shape (32, 32, 256)

        # Decoder com skip connections
        d1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x_combined)
        d1 = Concatenate()([d1, e2_in])

        d2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(d1)
        d2 = Concatenate()([d2, e1_in])

        outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d2)

        decoder = Model([latent_in, latent_comum_in, e1_in, e2_in], outputs, name='decoder')
        decoder.summary()
        return decoder

    def build_autoencoder(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        inputs = self.input_img
        latent, latent_comum, e1, e2 = self.encoder(inputs)
        outputs = self.decoder([latent, latent_comum, e1, e2])
        model = Model(inputs, outputs, name="autoencoder")
        return model

    def call(self, inputs):
        latent, latent_comum, e1, e2 = self.encoder(inputs)
        out = self.decoder([latent, latent_comum, e1, e2])
        return out
