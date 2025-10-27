import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
import random
import argparse

@keras.saving.register_keras_serializable()
class SkipAutoencoder2Latent(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_img = Input(shape=(128, 128, 3))
        self.latent_dim = random.randint(256, 512)
        self.latent_comum = 100
        self.encoder = None
        self.decoder = None
        self.autoencoder = self.build_autoencoder()

    def return_encoder(self):
        return self.encoder

    def return_decoder(self):
        return self.decoder

    def build_encoder(self):
        inputs = self.input_img
        e1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        p1 = MaxPooling2D((2, 2))(e1)

        e2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        p2 = MaxPooling2D((2, 2))(e2)

        latent = Conv2D(self.latent_dim, (3, 3), activation='relu', padding='same')(p2)
        latent_comum = Conv2D(self.latent_comum, (3, 3), activation='relu', padding='same')(p2)

        encoder = keras.Model(inputs, [latent, latent_comum, e1, e2], name='encoder')
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_in = Input(shape=(32, 32, self.latent_dim))
        e1_in = Input(shape=(128, 128, 32))
        e2_in = Input(shape=(64, 64, 64))

        d1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(latent_in)
        d1 = Concatenate()([d1, e2_in])

        d2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(d1)
        d2 = Concatenate()([d2, e1_in])

        outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d2)
        decoder = keras.Model([latent_in, e1_in, e2_in], outputs, name='decoder')
        decoder.summary()
        return decoder

    def build_autoencoder(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        inputs = self.input_img
        latent, latent_comum, e1, e2 = self.encoder(inputs)
        outputs = self.decoder([latent, e1, e2])
        model = Model(inputs, outputs, name="autoencoder")
        return model

    def call(self, inputs):
        latent, latent_comum, e1, e2 = self.encoder(inputs)
        out = self.decoder([latent, e1, e2])
        return out

    def load(self, model_path, model_weights):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"SkipAutoencoder2Latent": SkipAutoencoder2Latent}
        )
        model.load_weights(model_weights)
        return model