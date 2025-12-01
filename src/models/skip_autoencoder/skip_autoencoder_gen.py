import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
import random
import os

@tf.keras.utils.register_keras_serializable()
class SkipAutoencoderGen(tf.keras.Model):
    def __init__(self, id=1, latent_dim_space = (128, 256), **kwargs):
        super().__init__(*kwargs)
        self.input_img = Input(shape=(128, 128, 3))
        self.rng = random.Random(id) #cria um gerador para cada autoencoder -> assim posso passar um id e garantir que sempre dê o msm valor no espaço_latente
        self.id = id
        self.latent_dim = self.rng.randint(latent_dim_space[0], latent_dim_space[1])
        self.latent_comum = 100
        self.encoder = None
        self.decoder = None
        self.model = self.build_autoencoder()

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

        e1_in = Input(shape=(128, 128, 32))

        e2_in = Input(shape=(64, 64, 64))



        # Expande latents para shape espacial

        x = Dense(32*32*128, activation='relu')(latent_in)

        x = Reshape((32, 32, 128))(x)



        # Decoder com skip connections

        d1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)

        d1 = Concatenate()([d1, e2_in])



        d2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(d1)

        d2 = Concatenate()([d2, e1_in])



        outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d2)



        decoder = Model([latent_in, e1_in, e2_in], outputs, name='decoder')

        decoder.summary()

        return decoder



    def build_autoencoder(self):

        self.encoder = self.build_encoder()

        self.decoder = self.build_decoder()



        inputs = self.inpu

    def model_compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], **kwargs):

        super().compile(**kwargs)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    

    def load(self, model_path, model_weights=None):

        model = tf.keras.models.load_model(

            model_path,

            custom_objects={"SkipAutoencoder2Latent": SkipAutoencoder2Latent}

        )

        if model_weights is not None:

            model.load_weights(model_weights)

        return model



    def save_model(self, path, name):

        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, f'{name}-{self.id}.keras'))

        self.encoder.save(os.path.join(path, f'{name}-{self.id}-encoder.keras'))



    def save_weights(self, path, name, train):

        os.makedirs(path, exist_ok=True)

        self.model.save_weights(os.path.join(path, f'{name}-{self.id}-{train}.weights.h5'))

        self.encoder.save_weights(os.path.join(path, f'{name}-{self.id}-{train}-encoder.keras'))
