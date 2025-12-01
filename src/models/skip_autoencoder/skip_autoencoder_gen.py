import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
import random
import os
import numpy as np


class SkipAutoencoderGenerator:
    """
    Gerador pseudoaleatório de autoencoders com skip-connections.
    Estrutura sempre espelhada: Encoder -> Latent -> Decoder com skips simétricos.
    """

    def __init__(self, input_shape=(128,128,3), min_layers=2, max_layers=4,
                 filters_list=[8,16,32,64,128], model_name=None):
        
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.filters_list = filters_list

        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.model_name = model_name

    def set_model_name(self, name):
        self.model_name = name

    # -------------------------------------------------------------------------
    # 1) CALCULAR E GERAR CONFIGURAÇÃO ALEATÓRIA
    # -------------------------------------------------------------------------
    def calculate_layers(self):
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)

        filters_used = []
        skip_shapes = []
        pool_flags = []

        current_shape = self.input_shape

        # Encoder specs
        for i in range(num_layers):
            f = np.random.choice(self.filters_list)
            filters_used.append(f)

            do_pool = np.random.choice([0,1,1])   # pool mais provável
            if current_shape[0] <= 4 or current_shape[1] <= 4:
                do_pool = 0   # impede divisão absurda
            
            pool_flags.append(do_pool)
            
            # salva a forma para skip
            skip_shapes.append(current_shape)

            if do_pool == 1:
                current_shape = (current_shape[0]//2, current_shape[1]//2, f)
            else:
                current_shape = (current_shape[0], current_shape[1], f)

        latent_dim = np.random.randint(128, 256)

        return num_layers, filters_used, pool_flags, skip_shapes, current_shape, latent_dim

    # -------------------------------------------------------------------------
    # 2) CONSTRUÇÃO DO ENCODER
    # -------------------------------------------------------------------------
    def build_encoder(self, num_layers, filters_used, pool_flags):
        inputs = Input(shape=self.input_shape)

        x = inputs
        skips = []

        for i in range(num_layers):
            x = Conv2D(filters_used[i], (3,3), activation='relu', padding='same')(x)
            skips.append(x)

            if pool_flags[i] == 1:
                x = MaxPooling2D((2,2), padding='same')(x)

        flat = Flatten()(x)
        latent = Dense(filters_used[-1], activation='relu')(flat)

        encoder = Model(inputs, [latent] + skips, name="encoder")
        return encoder

    # -------------------------------------------------------------------------
    # 3) CONSTRUÇÃO DO DECODER (espelhado com skip obrigatório)
    # -------------------------------------------------------------------------
    def build_decoder(self, num_layers, filters_used, pool_flags, final_shape, latent_dim):
        
        latent_input = Input(shape=(filters_used[-1],))
        skip_inputs = [Input(shape=final_shape) for _ in range(num_layers)]

        # Dense → reshape inicial
        x = Dense(np.prod(final_shape), activation='relu')(latent_input)
        x = Reshape(final_shape)(x)

        # Decoder espelhado
        for i in range(num_layers-1, -1, -1):
            x = Concatenate()([x, skip_inputs[i]])

            stride = 2 if pool_flags[i] == 1 else 1
            x = Conv2DTranspose(filters_used[i], (3,3), strides=stride,
                                padding='same', activation='relu')(x)

        out = Conv2D(self.input_shape[2], (3,3), padding='same',
                     activation='sigmoid')(x)

        decoder = Model([latent_input] + skip_inputs, out, name="decoder")
        return decoder

    # -------------------------------------------------------------------------
    # 4) BUILD COMPLETO
    # -------------------------------------------------------------------------
    def build_model(self, save=False):
        (num_layers, filters_used, pool_flags, skip_shapes,
         final_shape, latent_dim) = self.calculate_layers()

        # encoder
        self.encoder = self.build_encoder(num_layers, filters_used, pool_flags)

        # decoder
        self.decoder = self.build_decoder(num_layers, filters_used,
                                          pool_flags, final_shape, latent_dim)

        # full AE
        inp = Input(shape=self.input_shape)
        latent_and_skips = self.encoder(inp)
        latent = latent_and_skips[0]
        skips  = latent_and_skips[1:]

        decoded = self.decoder([latent] + skips)
        self.autoencoder = Model(inp, decoded, name=self.model_name)

        return self.autoencoder

