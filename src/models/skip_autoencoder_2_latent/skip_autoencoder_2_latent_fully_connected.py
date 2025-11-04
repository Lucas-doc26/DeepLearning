import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
import os
from sklearn.decomposition import PCA


from .skip_autoencoder_2_latent import SkipAutoencoder2Latent


@keras.saving.register_keras_serializable()
class SkipAutoencoder2LatentFullyConnected(keras.Model):
    def __init__(self, encoder_model_path, encoder_weights_path=None, **kwargs):
        super().__init__(**kwargs)
        self.skip = self.load_skip_autoencoder(encoder_model_path, encoder_weights_path)
        self.model = self.build_model()

    def load_skip_autoencoder(self, encoder_model_path, encoder_weights_path):
        skip = tf.keras.models.load_model(
            encoder_model_path,
            custom_objects={"SkipAutoencoder2Latent": SkipAutoencoder2Latent}
        )
        encoder = skip.get_layer('encoder')  # pega a camada encoder do autoencoder
        if encoder_weights_path:
            encoder.load_weights(encoder_weights_path)
        
        encoder_model = Model(
            inputs=encoder.input,
            outputs=encoder.output[0],
            name='encoder_model'
        )

        encoder_model.summary()
        return encoder_model
    
    def build_model(self):
        #congela o encoder
        self.skip.trainable = False  
        for layer in self.skip.layers:
            layer.trainable = False

        model = Sequential([
            self.skip,
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax') 
        ], name='SkipAutoencoderFullyConnected')
        
        model.summary()
        model.compile()
        return model
    

    def call(self, x, training=False):
        return self.model(x, training=training)

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], **kwargs):
        super().compile(**kwargs)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)