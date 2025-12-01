import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
import os
from sklearn.decomposition import PCA


# Certifique-se de que SkipAutoencoder está no mesmo pacote
from .skip_autoencoder import SkipAutoencoder


@keras.saving.register_keras_serializable()
class SkipAutoencoderFullyConnected2(keras.Model):
    def __init__(self, encoder_model_path, encoder_weights_path=None, **kwargs):
        super().__init__(**kwargs)
        self.skip = self.load_skip_autoencoder(encoder_model_path, encoder_weights_path)
        self.model = self.build_model()

    def load_skip_autoencoder(self, encoder_model_path, encoder_weights_path):
        skip = tf.keras.models.load_model(
            encoder_model_path,
            custom_objects={"SkipAutoencoder": SkipAutoencoder}
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
        model = Sequential([
            Input(shape=(128,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax') 
        ], name='SkipAutoencoderFullyConnected')
        
        model.summary()
        model.compile()
        return model

    def train_model(self, train, valid, epochs=10, callbacks=[]):
        train_encoded = train.map(self.encode_batch)
        valid_encoded = valid.map(self.encode_batch)

        history = self.model.fit(
            train_encoded,
            validation_data=valid_encoded,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    
    def test_model(self, data):
        z = data.map(self.encode_batch)
        return self.model.predict(z)

    def call(self, x, training=False):
        return self.model(x, training=training)

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], **kwargs):
        super().compile(**kwargs)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def encode_batch(self, x, y):
        z = self.skip(x, training=False)
        return z, y