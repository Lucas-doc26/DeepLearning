import tensorflow as tf
import keras
from .variational_autoencoder import VariationalAutoencoder, Sampling
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential

@keras.saving.register_keras_serializable()
class VariationalAutoencoderFullyConnected(keras.Model):
    def __init__(self, model_path, model_weights, **kwargs):
        super().__init__(**kwargs)
        self.encoder = self.load_encoder(model_path, model_weights)
        self.model = self.create_model()

    def load_encoder(self, model_path, model_weights):
        skip = VariationalAutoencoder()
        skip.load(model_path=model_path)
        self.encoder = skip.return_encoder()
        self.encoder.load_weights(model_weights, skip_mismatch=True)
        del skip
        return self.encoder
    
    def create_model(self):
        
        encoder_model.trainable = False
        self.model = tf.keras.models.Sequential([
                    encoder_model,  
                    tf.keras.layers.Dropout(0.2),  
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(128,activation='relu'),  
                    tf.keras.layers.Dense(2, activation='softmax')  
                ], name=f'VariationalAutoencoderFullyConnected')
        
        self.model.summary()
        return self.model
    
    def call(self, x, training=False):
        return self.model(x, training=training)

@keras.saving.register_keras_serializable()
class VariationalAutoencoderFullyConnected(keras.Model):
    def __init__(self, encoder_model_path, encoder_weights_path=None, **kwargs):
        super().__init__(**kwargs)
        self.vae = self.load_vae_autoencoder(encoder_model_path, encoder_weights_path)
        self.model = self.build_model()

    def load_vae_autoencoder(self, encoder_model_path, encoder_weights_path):
        vae = tf.keras.models.load_model(
            encoder_model_path,
            custom_objects={"VariationalAutoencoder": VariationalAutoencoder, "Sampling": Sampling}
        )
        encoder = vae.get_layer('encoder')  # pega a camada encoder do autoencoder
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
        ], name='VariationalAutoencoderFullyConnected')
        
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
        z = self.vae(x, training=False)
        return z, y