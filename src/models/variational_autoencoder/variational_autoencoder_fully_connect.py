import tensorflow as tf
import keras
from .variational_autoencoder import VariationalAutoencoder

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
        print(self.encoder.output)
        encoder_outputs = self.encoder.output[-1] #z_mean, z_log, z
        encoder_model = tf.keras.models.Model(self.encoder.input, encoder_outputs)
        for layer in encoder_model.layers:
            layer.treinable = False
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