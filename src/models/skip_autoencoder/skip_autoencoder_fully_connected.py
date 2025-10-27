import tensorflow as tf
import keras 
from .skip_autoencoder import SkipAutoencoder

@keras.saving.register_keras_serializable()
class SkipAutoencoderFullyConnected(keras.Model):
    def __init__(self, model_path, model_weights, **kwargs):
        super().__init__(**kwargs)
        self.encoder = self.load_encoder(model_path, model_weights)
        self.model = self.create_model()

    def load_encoder(self, model_path, model_weights):
        skip = SkipAutoencoder()
        skip.load(model_path=model_path)
        self.encoder = skip.return_encoder()
        self.encoder.load_weights(model_weights, skip_mismatch=True)
        del skip
        return self.encoder
    
    def create_model(self):
        encoder_outputs = self.encoder.output[0]
        encoder_model = tf.keras.models.Model(self.encoder.input, encoder_outputs)
        self.model = tf.keras.models.Sequential([
                    encoder_model,  
                    tf.keras.layers.Flatten(), 
                    tf.keras.layers.Dropout(0.2),  
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(128,activation='relu'),  
                    tf.keras.layers.Dense(2, activation='softmax')  
                ], name=f'SkipAutoencoderFullyConnected')
        
        self.model.summary()
        return self.model
    
    def call(self, x, training=False):
        return self.model(x, training=training)