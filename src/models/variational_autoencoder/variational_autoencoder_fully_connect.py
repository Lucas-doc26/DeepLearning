import tensorflow as tf
import keras
from .variational_autoencoder import VariationalAutoencoder as VAE

@keras.saving.register_keras_serializable()
class VariationalAutoencoderFullyConnected(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.VariationalAutoencoderFullyConnected = None

    def create(self, encoder, freeze:bool=True):
        if encoder != None:
            if freeze:
                for layer in encoder.layers:
                    layer.trainable = False
                encoder.trainable = False

            encoder_output = encoder.output[2]  # [z_mean, z_log_var, z]
            encoder_input = encoder.input
            encoder_model = tf.keras.Model(encoder_input, encoder_output, name="encoder_for_VariationalAutoencoderFullyConnected")

            #crio o classificador com o enconder
            self.VariationalAutoencoderFullyConnected = tf.keras.models.Sequential([
                    encoder_model,  
                    tf.keras.layers.Dropout(0.2),  
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(128,activation='relu'),  
                    tf.keras.layers.Dense(2, activation='softmax')  
                ], name=f'VariationalAutoencoderFullyConnected-{encoder.name}')

            self.VariationalAutoencoderFullyConnected.summary()
            return self.VariationalAutoencoderFullyConnected
    
    def load(self, model_path, model_weights):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"VariationalAutoencoderFullyConnected": VariationalAutoencoderFullyConnected},
        )
        model.load_weights(model_weights)
        return model
    