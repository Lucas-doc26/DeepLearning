import tensorflow as tf
import keras 


@keras.saving.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.input_img = tf.keras.Input(shape=(128, 128, 3))
        self.latent_dim = 256
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_autoencoder()

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_autoencoder(self): 
        return self.model
    
    def build_encoder(self):
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(self.input_img)
        x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_dim, activation='relu', name='latent_vector')(x)
        self.encoder = tf.keras.Model(self.input_img, x, name='encoder')
        return self.encoder
    
    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(32 * 32 * 64, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((32, 32, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
        self.decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        return self.decoder

    def build_autoencoder(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        inputs = self.input_img
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        self.model = tf.keras.Model(inputs, outputs, name='autoencoder')
        return self.model
    
    def load(self, model_path, model_weights):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"Autoencoder": Autoencoder},
        )
        model.load_weights(model_weights)
        return model