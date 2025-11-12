import tensorflow as tf
import keras
from keras import layers
import os
from tensorflow.keras.models import Model

#No VariationalAutoencoder, ele gera dois vetores latentes: 
# z_mean = média gaussiana latente
# z_log_var = log da variância de z_mean
# Só que a gente quer amostrar um vetor z dessa distribuição pra cada input, mas de forma que dê pra fazer backpropagation (senão o gradiente trava).
#Por isso o sampling existe: deixar z derivável, ou seja, tem que ser reparametrizado
class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) para criar o z, o vetor que será codificado."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.Generator.from_seed(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = self.seed_generator.normal(shape=(batch, dim)) # rúido
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon #reparametrização(determinística + diferenciável) -> backpropagation mesmo usando amostragem
    
@keras.saving.register_keras_serializable()
class VariationalAutoencoder(keras.Model):
    def __init__(self, latent_dim:int = 128, input_shape:int = (128,128,3), **kwargs):
        super().__init__(**kwargs)
        self.img_input = keras.Input(shape=input_shape)
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        #Erros
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.autoencoder = self.build_autoencoder()

    def return_encoder(self):
        return self.encoder
    
    def return_decoder(self):
        return self.decoder

    def build_encoder(self):
        encoder_inputs = self.img_input
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)

        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        return encoder

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 32 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((32, 32, 64))(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def build_autoencoder(self):
        inputs = self.img_input
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        model = Model(inputs, outputs, name="autoencoder")
        return model

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0] # se vier em x, y -> pego só o x

        # GradientesTape -> monitora todas as operações pra saber quem afeta quem.
        with tf.GradientTape() as tape:
            #z_mean = média gausiana latente
            #z_log_var - log da variância 
            #z = z_mean + eps * std
            z_mean, z_log_var, z = self.encoder(data)

            #passo z para reconstruir 
            reconstruction = self.decoder(z)


            #calcula o erro de reconstrução
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            #KL = força cada z a ficar perto de uma Gaussiana padrão N(0, I).
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) 
            total_loss = reconstruction_loss + kl_loss #loss final de tudo

        grads = tape.gradient(total_loss, self.trainable_weights)
        #total_loss: valor escalar da função de custo.
        #self.trainable_weights: todos os pesos treináveis do encoder + decoder.
        #grads: lista de tensores com o gradiente parcial pra cada peso.

        # att os pesos usando esses gradientes
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #aqui encima de fato aconteceu a backpropagation:
        # encoder -> decoder -> total_loss
        # gradientes -> aplica os gradientes        

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)

        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) 
        total_loss = reconstruction_loss + kl_loss #loss final de tudo

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def load(self, model_path, model_weights=None):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"VariationalAutoencoder": VariationalAutoencoder, "Sampling": Sampling}
        )
        if model_weights is not None:
            model.load_weights(model_weights)
        return model

    def save_model(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.autoencoder.save(os.path.join(path, f'{name}.keras'))

    def save_weights(self, path, name, train):
        os.makedirs(path, exist_ok=True)
        self.autoencoder.save_weights(os.path.join(path, f'{name}-{train}.weights.h5'))
    

    





















