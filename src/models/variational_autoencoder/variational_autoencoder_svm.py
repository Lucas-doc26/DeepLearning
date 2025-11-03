from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .variational_autoencoder import VariationalAutoencoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class VariationalAutoencoderSVM():
    def __init__(self, model_path, model_weights):
        self.encoder = self.load_encoder(model_path, model_weights)                                                                          
        self.svm = None

    def load_encoder(self, model_path, model_weights):
        vae = VariationalAutoencoder()
        vae.load(model_path=model_path)
        self.encoder = vae.return_encoder()
        self.encoder.load_weights(model_weights, skip_mismatch=True)
        del vae
        return self.encoder                                 

    def train(self, x, y):
        _, _, z = self.encoder.predict(x)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf'))
        ])

        param_grid = {
            'svm__C': [0.1, 0.5, 0.8, 1, 1.5, 2, 3, 5, 10],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1]
        }

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        grid.fit(z, y)

        print("Melhores parâmetros:", grid.best_params_)
        print("Melhor score CV:", grid.best_score_)

        # salve o melhor modelo
        self.svm = grid.best_estimator_

    def test(self, x):
        _, _, z = self.encoder.predict(x)
        y_pred = self.svm.predict(z)
        return y_pred
