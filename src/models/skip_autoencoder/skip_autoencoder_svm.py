from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .skip_autoencoder import SkipAutoencoder

class SkipAutoencoderSVM():
    def __init__(self, model_path, model_weights):
        self.encoder = self.load_encoder(model_path, model_weights)                                                                          
        self.svm = None

    def load_encoder(self, model_path, model_weights):
        skip = SkipAutoencoder()
        skip.load(model_path=model_path)
        self.encoder = skip.return_encoder()
        self.encoder.load_weights(model_weights, skip_mismatch=True)
        del skip
        return self.encoder                                 

    def train(self, x, y):
        pred, _, _ = self.encoder.predict(x)

        # Flatten para 2D
        if len(pred.shape) > 2:
            pred = pred.reshape((pred.shape[0], -1))

        # hiperparâmetros
        param_grid = {'C':[2], 'gamma':['scale', 'auto']}

        grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=2, scoring='accuracy')
        grid.fit(pred, y)

        print("Melhores parâmetros:", grid.best_params_)
        print("Melhor score CV:", grid.best_score_)

        # salve o melhor modelo
        self.svm = grid.best_estimator_

    def test(self, x):
        pred, _, _ = self.encoder.predict(x)
        if len(pred.shape) > 2:
            pred = pred.reshape((pred.shape[0], -1))
        y_pred = self.svm.predict(pred)
        return y_pred
