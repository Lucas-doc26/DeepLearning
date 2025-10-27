from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVM_AE():
    def __init__(self, encoder, encoder_weights=None):
        self.encoder = encoder
        if encoder_weights != None:
            self.load_weights(encoder_weights)

        self.svm = None

    def load_weights(self, encoder_weights):
        self.encoder.load_weights(encoder_weights, skip_mismatch=True)

    def train(self, x, y):
        pred = self.encoder.predict(x)

        # Flatten para 2D
        if len(pred.shape) > 2:
            pred = pred.reshape((pred.shape[0], -1))

        # hiperparâmetros
        param_grid = {'C':[0.1, 1, 2], 'gamma':['scale', 'auto']}

        grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
        grid.fit(pred, y)

        print("Melhores parâmetros:", grid.best_params_)
        print("Melhor score CV:", grid.best_score_)

        # salve o melhor modelo
        self.svm = grid.best_estimator_

    def test(self, x):
        pred = self.encoder.predict(x)
        y_pred = self.svm.predict(pred)
        return y_pred

