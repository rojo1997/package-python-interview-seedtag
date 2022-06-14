from typing import  List, Tuple

from joblib import load
from sklearn.linear_model import LogisticRegression


class SklearnClassifier():
    def __init__(self, path: str):
        self.model: LogisticRegression = load(path)

    def predict(self, input_data: List[List[float]]):
        probas = self.model.predict_proba(input_data).tolist()
        return probas
