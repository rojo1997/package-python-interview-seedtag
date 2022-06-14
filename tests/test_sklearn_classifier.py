from interview_seedtag import SklearnClassifier

def test_sklearn_classifier():
    sklearn_classifier = SklearnClassifier("./model/sklearn.model")
    _ = sklearn_classifier.predict(
        [[0.92, 0.12, 0.31, 0.09]]
    )