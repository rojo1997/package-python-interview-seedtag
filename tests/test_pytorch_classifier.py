from interview_seedtag import PytorchClassifier

def test_pytorch_classifier():
    pytorch_classfier = PytorchClassifier("./model/pytorch.model")
    pytorch_classfier.predict(
        [[0.92, 0.12, 0.31, 0.09]]
    )