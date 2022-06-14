from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim: int = 4):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class PytorchClassifier:
    def __init__(self, path: str):
        self.model = Model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, input_data: List[List[float]]):
        with torch.no_grad():
            probas = self.model(torch.Tensor(input_data)).tolist()
        return probas
