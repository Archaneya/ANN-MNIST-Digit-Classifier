import torch.nn as nn
class ANN(nn.Module):

    def __init__(self,num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.model(x)