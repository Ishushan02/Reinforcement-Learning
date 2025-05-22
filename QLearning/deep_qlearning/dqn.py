import torch
import torch.nn as nn
from torch.nn import functional as Fn


class DQN(nn.Module):
    def __init__(self, stateDimension, hiddenDimension, outActions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_features=stateDimension, out_features=hiddenDimension)
        self.layer2 = nn.Linear(in_features=hiddenDimension, out_features=outActions)

    def forward(self, x):
        x = Fn.relu(self.layer1(x))
        return self.layer2(x)
    


# model = DQN(10, 15, 4)
# tensor_test = torch.randn(128, 10)
# out = model(tensor_test)
# print(out.shape)

