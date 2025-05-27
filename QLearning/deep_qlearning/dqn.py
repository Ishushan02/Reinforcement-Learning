import torch
import torch.nn as nn
from torch.nn import functional as Fn


class DQN(nn.Module):
    def __init__(self, stateDimension, hiddenDimension, outActions, enabling_dueling_dqn = False):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_features=stateDimension, out_features=hiddenDimension)
        self.enabling_dueling_dqn = enabling_dueling_dqn

        if self.enabling_dueling_dqn:
            
            self.fc_value = nn.Linear(hiddenDimension, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantage = nn.Linear(hiddenDimension, 256)
            self.advantage = nn.Linear(256, outActions)
        else:
            self.layer2 = nn.Linear(in_features=hiddenDimension, out_features=outActions)

    def forward(self, x):
        x = Fn.relu(self.layer1(x))

        if self.enabling_dueling_dqn:
            v = Fn.relu(self.fc_value(x))
            V = self.value(v)

            a = self.fc_advantage(x)
            A = self.advantage(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.layer2(x)

        return Q
    
# if __name__ == '__main__':



# model = DQN(10, 15, 4)
# tensor_test = torch.randn(128, 10)
# out = model(tensor_test)
# print(out.shape)

