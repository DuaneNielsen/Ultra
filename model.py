import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=20, stride=5)
        self.cbn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=2)
        self.cbn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=2)
        self.cbn3 = nn.BatchNorm1d(64)
        self.l1 = nn.Linear(22*64, 1)

    def forward(self, x):
        act_map = F.relu(self.cbn1(self.conv1(x)))
        act_map = F.relu(self.cbn2(self.conv2(act_map)))
        act_map = F.relu(self.cbn3(self.conv3(act_map)))
        act_map = F.relu(self.l1(act_map.view(x.shape[0], -1)))
        return act_map
