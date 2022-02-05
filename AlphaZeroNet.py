import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_dim=15, n_features=64, kernel=3, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, n_features, kernel, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(n_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class ResBlock(nn.Module):
    def __init__(self, n_features=64, kernel=3, stride=1, padding='same'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel, padding=padding, stride=stride)
        self.norm1 = nn.BatchNorm2d(n_features)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel, padding=padding, stride=stride)
        self.norm2 = nn.BatchNorm2d(n_features)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y += x
        return F.relu(y)


class OutBlock(nn.Module):
    def __init__(self, size_board=9, n_features=64, stride=1, padding='same'):
        super(OutBlock, self).__init__()

        self.policy_conv = nn.Conv2d(n_features, 2, 1, padding=padding, stride=stride)
        self.policy_norm = nn.BatchNorm2d(2)
        self.policy_linear = nn.Linear(2 * size_board**2, size_board**2 + 1)

        self.value_conv = nn.Conv2d(n_features, 1, 1, padding=padding, stride=stride)
        self.value_norm = nn.BatchNorm2d(1)
        self.value_linear1 = nn.Linear(size_board**2, n_features)
        self.value_linear2 = nn.Linear(n_features, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        p = self.policy_conv(x)
        p = self.policy_norm(p)
        p = F.relu(p)
        p = self.flatten(p)
        p = self.policy_linear(p)

        v = self.value_conv(x)
        v = self.value_norm(v)
        v = F.relu(v)
        v = self.flatten(v)
        v = self.value_linear1(v)
        v = F.relu(v)
        v = self.value_linear2(v)

        return p, torch.tanh(v)


class Net(nn.Module):
    def __init__(self, size_board=9, input_dim=15, n_features=64, kernel=3, n_blocks=5, stride=1, padding='same'):
        super(Net, self).__init__()
        self._n_blocks = n_blocks
        self.conv = ConvBlock(input_dim=15, n_features=64, kernel=3, stride=1, padding='same')
        for block in range(n_blocks):
            setattr(self, "res_%i"%block, ResBlock(n_features=64, kernel=3, stride=1, padding='same'))
        self.outblock = OutBlock(size_board=9, n_features=64, stride=1, padding='same')

    def forward(self,s):
        s = self.conv(s)
        for block in range(self._n_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
