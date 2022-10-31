# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def format_trans(X):
    """ (batch,timesteps,22,c) <-> (batch,timesteps,c,11,2)   """
    if len(X.size()) == 4 and X.size(-2) == 22:
        batch, T, n_points, c = X.size()
        x1 = X.transpose(2, 3)  # (batch,T,c,22)
        x1 = x1.reshape(batch, T, c, 2, 11)  # (batch,T,c,2,11)
        tmp1 = x1[:, :, :, 0, :]  # 0 - 10 (batch,T,c,11)
        tmp2 = torch.flip(x1[:, :, :, 1, :], dims=[-1])  # 21 - 11 (batch,T,c,11)
        out = torch.stack((tmp1, tmp2), -1)  # (batch,T,c,11,2)

    elif len(X.size()) == 5 and X.size(-1) == 2 and X.size(-2) == 11:
        batch, T, c, _, _ = X.size()
        tmp1 = X[:, :, :, :, 0]  # 0 - 10 (batch,T,c,11)
        tmp2 = torch.flip(X[:, :, :, :, 1], dims=[-1])  # 11 - 21 (batch,T,c,11)
        out = torch.cat((tmp1, tmp2), dim=-1)  # (b,t,c,22)
        out = out.transpose(-1, -2)
    else:
        print('fun-format_trans: input dim is wrong')
        raise EOFError
    return out


class Classifier_Rnn(nn.Module):
    """ Rnn 判别器 """

    def __init__(self, input_size=44, num_classes=2, hidden_size=200):
        super(Classifier_Rnn, self).__init__()
        self.rnn1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.dense1 = nn.Linear(hidden_size, 256)
        self.dense2 = nn.Sequential(nn.LeakyReLU(),
                                    nn.Linear(256, num_classes),
                                    )

        self.bn = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        # x -> (b,t,44)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        out, h = self.rnn1(x)
        out = out.permute(0, 2, 1)  # (b,c,t)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (b,t,c)

        out, _ = self.rnn2(out, h)

        out = out[:, -1, :]  # (b,c)
        out = self.dp(out)
        feature = self.dense1(out)  # (b,256)
        out = self.dense2(feature)  # (b,2)

        return out, feature


class Classifier_Cnn_40(nn.Module):
    """
    3D卷积分类器 t=40
    """

    def __init__(self):
        super(Classifier_Cnn_40, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv3d(2, 128, kernel_size=2, stride=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 2, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(3, 2, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(4, 1, 1), stride=1),

        )
        self.net2 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.Tanh(),
            nn.Conv3d(256, 2, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # x -> (b,t,44)
        b, t, _ = x.shape
        out = x.reshape(b, t, 22, 2)  # (b,t,22,c)
        out = format_trans(out)  # (b,t,c,11,2)
        out = out.transpose(1, 2)  # (b,c,t,11,2)
        out = self.net1(out)  # (b,256,1,1,1)
        feature = out.squeeze()  # (b,256)
        out = self.net2(out)  # (b,2,1,1,1)
        return out.squeeze(), feature


class Classifier_Cnn_40_reduction(nn.Module):
    """
    3D卷积分类器 t=40
    """

    def __init__(self, num_classes=4):
        super(Classifier_Cnn_40_reduction, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv3d(2, 128, kernel_size=2, stride=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 2, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(3, 2, 1), stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(4, 1, 1), stride=1),

        )
        self.net2 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.Tanh(),
            nn.Conv3d(256, 2, kernel_size=1, stride=1)
        )

        self.net3 = nn.Sequential(
            nn.BatchNorm3d(2),
            nn.Tanh(),
            nn.Conv3d(2, num_classes, kernel_size=1, stride=1),
            nn.BatchNorm3d(num_classes),
            nn.Tanh(),
            nn.Conv3d(num_classes, num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # x -> (b,t,44)
        b, t, _ = x.shape
        out = x.reshape(b, t, 22, 2)  # (b,t,22,c)
        out = format_trans(out)  # (b,t,c,11,2)
        out = out.transpose(1, 2)  # (b,c,t,11,2)
        out = self.net1(out)  # (b,256,1,1,1)
        feature = self.net2(out)  # (b,2,1,1,1)
        out = self.net3(feature)  # (b,4,1,1,1)
        return out.squeeze(), feature.squeeze()


class Classifier_dim_reduction(nn.Module):
    def __init__(self, input_size=44, num_classes=4, hidden_size=200):
        super(Classifier_dim_reduction, self).__init__()
        self.rnn1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.dense1 = nn.Linear(hidden_size, 256)
        self.dense2 = nn.Sequential(nn.LeakyReLU(),
                                    nn.Linear(256, 2),
                                    )
        self.dense3 = nn.Sequential(nn.Tanh(),
                                    nn.Linear(2, num_classes),
                                    )

        self.bn = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        # x -> (b,t,44)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        out, h = self.rnn1(x)
        out = out.permute(0, 2, 1)  # (b,c,t)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (b,t,c)

        out, _ = self.rnn2(out, h)

        out = out[:, -1, :]  # (b,c)
        out = self.dp(out)
        out = self.dense1(out)  # (b,256)
        feature = self.dense2(out)  # (b,2)
        out = self.dense3(feature)  # (b,2)

        return out, feature
