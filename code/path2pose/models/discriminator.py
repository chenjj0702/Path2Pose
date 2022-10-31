import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


## func
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


##
class Disc_cnn_40(nn.Module):
    def __init__(self, args, device):
        super(Disc_cnn_40, self).__init__()
        self.args = args
        self.device = device

        self.net = nn.Sequential(
            nn.Conv3d(args.dim * 2, 128, kernel_size=2, stride=1),
            nn.LeakyReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 2, 1), stride=2),
            nn.LeakyReLU(),
            nn.Dropout(args.disc_dp),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 1), stride=2),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, kernel_size=(3, 2, 1), stride=2),
            nn.LeakyReLU(),
            nn.Dropout(args.disc_dp),

            nn.Conv3d(256, 512, kernel_size=(4, 1, 1), stride=1),
            nn.Tanh(),

            nn.Conv3d(512, 1, kernel_size=1, stride=1)
        )

        self.norm()

    def norm(self):
        if self.args.disc_sn:
            for m in self.net.modules():
                if isinstance(m, nn.Conv3d):
                    spectral_norm(m)

    def forward(self, pose, path, hist):
        b, t, _ = path.size()
        if len(hist.size()) == 2:
            hist = hist.unsqueeze(1)

        pose = torch.cat((hist, pose), dim=1)
        pose = pose.reshape(b, t, self.args.n_points, self.args.dim)
        pose = format_trans(pose)  # reshape
        path = path.unsqueeze(-1).unsqueeze(-1).expand_as(pose)

        out = torch.cat((pose, path), dim=2)
        out = out.transpose(1, 2)

        out = self.net(out)
        return out.squeeze()


## main
if __name__ == '__main__':
    pass
