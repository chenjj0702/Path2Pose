""" 杂散的工具函数 """
import matplotlib.pyplot as plt
import numpy as np


def plot_lr(fun, lr_base, max_epoch, name):
    """ 学习率曲线 """
    r = list(range(1, max_epoch+1))
    lr = [fun(x) * lr_base for x in r]

    plt.figure()
    plt.plot(lr)
    plt.savefig(name)


def format_trans(X):
    """ (batch,timesteps,24,c) <-> (batch,timesteps,c,12,2)   """
    if X.ndim == 4 and X.shape[-2] == 24:
        batch, T, n_points, c = X.shape
        x1 = X.transpose(0, 1, 3, 2)  # (batch,T,c,24)
        # x1 = x1.reshape(batch, T, c, 2, 12).transpose(0, 1, 2, 4, 3)
        x1 = x1.reshape(batch, T, c, 2, 12)  # (batch,T,c,2,12)
        tmp1 = x1[:, :, :, 0, :]  # 0 - 11 (batch,T,c,12)
        tmp2 = np.flip(x1[:, :, :, 1, :], axis=-1)  # 12 - 23 (batch,T,c,12)
        out = np.stack((tmp1, tmp2), -1)  # (batch,T,c,2,12)

    elif X.ndim == 5 and X.shape[-1] == 2 and X.shape[-2] == 12:
        batch, T, c, _, _ = X.shape
        tmp1 = X[:, :, :, :, 0]  # 0 - 11 (batch,T,c,12)
        tmp2 = np.flip(X[:, :, :, :, 1], axis=-1)  # 12 - 23 (batch,T,c,12)
        out = np.stack((tmp1, tmp2), axis=-2)  # (batch,T,c,2,12)
        out = out.reshape((batch, T, c, -1))  # (batch,T,c,24)
        out = out.swapaxes(-1, -2)
    else:
        print('fun-format_trans: input dim is wrong')
        raise EOFError
    return out
