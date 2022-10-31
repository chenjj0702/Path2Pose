import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
import scipy.interpolate as spi
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from openpyxl import Workbook


##
def get_path(X, point):
    if X.shape[-1] == 44:
        X = X.reshape(X.shape[0], X.shape[1], 22, 2)  # (n,T,22,2)

    out = X[:, :, point, :]
    return out


def cal_dtw(a, b):
    # a = spline_interpolate(a)
    # b = spline_interpolate(b)

    # euclidean_norm = lambda x, y: np.abs(x - y)
    euclidean_norm = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
    d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=euclidean_norm)
    return d


def cal_body_len(_pose):
    """
    :param _pose: (n,t,22,2)
    :return:
    """
    tmp1 = _pose[:, :, :12, :]
    tmp2 = np.concatenate((_pose[:, :, 11:, :], _pose[:, :, [0], :]), axis=2)
    tmp2 = np.flip(tmp2, axis=2)
    _pose_trans = np.stack((tmp1, tmp2), axis=2)  # (n,t,2,12,2)
    seg = _pose_trans.mean(axis=2)  # (n,t,12,2)
    seg_len = np.sqrt(np.sum(np.square(np.diff(seg, axis=2)), axis=-1))  # (n,t,11)
    _body = np.sum(seg_len, axis=-1)  # (n,t)
    ave_body = _body.mean(-1)  # (n,)
    return ave_body


## main
if __name__ == '__main__':
    np.random.seed(1)
    npz_name = Path(
        '../../../../results/path2pose/v11_public_attncn_220812_202737/val/epoch10000/test_results.npz')

    obj_dir = Path('../../../../results/path2pose/evaluation_morphology/path_analysis')
    if not obj_dir.exists():
        obj_dir.mkdir(parents=True)

    load_data = np.load(npz_name)
    raw_real = load_data['real']
    raw_fake = load_data['fake']
    n, t, _ = raw_real.shape
    raw_real = raw_real.reshape(n, t, 22, 2)  # (n,t,22,2)
    raw_fake = raw_fake.reshape(n, t, 22, 2)

    base_point = 11
    path_real = raw_real[:, :, base_point, :]  # (n,t,2)
    path_fake = raw_fake[:, :, base_point, :]

    """ plot samples """
    # plt.rcParams['figure.figsize'] = [3, 3]
    # plt.rcParams['figure.dpi'] = 300
    # ids = np.random.choice(len(path_real), 20)
    # for i, id_i in enumerate(ids):
    #     real_i = path_real[i]
    #     fake_i = path_fake[i]
    #     plt.figure()
    #     plt.plot(real_i[:, 0], real_i[:, 1], '-', color='orange', label='real')
    #     plt.plot(fake_i[:, 0], fake_i[:, 1], 'o', color='blue', markersize=3, label='fake')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.legend()
    #     name = str(obj_dir / (str(i)+'.jpg'))
    #     plt.savefig(name)

    """ cc """
    cc = np.zeros((len(path_real), 2))
    for i in range(len(path_real)):
        for j in range(2):
            a = pd.Series(path_real[i, :, j])
            b = pd.Series(path_fake[i, :, j])
            cc[i, j] = b.corr(a)
    cc = cc.mean(1)

    """ NPD """
    body_len = cal_body_len(raw_real)
    dist = np.mean(np.sqrt(np.sum(np.square(path_real - path_fake), -1)), -1)  # (n,)
    dist = dist / body_len

    """ save """
    name = str(obj_dir / 'path_cc_euler.xlsx')
    wb = Workbook()
    ws = wb.active
    ws.append(['sample', 'cc', 'distance'])
    for i in range(len(cc)):
        ws.append([i + 1, cc[i], dist[i]])
    wb.save(name)
