import math
import numpy as np
from scipy import linalg
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


## func
def select_pose(x, train_rate):
    n_train = int(len(x) * train_rate)
    train_index = np.random.choice(len(x), n_train, replace=False)
    test_index = np.array(list(set(np.arange(len(x))).difference(train_index)))

    train_x = x[train_index]
    test_x = x[test_index]
    return train_x, test_x


def build_data(root, file_list):
    train_pose_list, train_label_list = [], []
    test_pose_list, test_label_list = [], []

    for i, file_i in enumerate(file_list):
        file_i = str(root / file_i)
        load_i = np.load(file_i)

        if i == 0:
            real_tmp = load_i['real_full']
            train_pose, test_pose = select_pose(real_tmp, 0.7)
            train_label = np.ones((len(train_pose, )))
            test_label = np.ones(len(test_pose, ))

            train_pose_list.append(train_pose)
            test_pose_list.append(test_pose)
            train_label_list.append(train_label)
            test_label_list.append(test_label)

        fake_tmp = load_i['fake_full']
        train_pose, test_pose = select_pose(fake_tmp, 0.7)
        train_label = np.zeros((len(train_pose, )))
        test_label = np.zeros(len(test_pose, ))

        train_pose_list.append(train_pose)
        test_pose_list.append(test_pose)
        train_label_list.append(train_label)
        test_label_list.append(test_label)

    train_pose = np.concatenate(train_pose_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    test_pose = np.concatenate(test_pose_list, axis=0)
    test_label = np.concatenate(test_label_list, axis=0)

    return train_pose, train_label, test_pose, test_label


def get_activations(data, model_, device, batch_size=128):
    model_.eval()

    if batch_size > len(data):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(data)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    features = []
    for batch in loader:
        batch = batch.to(device)
        _, feature = model_(batch)
        features.append(feature)

    features = torch.cat(features, 0).detach().cpu().numpy()
    return features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def _compute_statistics_of_path(data, model_, batch_size, device):
    """Calculation of the statistics used by the FID.
        Params:
        -- data        : [n, seq_len, d_pose]
        -- model       :
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.

        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """

    act = get_activations(data, model_, device, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


def calculate_fid(model, data1_, data2_, batch_size, device, times=5):
    """Calculates the FID of two paths"""
    fid_list = []

    for t in range(times):
        m1, s1 = _compute_statistics_of_path(data1_, model, batch_size, device)
        m2, s2 = _compute_statistics_of_path(data2_, model, batch_size, device)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        fid_list.append(fid_value)

    fid_value = np.stack(fid_list, 0).mean(0)

    return fid_value


def cal_main(model_, device, real_data, fake_data, N):
    assert len(real_data) == len(fake_data)
    win = math.floor(len(real_data) / N)
    start_i = 0
    end_i = win
    fid_list = []
    while end_i <= len(real_data):
        y0 = real_data[start_i: end_i].astype(np.float32)
        y1 = fake_data[start_i: end_i].astype(np.float32)

        fid1 = calculate_fid(model_, y0, y1, 256, device)
        fid_list.append(fid1)
        start_i += win
        end_i += win
    fid_list = np.array(fid_list)
    return fid_list


## main
if __name__ == '__main__':
    root_dir = Path('../../../../results/path2pose/assessment_fdd_nrds_5_35_SNGAN_public/fdd_gcn3_20000')
    FDD_pool = []
    data_path = root_dir / 'data_for_fdd.npz'
    data_load = np.load(str(data_path), allow_pickle=True)
    raw_data = data_load['test_data'].item()

    real_pose = raw_data['real']
    del raw_data['real']
    fake_pose = raw_data

    model_path = str(root_dir / 'epoch100.pt')
    dev = torch.device('cuda:0')
    model = torch.load(model_path, map_location=dev)

    boot_strap = 10
    label_pool = []
    fdd_pool = np.zeros((boot_strap, len(fake_pose)))
    for i, (k, v) in enumerate(tqdm(fake_pose.items())):
        label_pool.append(k)
        fdd_pool[:, i] = cal_main(model, dev, real_pose, v, boot_strap)

    # save data for origin plot
    excel_name = root_dir / 'fdd_pool.xlsx'
    pd_writer = pd.ExcelWriter(str(excel_name))
    fdd_df = pd.DataFrame(fdd_pool)
    fdd_df.to_excel(pd_writer, sheet_name='1', index=False, header=label_pool)
    pd_writer.save()
    pd_writer.close()
