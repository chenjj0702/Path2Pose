import numpy as np
from scipy import linalg
import torch
import pandas as pd
from pathlib import Path
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


def get_activations(data, model_, device, batch_size=10000):
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


## main
if __name__ == '__main__':
    root_dir = Path('../../../../results/path2pose/assessment_long_pose_public/epoch20000_dataset340')  # todo 选择文件夹
    FID_pool = []
    for dir_i in root_dir.iterdir():
        if 'cross_validation' in dir_i.name:
            data_path = dir_i / 'data_for_fid.npz'  # 计算fid的测试数据
            data_load = np.load(str(data_path), allow_pickle=True)

            pose = data_load['test_pose']
            label1 = data_load['test_label1']
            label2 = data_load['test_label2']

            """ 数据预处理 """
            # pose 中的数据是按照 真1，假1，真2，假2，真3，假3，真4，假4 的顺序排列的， 需要按组划分，每组有50个真和50个假
            win = (label2 == 1).sum()
            pose_pool = []
            for i in [1, 2, 3, 4]:
                index_real = np.argwhere(label2 == i)[:, 0]
                index_fake = index_real + win
                tmp_real = pose[index_real]
                tmp_fake = pose[index_fake]

                tmp = np.stack((tmp_real, tmp_fake), axis=0)  # (2,50,35,44)
                pose_pool.append(tmp)
            pose_pool = np.stack(pose_pool, axis=0).astype(np.float32)  # (4,2,50,35,44)

            """ 计算FID """
            model_path = dir_i / 'epoch100.pt'  # todo
            model = torch.load(str(model_path))
            dev = torch.device('cuda:0')
            FID = np.zeros(len(pose_pool))

            for i in range(len(pose_pool)):
                tmp_real = pose_pool[i, 0]
                tmp_fake = pose_pool[i, 1]
                batch_size = min(128, len(tmp_real))
                FID[i] = calculate_fid(model, tmp_real, tmp_fake, batch_size, dev)

            label = range(1, len(pose_pool)+1)
            zipped = zip(label, FID)
            print_list = [f'{a}: {b:.3f}' for a, b in list(zipped)]
            print(';  '.join(print_list))

            FID_pool.append(FID)
    FID_pool = np.stack(FID_pool, axis=0)  # (n,4)

    excel_name = root_dir / 'fid_pool.xlsx'
    pd_writer = pd.ExcelWriter(str(excel_name))
    fid_df = pd.DataFrame(FID_pool)
    fid_df.to_excel(pd_writer, sheet_name='1', index=False, header=['1', '2', '3', '4'])
    pd_writer.save()
