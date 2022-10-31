"""
dimensional reduction based on a multiple classifier
--mode 'train': first train a classifier
        'test': perform a dim reduction based on the trained classifier
"""
import numpy as np
import torch
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from classifier import Classifier_Cnn_40_reduction
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from enhance import enh
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


## fun
def rotate(data, ag):
    t, d_pose = data.shape
    data = data.reshape(t, 22, 2)
    w = np.array([[np.cos(ag), np.sin(ag)], [-np.sin(ag), np.cos(ag)]])
    v2 = data.reshape(-1, 2)  # (N,2)
    vw = np.matmul(v2, w).reshape(data.shape)
    vw = vw.reshape(t, d_pose)
    return vw


class DanceDataset(Dataset):
    def __init__(self, dances, labels=None, detail_labels=None):
        if labels is not None:
            assert (len(labels) == len(dances)), \
                'the number of dances should be equal to the number of labels'
        self.dances = dances
        self.labels = labels
        self.detail_labels = detail_labels

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        # angle = np.random.rand() * 2 * np.pi
        out_dance = self.dances[index]
        # out_dance = rotate(out_dance, angle)

        if self.labels is not None:
            return out_dance, self.labels[index], self.detail_labels[index]
        else:
            return out_dance


def prepare_dataloader(dance_data, labels, detail_labels, batch):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(dance_data, labels, detail_labels),
        # num_workers=8,
        batch_size=batch,
        shuffle=True,
        # collate_fn=paired_collate_fn,
        pin_memory=True
    )

    return data_loader


def select_pose(x, train_rate):
    np.random.seed(1)
    n_train = int(len(x) * train_rate)
    train_index = np.random.choice(len(x), n_train, replace=False)
    test_index = np.array(list(set(np.arange(len(x))).difference(train_index)))

    train_x = x[train_index]
    test_x = x[test_index]
    return train_x, test_x


def down_sample(data_, label_, group_, num_=300):
    data_out = []
    label_out = []
    for g_i in group_:
        index_ = np.argwhere(label_ == g_i).squeeze()
        index_down = np.random.choice(index_, num_)
        data_out.append(data_[index_down])
        label_out.append(label_[index_down])
    data_out = np.concatenate(data_out, axis=0)
    label_out = np.concatenate(label_out, axis=0)
    return data_out, label_out


def cal_emd(a, b):
    """ 计算二维分布的推土机距离 """
    d = cdist(a, b)
    assignment = linear_sum_assignment(d)
    emd = d[assignment].sum() / len(assignment[0])
    return emd


def build_data(root, file_list, enhance=False, train_rate=0.9):
    train_pose_list, train_label_list, train_label2_list = [], [], []
    test_pose_list, test_label_list, test_label2_list = [], [], []

    for i, file_i in enumerate(file_list):
        file_i = str(root / file_i)
        load_i = np.load(file_i)

        if i == 0:
            real_tmp = load_i['real']
            # real_tmp = enh(real_tmp, 8)  # todo 数据增强
            train_pose_, test_pose_ = select_pose(real_tmp, train_rate)
            if len(test_pose_) > 1000:
                index = np.random.choice(len(test_pose_), 1000)
                test_pose_ = test_pose_[index]
            if enhance:
                train_pose_, test_pose_ = enh(train_pose_, 8), enh(test_pose_, 8)
            train_label_ = np.ones((len(train_pose_, )))  # 真假label
            test_label_ = np.ones(len(test_pose_))
            train_label2 = np.ones((len(train_pose_, ))) * i  # 详细label 0 为真
            test_label2 = np.ones(len(test_pose_)) * i

            train_pose_list.append(train_pose_)
            test_pose_list.append(test_pose_)
            train_label_list.append(train_label_)
            test_label_list.append(test_label_)
            train_label2_list.append(train_label2)
            test_label2_list.append(test_label2)

        fake_tmp = load_i['fake']
        # fake_tmp = enh(fake_tmp, 8)  # todo 数据增强
        train_pose_, test_pose_ = select_pose(fake_tmp, train_rate)
        if len(test_pose_) > 1000:
            index = np.random.choice(len(test_pose_), 1000)
            test_pose_ = test_pose_[index]
        if enhance:
            train_pose_, test_pose_ = enh(train_pose_, 8), enh(test_pose_, 8)
        train_label_ = np.zeros((len(train_pose_, )))
        test_label_ = np.zeros(len(test_pose_, ))
        train_label2 = np.ones(len(train_pose_)) * (i + 1)
        test_label2 = np.ones(len(test_pose_)) * (i + 1)

        train_pose_list.append(train_pose_)
        test_pose_list.append(test_pose_)
        train_label_list.append(train_label_)
        test_label_list.append(test_label_)
        train_label2_list.append(train_label2)
        test_label2_list.append(test_label2)

    train_pose_ = np.concatenate(train_pose_list, axis=0)
    train_label_ = np.concatenate(train_label_list, axis=0)
    test_pose_ = np.concatenate(test_pose_list, axis=0)
    test_label_ = np.concatenate(test_label_list, axis=0)
    train_label2 = np.concatenate(train_label2_list, axis=0)
    test_label2 = np.concatenate(test_label2_list, axis=0)
    save_test = np.stack(test_pose_list, axis=0)  # 保存test数据，用于计算FID

    return train_pose_, train_label_, train_label2, test_pose_, test_label_, test_label2, save_test


def plot_dots(feature_, label_, name_):
    unique_label = list(set(np.sort(label_)))
    reduce_dim, detail_label = down_sample(feature_, label_, unique_label)

    plt.figure(figsize=(8, 8), dpi=300)
    for label_i in unique_label:
        index = detail_label == label_i
        feature = reduce_dim[index]
        if label_i == 0:
            plt.scatter(feature[:, 0], feature[:, 1], c='r', marker='*', s=30, linewidths=0, label='real', alpha=0.7)
        elif label_i == 1:
            plt.scatter(feature[:, 0], feature[:, 1], c='m', marker='o', s=30, linewidths=0, label='attn_cn', alpha=0.3)
        elif label_i == 2:
            plt.scatter(feature[:, 0], feature[:, 1], c='y', marker='s', s=30, linewidths=0, label='gcn', alpha=0.3)
        elif label_i == 3:
            plt.scatter(feature[:, 0], feature[:, 1], c='b', marker='^', s=30, linewidths=0, label='rnn', alpha=0.3)

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name_)
    plt.close()


def val(dev_data, classifier, epoch_i, device, args):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    FP = []
    sample_pool = []
    reduce_dim_list = []
    detail_label_list = []
    for i, batch in enumerate(dev_data):
        dance, label1, detail_label = map(lambda x: x.to(device), batch)
        dance = dance.float()
        label1 = label1.long()
        detail_label = detail_label.long()
        label = detail_label  # todo

        pred, reduce_dim = classifier(dance)
        loss = F.cross_entropy(pred, label, args.w)
        avg_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

        """ 降维 """
        reduce_dim_list.append(reduce_dim.detach().cpu().numpy())
        detail_label_list.append(detail_label.detach().cpu().numpy())

        """ 假阳性 """
        pred_real = torch.max(pred, 1)[1].view(-1).detach().cpu().numpy() == 0  # 所有判定为真的样本

        detail_label = detail_label.detach().cpu().numpy()
        detail_label_pool = set(np.sort(detail_label))
        detail_label_pool = list(detail_label_pool)[1:]  # 1:attn_cn; 2:gcn; 3:rnn

        fp_in_batch = np.zeros(len(detail_label_pool))
        for j, label_i in enumerate(detail_label_pool):
            tmp_class = detail_label == label_i
            fp_in_batch[j] = np.stack((pred_real.astype(bool), tmp_class), axis=1).all(axis=1).sum()

        FP.append(fp_in_batch)
        sample_pool.append(len(pred_real))
        size += batch[0].shape[0]

    """ 计算推土机距离 """
    Y = np.concatenate(reduce_dim_list, axis=0)
    X = np.concatenate(detail_label_list, axis=0)
    X_pool = set(np.sort(X))
    Y_list = []
    for x_i in X_pool:
        Y_list.append(Y[np.argwhere(X == x_i).squeeze()])

    emd = np.zeros(len(X_pool) - 1)
    for i in range(1, len(X_pool)):
        emd[i-1] = cal_emd(Y_list[i], Y_list[0])

    excel_name = args.save_dir / (str(epoch_i) + 'dim_reduction.xlsx')
    for i in range(len(Y_list)):
        tmp = Y_list[i]
        pd_tmp = pd.DataFrame(tmp)
        if i == 0:
            pd_tmp.to_excel(excel_name, sheet_name='real', index=False)
        else:
            writer = pd.ExcelWriter(excel_name, engine='openpyxl', mode='a')
            pd_tmp.to_excel(writer, sheet_name=str(i), index=False)
            writer.save()
            writer.close()

    """ 绘制 """
    if epoch_i % 100 == 0 or epoch_i == 1:
        name = args.save_dir / ('test_epoch' + str(epoch_i) + '.jpg')
        reduce_dim = np.concatenate(reduce_dim_list, axis=0)
        detail_label = np.concatenate(detail_label_list, axis=0)
        plot_dots(reduce_dim, detail_label, name)

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, size))

    FP = np.stack(FP, axis=0)
    FP = np.sum(FP, axis=0)
    FP = FP / np.stack(sample_pool, axis=0).sum()
    print(f'FP rate: attn_cn-gcn-rnn : {FP}')
    print(f'earth moving distance: attn_cn-gcn-rnn : {emd} \n')
    return accuracy, FP, emd


def train(train_data, dev_data, classifier, args):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, classifier.parameters()), lr=args.lr)
    steps = 0
    val_train = []
    val_test = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        train_pool = []
        feature_list = []
        label2_list = []
        for i, batch in enumerate(train_data):
            dance, label1, label2 = map(lambda x: x.to(args.device), batch)
            dance = dance.float()
            label1 = label1.long()
            label2 = label2.long()
            label = label2  # todo
            optimizer.zero_grad()
            pred, feature = classifier(dance)
            loss = F.cross_entropy(pred, label, weight=args.w)
            loss.backward()
            optimizer.step()
            steps += 1

            feature_list.append(feature.detach().cpu().numpy())
            label2_list.append(label2.detach().cpu().numpy())

            # accuracy
            corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
            train_acc = 100.0 * corrects / batch[0].shape[0]
            train_pool.append(train_acc.detach().cpu().numpy())

        # if epoch % 10 == 0 or epoch == 1:
        #     feature_list = np.concatenate(feature_list, 0)
        #     label2_list = np.concatenate(label2_list, 0)
        #     fig_name = args.save_dir / ('train_epoch' + str(epoch) + '.jpg')
        #     plot_dots(feature_list, label2_list, fig_name)

        train_pool = np.array(train_pool).mean()
        val_train.append(train_pool)
        print(f'epoch{epoch} - loss: {loss:.6f} - acc: {train_pool:.6f}')
        with open(str(args.log_train), 'a', newline='') as p:
            writer = csv.writer(p)
            writer.writerow([epoch, train_pool])

        # evaluate the model on test set at each epoch
        dev_acc, FP, emd = val(dev_data, classifier, epoch, args.device, args)
        with open(str(args.log_test), 'a', newline='') as p:
            writer = csv.writer(p)
            writer.writerow([epoch, dev_acc.detach().cpu().numpy()] + FP.tolist() + emd.tolist())

        if epoch % args.save_interval == 0:
            name = args.save_dir / ('epoch' + str(epoch) + '.pt')
            torch.save(classifier, name)

    return classifier


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')  # todo train or test
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_dir', metavar='PATH',
                        default='../../../../results/path2pose/assessment_fdd_nrds_5_35_SNGAN_public/dim_reduction_gcn3_20000')

    args = parser.parse_args()

    args.save_dir = Path(args.save_dir)
    if not args.save_dir.exists():
        args.save_dir.mkdir()

    if args.mode == 'train':
        args.log_train = args.save_dir / 'log_train.csv'
        with open(str(args.log_train), 'w') as p:
            writer = csv.writer(p)
            writer.writerow(['epoch', 'acc'])

        args.log_test = args.save_dir / 'log_test.csv'
        with open(str(args.log_test), 'w') as p:
            writer = csv.writer(p)
            writer.writerow(['epoch', 'acc',
                             'FP_attncn', 'FP_gcn', 'FP_rnn',
                             'emb_attncn', 'emb_gcn', 'emb_rnn'])

    root_dir = Path('../../../../results/path2pose')  # todo
    npz_list = [
        'v11_public_attncn_220812_202737/val/epoch20000/test_results.npz',
        'v11_public_gcn3_220815_201054/val/epoch20000/test_results.npz',
        'v11_public_rnn_220812_203203/val/epoch20000/test_results.npz',
        # 'v5_GenAttention_DiscCnnSN_predict_5_35_nosm_for_assess_220505_101015/test/40000/test_results.npz',
    ]

    train_pose, train_label, train_detail_label, test_pose, test_label, test_detail_label, save_data = \
        build_data(root_dir, npz_list, enhance=False, train_rate=0.8)

    train_loader = prepare_dataloader(train_pose, train_label, train_detail_label, args.batch_size)
    test_loader = prepare_dataloader(test_pose, test_label, test_detail_label, args.batch_size)

    args.device = torch.device('cuda:0')  # todo
    args.w = torch.ones(len(npz_list)+1).float().to(args.device)

    if args.mode == 'train':
        model = Classifier_Cnn_40_reduction(len(npz_list) + 1)
        model = model.to(args.device)
        model = train(train_loader, test_loader, model, args)
    elif args.mode == 'test':
        model_path = args.save_dir / 'epoch200.pt'
        model = torch.load(str(model_path))
        acc, fp, emb_dist = val(test_loader, model, 200, args.device, args)
