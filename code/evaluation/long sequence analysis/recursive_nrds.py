"""
binary classifier for each individual segment
"""
import numpy as np
import pandas as pd
import torch
import math
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from classifier import Classifier_Rnn, Classifier_Cnn_35
import torch.nn.functional as F
import torch.optim as optim
from enhance import enh


## fun
class DanceDataset(Dataset):
    def __init__(self, dances, labels=None):
        if labels is not None:
            assert (len(labels) == len(dances)), \
                'the number of dances should be equal to the number of labels'
        self.dances = dances
        self.labels = labels

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.dances[index], self.labels[index]
        else:
            return self.dances[index]


def prepare_dataloader(dance_data, labels, batch):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(dance_data, labels),
        # num_workers=8,
        batch_size=batch,
        shuffle=True,
        # collate_fn=paired_collate_fn,
        pin_memory=True
    )

    return data_loader


def select_pose(x, train_rate):
    n_train = int(len(x) * train_rate)
    train_index = np.random.choice(len(x), n_train, replace=False)
    test_index = np.array(list(set(np.arange(len(x))).difference(train_index)))

    train_x = x[train_index]
    test_x = x[test_index]
    return train_x, test_x


# def build_data(root, file_list, labels, enhance=False, train_rate=0.8):
#     train_pose_list, train_label_list, train_label2_list = [], [], []
#     test_pose_list, test_label_list, test_label2_list = [], [], []
#
#     save_test = {}
#     for i, file_i in enumerate(file_list):
#         file_i = str(root / file_i)
#         load_i = np.load(file_i)
#
#         if i == 0:
#             real_tmp = load_i['real']
#             train_pose_, test_pose_ = select_pose(real_tmp, train_rate)
#             if enhance:
#                 train_pose_, test_pose_ = enh(train_pose_, 8), enh(test_pose_, 8)
#             train_label_ = np.ones((len(train_pose_, )))  # 真假label
#             test_label_ = np.ones(len(test_pose_))
#             train_label2 = np.ones((len(train_pose_, ))) * i  # 详细label 0 为真
#             test_label2 = np.ones(len(test_pose_)) * i
#
#             train_pose_list.append(train_pose_)
#             test_pose_list.append(test_pose_)
#             train_label_list.append(train_label_)
#             test_label_list.append(test_label_)
#             train_label2_list.append(train_label2)
#             test_label2_list.append(test_label2)
#             save_test['real'] = test_pose_
#
#         fake_tmp = load_i['fake']
#         train_pose_, test_pose_ = select_pose(fake_tmp, train_rate)
#         if enhance:
#             train_pose_, test_pose_ = enh(train_pose_, 8), enh(test_pose_, 8)
#         train_label_ = np.zeros((len(train_pose_, )))
#         test_label_ = np.zeros(len(test_pose_, ))
#         train_label2 = np.ones(len(train_pose_)) * (i + 1)
#         test_label2 = np.ones(len(test_pose_)) * (i + 1)
#
#         train_pose_list.append(train_pose_)
#         test_pose_list.append(test_pose_)
#         train_label_list.append(train_label_)
#         test_label_list.append(test_label_)
#         train_label2_list.append(train_label2)
#         test_label2_list.append(test_label2)
#         save_test[labels[i]] = test_pose_  # 保存test数据，用于计算FID
#
#     train_pose_ = np.concatenate(train_pose_list, axis=0)
#     train_label_ = np.concatenate(train_label_list, axis=0)
#     test_pose_ = np.concatenate(test_pose_list, axis=0)
#     test_label_ = np.concatenate(test_label_list, axis=0)
#     train_label2 = np.concatenate(train_label2_list, axis=0)
#     test_label2 = np.concatenate(test_label2_list, axis=0)
#
#     return train_pose_, train_label_, train_label2, test_pose_, test_label_, test_label2, save_test


def build_group(real_, fake_, index_, mode):
    real_ = real_[index_]  # (n,T,22,2)
    fake_ = fake_[index_]

    label1 = []
    pose_list = []

    if mode == 'train':
        real_ = enh(real_, 1)
        fake_ = enh(fake_, 1)
    else:
        pass

    # pose
    pose_list.append(real_)
    pose_list.append(fake_)

    # label1
    label1.append(np.ones(len(real_)))  # real-1
    label1.append(np.zeros(len(fake_)))  # fake-0

    out_pose = np.concatenate(pose_list, axis=0)
    out_label1 = np.concatenate(label1, axis=0)

    return out_pose, out_label1


def build_dataset(real_data_, fake_data_, train_index_, test_index_):
    """
    :param test_index_:
    :param train_index_:
    :param real_data_:
    :param fake_data_:
    :return:
    """
    train_data_, train_label_1 = build_group(real_data_, fake_data_, train_index_, 'train')
    test_data_, test_label_1 = build_group(real_data_, fake_data_, test_index_, 'test')

    return train_data_, train_label_1, test_data_, test_label_1


def val(dev_data, classifier, device, args_):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    FP = []
    sample_pool = []
    for i, batch in enumerate(dev_data):
        dance, label = map(lambda x: x.to(device), batch)
        dance = dance.float()
        label = label.long()

        pred, _ = classifier(dance)
        loss = F.cross_entropy(pred, label, args_.w)
        avg_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

        size += batch[0].shape[0]

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                     accuracy,
                                                                     corrects,
                                                                     size))
    return accuracy


def train(train_data, dev_data, classifier, args_, sheet_ind):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, classifier.parameters()), lr=args_.lr)
    steps = 0
    val_train = []
    val_test = []
    for epoch in range(1, args_.epochs + 1):
        classifier.train()
        train_pool = []
        for i, batch in enumerate(train_data):
            dance, label = map(lambda x: x.to(args_.device), batch)
            label = label.long()
            dance = dance.float()
            optimizer.zero_grad()
            pred, _ = classifier(dance)
            loss = F.cross_entropy(pred, label, weight=args_.w)
            loss.backward()
            optimizer.step()
            steps += 1

            corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
            train_acc = 100.0 * corrects / batch[0].shape[0]
            train_pool.append(train_acc.detach().cpu().numpy())

        train_pool = np.array(train_pool).mean()
        val_train.append(train_pool)

        print(f'epoch{epoch} - loss: {loss:.6f} - acc: {train_pool:.6f}')

        # evaluate the model on test set at each epoch
        dev_acc = val(dev_data, classifier, args_.device, args_)
        val_test.append(dev_acc.detach().cpu().numpy().mean())

        # if epoch % args_.save_interval == 0:
        #     name = args_.save_dir / ('epoch' + str(epoch) + '.pt')
        #     torch.save(classifier, name)

    content = np.stack((val_train, val_test), axis=1)
    content = pd.DataFrame(content, columns=['train', 'test'])
    if sheet_ind == 0:
        content.to_excel(args_.check, index=True, sheet_name='1')
    else:
        writer = pd.ExcelWriter(args_.check, engine='openpyxl', mode='a')
        content.to_excel(writer, sheet_name=str(sheet_ind+1), index=True)
        writer.save()
        writer.close()


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--segment', type=int, default=3)  # todo
    parser.add_argument('--model', type=str, default='Cnn')
    parser.add_argument('--root_dir', metavar='PATH',
                        default='../../../../results/path2pose/assessment_long_pose_public/epoch20000/nrds_10fold/')  # todo dir
    args = parser.parse_args()
    args.num_seg = 4
    np.random.seed(1)

    if not Path(args.root_dir).exists():
        Path(args.root_dir).mkdir(parents=True)
    args.check = args.root_dir + '/segment' + str(args.segment) + '.xlsx'

    """ 数据预处理 """
    # load data
    npz_load = '../../../../results/path2pose/v11_public_attncn_220812_202737/recursive/40000/long_pose.npz'
    data_load = np.load(npz_load)
    real_data = data_load['real_seg']
    fake_data = data_load['fake_seg']
    real_data = real_data - real_data[:, :, [0], [11], :][:, :, :, np.newaxis]
    fake_data = fake_data - fake_data[:, :, [0], [11], :][:, :, :, np.newaxis]
    real_data = real_data.reshape((real_data.shape[0], real_data.shape[1], real_data.shape[2], 44))
    fake_data = fake_data.reshape((fake_data.shape[0], fake_data.shape[1], fake_data.shape[2], 44))

    real_data = real_data[:, args.segment]
    fake_data = fake_data[:, args.segment]

    # split into train set and test set
    L = len(real_data)
    win = 100  # n_test
    N = 10
    step = math.floor((L - win) / (N - 1))
    for i in tqdm(range(N)):
        print(f'start cross validation {i} th ...')
        test_index = np.arange(win) + i * step
        train_index = np.array(list(set(list(np.arange(len(real_data)))).difference(set(test_index))))
        train_pose, train_label, test_pose, test_label = \
            build_dataset(real_data, fake_data, train_index, test_index)

        """ training """
        train_loader = prepare_dataloader(train_pose, train_label, args.batch_size)
        test_loader = prepare_dataloader(test_pose, test_label, args.batch_size)

        args.device = torch.device('cuda')
        args.w = torch.tensor([1, 1]).float().to(args.device)  # todo

        model_pool = {'Rnn': Classifier_Rnn,
                      'Cnn': Classifier_Cnn_35}
        print('model:' + args.model)

        model = model_pool[args.model]()
        model = model.to(args.device)
        train(train_loader, test_loader, model, args, i)
