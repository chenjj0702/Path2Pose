"""
对循环生成的长数据分段计算FID
input :
"""
import numpy as np
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
        if self.labels is not None:
            return self.dances[index], self.labels[index], self.detail_labels[index]
        else:
            return self.dances[index]


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
    real_ = real_[index_]  # (n,4,T,22,2)
    fake_ = fake_[index_]

    label1 = []
    label2 = []
    pose_list = []
    for k in range(real_.shape[1]):
        real_k = real_[:, k]
        fake_k = fake_[:, k]
        if mode == 'train':
            real_k = enh(real_k, 1)
            fake_k = enh(fake_k, 1)
        else:
            pass

        # pose
        pose_list.append(real_k)
        pose_list.append(fake_k)

        # label1
        label1.append(np.ones(len(real_k)))
        label1.append(np.zeros(len(fake_k)))

        # label2
        label2.append(np.ones(len(real_k)) * (k + 1))
        label2.append(np.zeros(len(fake_k)))

    out_pose = np.concatenate(pose_list, axis=0)
    out_label1 = np.concatenate(label1, axis=0)
    out_label2 = np.concatenate(label2, axis=0)

    return out_pose, out_label1, out_label2


def build_dataset(real_data_, fake_data_, train_index_, test_index_):
    """
    设定测试集数量，将数据集分成训练集和测试集，并制定每个样本的二分类 label1 和多分类标签 label2
    :param test_index_:
    :param train_index_:
    :param real_data_:
    :param fake_data_:
    :return:
    """
    train_data_, train_label_1, train_label_2 = build_group(real_data_, fake_data_, train_index_, 'train')
    test_data_, test_label_1, test_label_2 = build_group(real_data_, fake_data_, test_index_, 'test')

    return train_data_, train_label_1, train_label_2, test_data_, test_label_1, test_label_2


def val(dev_data, classifier, device, args_):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    FP = []
    sample_pool = []
    for i, batch in enumerate(dev_data):
        dance, label, detail_label = map(lambda x: x.to(device), batch)
        dance = dance.float()
        label = label.long()
        detail_label = detail_label.float()

        pred, _ = classifier(dance)
        loss = F.cross_entropy(pred, label, args_.w)
        avg_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

        # 找出识别为真的神经元
        pred_real = torch.max(pred, 1)[1].view(-1).detach().cpu().numpy()

        detail_label = detail_label.detach().cpu().numpy()
        # detail_label_pool = set(np.sort(detail_label))
        # detail_label_pool = list(detail_label_pool)[1:]  # 1:attn_cn; 2:gcn; 3:rnn
        detail_label_pool = np.arange(1, args_.num_seg+1).tolist()

        fp_in_batch = np.zeros(len(detail_label_pool))
        for j, label_i in enumerate(detail_label_pool):
            tmp_class = detail_label == label_i
            fp_in_batch[j] = np.stack((pred_real.astype(bool), tmp_class), axis=1).all(axis=1).sum()

        FP.append(fp_in_batch)
        sample_pool.append(len(pred_real))
        size += batch[0].shape[0]
    FP = np.stack(FP, axis=0)
    FP = np.sum(FP, axis=0)
    FP = FP / np.stack(sample_pool, axis=0).sum()

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                     accuracy,
                                                                     corrects,
                                                                     size))
    print(f"假阳性: {'-'.join(args_.labels)} : {FP}")
    return accuracy, FP


def train(train_data, dev_data, classifier, args_):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, classifier.parameters()), lr=args_.lr)
    steps = 0
    val_train = []
    for epoch in range(1, args_.epochs + 1):
        classifier.train()
        train_pool = []
        for i, batch in enumerate(train_data):
            dance, label, _ = map(lambda x: x.to(args_.device), batch)
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
        with open(str(args_.log_train), 'a', newline='') as p:
            writer_ = csv.writer(p)
            writer_.writerow([epoch, train_pool])

        # evaluate the model on test set at each epoch
        dev_acc, FP = val(dev_data, classifier, args_.device, args_)
        fp_rate = FP / np.sum(FP)
        with open(str(args_.log_test), 'a', newline='') as p:
            writer_ = csv.writer(p)
            writer_.writerow([epoch, dev_acc.detach().cpu().numpy()] + FP.tolist() + fp_rate.tolist())

        if epoch % args_.save_interval == 0:
            name = args_.save_dir / ('epoch' + str(epoch) + '.pt')
            torch.save(classifier, name)


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--model', type=str, default='Cnn')  # todo [Rnn, Cnn, CnRnn]
    parser.add_argument('--root_dir', metavar='PATH',
                        default='../../../../results/path2pose/assessment_long_pose_public/epoch20000_dataset340')  # todo 目录
    args = parser.parse_args()
    args.num_seg = 4
    np.random.seed(1)

    """ 数据预处理 """
    # load data
    npz_load = '../../../../results/path2pose/v11_public_attncn_220812_202737/recursive/40000/long_pose.npz'
    data_load = np.load(npz_load)
    real_data = data_load['real_seg']
    fake_data = data_load['fake_seg']
    real_data = real_data - real_data[:, :, [0], [11], :][:, :, :, np.newaxis]  # 将每一小段数据的首帧平移到原点
    fake_data = fake_data - fake_data[:, :, [0], [11], :][:, :, :, np.newaxis]
    real_data = real_data.reshape((real_data.shape[0], real_data.shape[1], real_data.shape[2], 44))
    fake_data = fake_data.reshape((fake_data.shape[0], fake_data.shape[1], fake_data.shape[2], 44))

    real_data = real_data[:340]
    fake_data = fake_data[:340]

    # split into train set and test set
    L = len(real_data)
    win = 50
    N = 10
    step = math.floor((L - win) / (N - 1))
    n_test = 50
    for i in range(N):
        print(f'start cross validation {i} th ...')
        test_index = np.arange(win) + i * step
        train_index = np.array(list(set(list(np.arange(len(real_data)))).difference(set(test_index))))
        train_pose, train_label, train_detail_label, test_pose, test_label, test_detail_label = \
            build_dataset(real_data, fake_data, train_index, test_index)

        """ file set up """
        model_list = ['1', '2', '3', '4']
        args.save_dir = Path(args.root_dir) / ('cross_validation_' + str(i))
        if not args.save_dir.exists():
            args.save_dir.mkdir(parents=True)

        args.log_train = args.save_dir / 'log_train.csv'
        with open(str(args.log_train), 'w') as p:
            writer = csv.writer(p)
            writer.writerow(['epoch', 'acc'])

        args.log_test = args.save_dir / 'log_test.csv'
        with open(str(args.log_test), 'w') as p:
            writer = csv.writer(p)
            tmp1 = ['FP_' + x for x in model_list]
            tmp2 = ['FP_rate_' + x for x in model_list]
            writer.writerow(['epoch', 'acc'] + tmp1 + tmp2)

        # save test data
        save_path = args.save_dir / 'data_for_fid.npz'
        np.savez(save_path, test_pose=test_pose, test_label1=test_label, test_label2=test_detail_label)

        """ training """
        train_loader = prepare_dataloader(train_pose, train_label, train_detail_label, args.batch_size)
        test_loader = prepare_dataloader(test_pose, test_label, test_detail_label, args.batch_size)

        args.device = torch.device('cuda')
        args.w = torch.tensor([1, 1]).float().to(args.device)  # todo
        args.labels = model_list

        model_pool = {'Rnn': Classifier_Rnn,
                      'Cnn': Classifier_Cnn_35}
        print('model:' + args.model)

        model = model_pool[args.model]()
        model = model.to(args.device)
        train(train_loader, test_loader, model, args)
