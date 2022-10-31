"""
train the classifier for FDD
"""
import numpy as np
import torch
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from classifier import Classifier_Rnn, Classifier_Cnn_40
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
    np.random.seed(1)
    train_index = np.random.choice(len(x), n_train, replace=False)
    test_index = np.array(list(set(np.arange(len(x))).difference(train_index)))

    train_x = x[train_index]
    test_x = x[test_index]
    return train_x, test_x


def build_data(root, file_list, labels, enhance=False, train_rate=0.8):
    train_pose_list, train_label_list, train_label2_list = [], [], []
    test_pose_list, test_label_list, test_label2_list = [], [], []

    save_test = {}
    for i, file_i in enumerate(file_list):
        file_i = str(root / file_i)
        load_i = np.load(file_i)

        if i == 0:
            real_tmp = load_i['real']
            train_pose_, test_pose_ = select_pose(real_tmp, train_rate)
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
            save_test['real'] = test_pose_

        fake_tmp = load_i['fake']
        train_pose_, test_pose_ = select_pose(fake_tmp, train_rate)
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
        save_test[labels[i]] = test_pose_  # 保存test数据，用于计算FID

    train_pose_ = np.concatenate(train_pose_list, axis=0)
    train_label_ = np.concatenate(train_label_list, axis=0)
    test_pose_ = np.concatenate(test_pose_list, axis=0)
    test_label_ = np.concatenate(test_label_list, axis=0)
    train_label2 = np.concatenate(train_label2_list, axis=0)
    test_label2 = np.concatenate(test_label2_list, axis=0)

    return train_pose_, train_label_, train_label2, test_pose_, test_label_, test_label2, save_test


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
        detail_label_pool = set(np.sort(detail_label))
        detail_label_pool = list(detail_label_pool)[1:]  # 1:attn_cn; 2:gcn; 3:rnn

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
    print(f"FP_rate: {'-'.join(args_.labels)} : {FP}")
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
    parser.add_argument('--model', type=str, default='Cnn')
    parser.add_argument('--code_root_dir', metavar='PATH', default='../../../../results/path2pose/')
    parser.add_argument('--obj_dir', metavar='PATH', default='assessment_fdd_nrds_5_35_SNGAN_public/')  # todo 项目名,一级目录
    args = parser.parse_args()
    args.save_dir = args.code_root_dir + args.obj_dir + 'fdd_gcn3_20000'  # 二级目录
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    data_dir = Path('../../../../results/path2pose')  # 'D:/Data/Workspace/Dance_rev/path2pose'
    model_list = ['attn_cn', 'gcn', 'rnn']  # ['attn_cn', 'gcn', 'rnn']; ['8000', '11000', '12500', '16000', '17000', '18000', '20000'] todo
    npz_list = [  # todo
        'v11_public_attncn_220812_202737/val/epoch20000/test_results.npz',
        'v11_public_gcn3_220815_201054/val/epoch20000/test_results.npz',
        # 'v11_public_gcn2_220815_200225/val/epoch10000/test_results.npz',
        # 'v11_public_gcn_220812_203041/val/epoch10000/test_results.npz',
        'v11_public_rnn_220812_203203/val/epoch20000/test_results.npz',
    ]

    assert [model_list[i] in npz_list[i] for i in range(len(npz_list))]

    args.save_dir = Path(args.save_dir)
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
        writer.writerow(['epoch', 'acc']+tmp1+tmp2)

    train_pose, train_label, train_detail_label, test_pose, test_label, test_detail_label, save_data\
        = build_data(data_dir, npz_list, model_list, enhance=True, train_rate=0.8)
    save_path = args.save_dir / 'data_for_fdd.npz'
    np.savez(save_path, test_data=save_data)

    train_loader = prepare_dataloader(train_pose, train_label, train_detail_label, args.batch_size)
    test_loader = prepare_dataloader(test_pose, test_label, test_detail_label, args.batch_size)

    args.device = torch.device('cuda:3')  # todo gpu
    args.w = torch.tensor([1, len(npz_list)]).float().to(args.device)
    args.labels = model_list

    model_pool = {'Rnn': Classifier_Rnn,
                  'Cnn': Classifier_Cnn_40}
    print('model:' + args.model)

    model = model_pool[args.model]()
    model = model.to(args.device)
    train(train_loader, test_loader, model, args)
