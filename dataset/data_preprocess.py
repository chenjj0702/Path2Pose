"""
preprocess pose data from DLPose dataset

python data_preprocess.py --save_path ./ --dataset ./sm_points.json --n_test 1000 --window 40 --step 10
"""
import sys
sys.path.append('../code/utils')
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import json
import math
import argparse
# from plot_pose import plot_pose_seq
from visualize import visualize_mul_seq


## func
def remove(data, rm_seg: dict):
    for k, v in rm_seg.items():
        if len(v) == 0:
            del data[k]
        else:
            data[k] = data[k][v[0]:v[-1] + 1]
    return data


def split_seq(data: dict, win_: int, step_: int, points):
    # data - dict{'seg_name': (T,22,2)}

    out_list = []
    for (k_s, v_s) in data.items():  # v: (T,22,2)
        v_s = np.array(v_s)
        # plt.figure()
        # for d in v_s[0]:
        #     plt.plot(d[0], d[1], 'o')
        #     plt.pause(0.5)

        # reverse  todo 将从头部开始的顺时针转化为从尾部开始的逆时针
        tmp_v1 = v_s[:, :11, :]  # (0-10)
        tmp_v2 = v_s[:, 11:, :]  # (11-21)
        v_s = np.concatenate((tmp_v2, tmp_v1), axis=1)
        v_s = np.flip(v_s, axis=1)
        v_s = np.concatenate((v_s[:, [-1], :], v_s[:, :-1, :]), axis=1)
        # plt.figure()
        # for d in v_s[0]:
        #     plt.plot(d[0], d[1], 'o')
        #     plt.pause(0.5)

        if len(v_s) < win_:
            continue
        else:
            starti, endi = 0, win_
            while endi <= len(v_s):
                tmp_ = v_s[starti:endi]  # (T,22,2)
                if isinstance(points, int):
                    tmp0 = tmp_[0, points, :]
                elif isinstance(points, list):
                    tmp0 = tmp_[[0], points, :].mean(0, keepdims=True)
                else:
                    raise EOFError
                tmp_ = tmp_ - tmp0
                out_list.append(tmp_)
                starti += step_
                endi += step_
    out = np.stack(out_list, axis=0)  # (n,T,22,2)
    return out


def split_train_test(data, test_rate):
    # data - (n,T,22,2)
    if isinstance(test_rate, int):
        n_test_ = test_rate
    elif test_rate < 1:
        n_test_ = int(len(data) * test_rate)
    else:
        raise EOFError

    test_id = np.random.choice(len(data), n_test_, False)
    train_id = list(set(list(range(len(data)))).difference(set(test_id)))

    out_dict = {'train': data[train_id],
                'test': data[test_id]}
    return out_dict


def split_all(data, win_, step_, base_point, num_test):
    """
    @param data:
    @param win_:
    @param step_:
    @param base_point:
    @param num_test:
    @return:
    """
    # sort segments
    x = sorted(data.items(), key=lambda i: len(i[1]), reverse=True)

    # main
    out_dict = {'train': [], 'test': []}
    n_test_glb = 0
    for k_s, v_s in x:
        v_s = np.array(v_s)  # (n,22,2)
        n_train_threshold = math.floor((math.floor(len(v_s) / 4) - win_) / step_) + 1
        # n_all = math.floor((len(v_s) - win_) / step_) + 1

        # reverse
        tmp_v1 = v_s[:, :11, :]  # (0-10)
        tmp_v2 = v_s[:, 11:, :]  # (11-21)
        v_s = np.concatenate((tmp_v2, tmp_v1), axis=1)
        v_s = np.flip(v_s, axis=1)
        v_s = np.concatenate((v_s[:, [-1], :], v_s[:, :-1, :]), axis=1)

        n_local_train = 0
        if len(v_s) < win_:
            continue
        else:
            starti, endi = 0, win_
            long_test = 0
            flag = 'norm'
            while endi <= len(v_s):
                tmp_ = v_s[starti:endi]  # (T,22,2)
                if isinstance(base_point, int):
                    tmp0 = tmp_[0, base_point, :]
                elif isinstance(base_point, list):
                    tmp0 = tmp_[[0], base_point, :].mean(0, keepdims=True)
                else:
                    raise EOFError
                tmp_ = tmp_ - tmp0

                if (n_test_glb >= num_test) | (flag == 'train'):
                    out_dict['train'].append(tmp_)
                    n_local_train += 1
                    starti += step_
                    endi += step_
                elif flag == 'test':
                    out_dict['test'].append(tmp_)
                    n_test_glb += 1
                    starti += win_
                    endi += win_
                elif endi <= math.floor(len(v_s) / 4):
                    out_dict['train'].append(tmp_)
                    n_local_train += 1
                    starti += step_
                    endi += step_
                else:
                    if len(v_s) < endi - step_ + win_:
                        flag = 'train'
                    else:
                        flag = 'test'
                        starti += win_ - step_
                        endi += win_ - step_
                        start_test = starti
                    continue

    for k_, v_ in out_dict.items():
        out_dict[k_] = np.array(v_)

    return out_dict


def enh(data_dict, enh_num, enh_test=True):
    out_dict = {}
    angles = np.linspace(0, np.pi * 2, num=enh_num, endpoint=False)

    for k_, v_ in data_dict.items():
        if k_ == 'test' and enh_test is False:  # 测试集不需要数据增强
            out_dict[k_] = v_
        else:
            d_list = []
            for ag in angles:
                w = np.array([[np.cos(ag), np.sin(ag)], [-np.sin(ag), np.cos(ag)]])
                v2 = v_.reshape(-1, 2)  # (N,2)
                vw = np.matmul(v2, w).reshape(v_.shape)
                d_list.append(vw)
            d_list = np.concatenate(d_list, axis=0)
            out_dict[k_] = d_list
    return out_dict


def scale(data_dict, delt=None):
    if delt is None:
        tmp_list = []
        for _, v_ in data_dict.items():
            tmp_list.append(v_)

        tmp_list = np.concatenate(tmp_list, axis=0)
        minx, miny = np.min(tmp_list[:, :, :, 0]), np.min(tmp_list[:, :, :, 1])
        maxx, maxy = np.max(tmp_list[:, :, :, 0]), np.max(tmp_list[:, :, :, 1])
        delt = max([abs(minx), abs(miny), abs(maxx), abs(maxy)]) * 1.02

    out_dict = {}
    for k_, v_ in data_dict.items():
        vs = v_ / delt
        out_dict[k_] = vs
    return out_dict, delt


def move(data_dict, points):
    pose_dict = {}
    path_dict = {}

    for k_, v_ in data_dict.items():
        if isinstance(points, int):
            tmp_ = v_[:, :, [points], :].repeat(v_.shape[2], axis=2)  # (n,T,24,2)
        elif isinstance(points, list):
            tmp_ = v_[:, :, points, :].mean(2, keepdims=True).repeat(v_.shape[2], axis=2)  # (n,T,24,2)
        else:
            raise EOFError
        pose_dict[k_] = v_ - tmp_
        path_dict[k_] = v_[:, :, [points], :].mean(2)  # (n,T,2)
    return pose_dict, path_dict


def plot_a_img(x, r_axis=None, fig_size=(4, 4), dpi=300, num=True):
    # x (22,2)
    assert x.ndim == 2 and x.shape[0] == 22 and x.shape[1] == 2

    plt.figure(figsize=fig_size, dpi=dpi)
    x_min, x_max, y_min, y_max = x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max()
    r_x, r_y = x_max - x_min, y_max - y_min
    r = max(r_x, r_y) * 1.3
    center = x.mean(0)

    if r_axis is None:
        x_min, x_max, y_min, y_max = center[0] - r / 2, center[0] + r / 2, center[1] - r / 2, center[1] + r / 2
        plt.axis([x_min, x_max, y_min, y_max])
    else:
        plt.axis(r_axis)

    # lines
    line_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                 [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 0],
                 [1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15], [8, 14], [9, 13], [10, 12]]
    lines = []
    for pair in line_list:
        l_i, = plt.plot(x[pair, 0], x[pair, 1], 'cyan', alpha=0.8, linewidth=2)
        lines.append(l_i)

    # dots
    dots = []
    for i_, x_i in enumerate(x):
        if i_ == 0:
            dot, = plt.plot(x_i[0], x_i[1], color='red', marker='o', markersize=8)
        elif i_ == 11:
            dot, = plt.plot(x_i[0], x_i[1], color='green', marker='o', markersize=8)
        else:
            dot, = plt.plot(x_i[0], x_i[1], color='black', marker='o', markersize=5, alpha=1)
        dots.append(dot)

    # text
    if num:
        text = x + (x - center) / abs(x - center) * r * 0.03
        for j, d_i in enumerate(x):
            plt.text(text[j, 0], text[j, 1], str(j))

    return dots, lines, line_list


##
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--dataset', type=str, default='./sm_points.json')
parser.add_argument('--n_test', type=int, default=1000)
parser.add_argument('--window', type=int, default=40)
parser.add_argument('--step', type=int, default=10)
parser.add_argument('--basepoint', type=int, default=11)
parser.add_argument('--enhance', type=int, default=4)
parser.add_argument('--scale_rate', type=int, default=None)  # 859.7914501964874

args = parser.parse_args()
np.random.seed(1234)

with open(args.dataset, 'r') as f:
    raw_data = json.load(f)

# remove
rm_segs = {'1102_00': [],
           '2205_00': [0, 238],
           '2700_00': [0, 376],
           '3101_00': []}
raw_data = remove(raw_data, rm_segs)

# split
train_test_dict = split_all(raw_data, args.window, args.step, args.basepoint, args.n_test)

# enhance
enh_dict = enh(train_test_dict, enh_num=args.enhance, enh_test=False)

# scale
scale_dict, rate = scale(enh_dict, args.scale_rate)
for k, v in scale_dict.items():
    scale_dict[k] = v.astype(np.float32)

# save
save_file = args.save_path + f'pubic_larva_refine_pose_head_enh{args.enhance}_win{args.window}_step{args.step}_test{args.n_test}_.npz'  # !!!
np.savez(save_file,
         full_dict=scale_dict,
         rate=rate
         )