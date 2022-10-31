import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools.func import format_trans
from tools.animation_larva import plot_pose_seq
from pathlib import Path
from tqdm import tqdm

matplotlib.use('Agg')


def plot_frame(x):
    # x (22,n_dim)
    assert x.ndim == 2 and x.shape[0] == 22

    # lines
    l1 = []
    x = np.append(x, np.expand_dims(x[0], axis=0), axis=0)
    for i in range(len(x) - 1):
        l_i, = plt.plot([x[i, 0], x[i + 1, 0]], [x[i, 1], x[i + 1, 1]], 'gold', alpha=1, linewidth=3)
        l1.append(l_i)

    l2 = []
    point_map = [[1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16],
                 [7, 15], [8, 14], [9, 13], [10, 12]]
    for a, b in point_map:
        l_i = plt.plot([x[a, 0], x[b, 0]], [x[a, 1], x[b, 1]], 'gold', alpha=1, linewidth=3)[0]
        l2.append(l_i)

    # head and tail
    d0 = plt.plot(x[0, 0], x[0, 1], color='red', marker='o', markersize=10)
    d12 = plt.plot(x[11, 0], x[11, 1], color='green', marker='o', markersize=10)
    # scatter
    s = plt.scatter(x[1:22, 0], x[1:22, 1], s=40, color='white', alpha=1)

    return d0, d12, s, l1, l2


def plot_a_img(x, fake_path, real_path, r, **kwargs):
    # x -> (c,12,2) or (24,2)
    params = dict(filename=None,
                  fig_size=(8, 8),
                  axis_state='on',
                  tick_state='on',
                  )
    for k, v in kwargs.items():
        params[k] = v

    if x.ndim == 3 and x.shape == (2, 11, 2):
        x = x[np.newaxis, np.newaxis, :, :, :]  # (1,1,2,12,2)
        x = format_trans(x)  # (1,1,24,2)
        x = x.squeeze()
    elif x.ndim == 2 and x.shape == (22, 2):
        pass
    else:
        raise Exception('input for func plot_a_img should be (24,2) or (2,12,2)')
    assert x.shape == (22, 2)

    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.figure(figsize=params['fig_size'])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis(params['axis_state'])
    if params['tick_state'] == 'off':
        plt.xticks([])
        plt.yticks([])
    plt.axis(r)

    if real_path is not None:
        plt.plot(real_path[:, 0], real_path[:, 1], '-', color='lightsalmon')

    if fake_path is not None:
        color_list = ['azure', 'lightskyblue', 'cyan', 'deepskyblue', 'mediumslateblue', 'blueviolet', 'violet', 'deeppink', 'crimson', 'peru']
        plt.plot(fake_path[0:5, 0], fake_path[0:5, 1], 'o', color=color_list[0], markersize=6)
        start_i = 5
        end_i = start_i + 35
        k = 1
        while end_i <= len(fake_path):
            plt.plot(fake_path[start_i: end_i, 0], fake_path[start_i: end_i, 1], 'o', color=color_list[k], markersize=6)
            start_i = start_i + 35
            end_i = end_i + 35
            k += 1

    plot_frame(x)
    if params['filename'] is not None:
        img_savename = params['filename']
        plt.savefig(img_savename)
        plt.close('all')


def visualize_seq(x, track_points, path, save_path):
    assert x.ndim == 3 and x.shape[2] == 2 and x.shape[1] == 22

    if path is not None:
        minx, maxx = min(x[:, :, 0].min(), path[:, 0].min()), max(x[:, :, 0].max(), path[:, 0].max())
        miny, maxy = min(x[:, :, 1].min(), path[:, 1].min()), max(x[:, :, 1].max(), path[:, 1].max())
    else:
        minx, maxx = x[:, :, 0].min(), x[:, :, 0].max()
        miny, maxy = x[:, :, 1].min(), x[:, :, 1].max()
    deltx = maxx - minx
    delty = maxy - miny
    delt = max(deltx, delty)
    r = [minx, minx + delt, miny, miny + delt]

    if track_points is None:
        track = None
    elif isinstance(track_points, int):
        track = x[:, track_points, :]  # (T,2)
    else:
        track = x[:, track_points, :].mean(1)

    if Path(save_path).is_dir():
        for i, x_i in enumerate(x):
            name = os.path.join(save_path, str(i).zfill(2) + '.jpg')
            plot_a_img(x_i, track, path, r, filename=name, axis_state='off', tick_state='off')
    else:
        plot_pose_seq(x, save_path, traj_real=path, range=r, traj_flag=track_points, axis_state='off', tick_state='off')


def visualize_mul_seq(x, path, save_path, id_list, form='.gif', track_points=None, append=''):
    assert x.ndim == 4 and x.shape[-1] == 2 and x.shape[-2] == 22
    # assert path.ndim == 3 and path.shape[0] == x.shape[0] and path.shape[1] == x.shape[1]
    assert len(x) == len(id_list)

    for i, x_i in enumerate(tqdm(x)):

        if form in ['.jpg']:
            name = os.path.join(save_path, str(id_list[i]).zfill(2)+'_'+append)
            if not os.path.exists(name):
                os.mkdir(name)
        elif form in ['.gif']:
            name = os.path.join(save_path, str(id_list[i]).zfill(2) + '_' + append + form)
        else:
            raise EOFError

        if path is None:
            path_i = None
        else:
            path_i = path[i]

        visualize_seq(x_i, track_points, path_i, name)

