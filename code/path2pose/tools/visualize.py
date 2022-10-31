import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools.func import format_trans
from tools.animation_larva import plot_pose_seq
from pathlib import Path
from tqdm import tqdm

matplotlib.use('Agg')


def plot_frame(x, append):
    # x (22,n_dim)
    assert x.ndim == 2 and x.shape[0] == 22

    # lines
    l1 = []
    x = np.append(x, np.expand_dims(x[0], axis=0), axis=0)
    for i in range(len(x) - 1):
        if append == 'real':
            l_i, = plt.plot([x[i, 0], x[i + 1, 0]], [x[i, 1], x[i + 1, 1]], 'cyan', alpha=0.8, linewidth=4)
        else:
            l_i, = plt.plot([x[i, 0], x[i + 1, 0]], [x[i, 1], x[i + 1, 1]], 'orange', alpha=0.8, linewidth=4)
        l1.append(l_i)

    l2 = []
    point_map = [[1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16],
                 [7, 15], [8, 14], [9, 13], [10, 12]]
    for a, b in point_map:
        if append == 'real':
            l_i = plt.plot([x[a, 0], x[b, 0]], [x[a, 1], x[b, 1]], 'cyan', alpha=0.8, linewidth=4)[0]
        else:
            l_i = plt.plot([x[a, 0], x[b, 0]], [x[a, 1], x[b, 1]], 'orange', alpha=0.8, linewidth=4)[0]
        l2.append(l_i)

    # head and tail
    d0 = plt.plot(x[0, 0], x[0, 1], color='red', marker='o', markersize=20)
    d12 = plt.plot(x[11, 0], x[11, 1], color='green', marker='o', markersize=20)
    # scatter
    s = plt.scatter(x[1:22, 0], x[1:22, 1], s=25, color='black', alpha=0.8)

    return d0, d12, s, l1, l2


def plot_a_img(x, fake_path, real_path, r, **kwargs):
    # x -> (c,12,2) or (24,2)
    params = dict(filename=None,
                  fig_size=(8, 8),
                  axis_state='on',
                  tick_state='on',
                  append=''
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

    plt.rcParams['figure.figsize'] = params['fig_size']
    plt.rcParams['savefig.dpi'] = 300
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.axis(params['axis_state'])
    if params['tick_state'] == 'off':
        plt.xticks([])
        plt.yticks([])
    plt.axis(r)

    if fake_path is not None:
        plt.plot(fake_path[:, 0], fake_path[:, 1], '-', color='blue', markersize=8, alpha=0.7)
    if real_path is not None:
        plt.plot(real_path[:, 0], real_path[:, 1], '-', color='red', markersize=8, alpha=0.7)
    plot_frame(x, params['append'])
    if params['filename'] is not None:
        img_savename = params['filename']
        plt.savefig(img_savename)
        plt.close('all')


def visualize_seq(x, track_points, path, save_path, form, append=''):

    if path is not None:
        minx, maxx = min(x[:, :, 0].min(), path[:, 0].min()), max(x[:, :, 0].max(), path[:, 0].max())
        miny, maxy = min(x[:, :, 1].min(), path[:, 1].min()), max(x[:, :, 1].max(), path[:, 1].max())
    else:
        minx, maxx = x[:, :, 0].min(), x[:, :, 0].max()
        miny, maxy = x[:, :, 1].min(), x[:, :, 1].max()
    deltx = (maxx - minx) * 1.05 / 2
    delty = (maxy - miny) * 1.05 / 2
    mid_x = (maxx + minx) / 2
    mid_y = (maxy + miny) / 2
    r = [mid_x-deltx, mid_x+deltx, mid_y-delty, mid_y+delty]
    if deltx < delty:
        size_x = 5
        size_y = size_x / deltx * delty
    else:
        size_y = 5
        size_x = size_y / delty * deltx

    if track_points is None:
        track = None
    elif isinstance(track_points, int):
        track = x[:, track_points, :]  # (T,2)
    else:
        track = x[:, track_points, :].mean(1)

    if 'jpg' in form:
        if not Path(save_path).exists():
            Path(save_path).mkdir()
        for i, x_i in enumerate(x):
            name = os.path.join(save_path, str(i).zfill(2) + '.jpg')
            plot_a_img(x_i, track, path, r, filename=name, axis_state='on', tick_state='off', fig_size=(size_x, size_y), append=append)
    elif 'gif' in form:
        save_name = save_path + '.gif'
        plot_pose_seq(x, save_name, traj_real=path, range=r, traj_flag=track_points, axis_state='on',
                      fig_size=(size_x, size_y), tick_state='on', append=append)

    # if Path(save_path).is_dir():
    #     for i, x_i in enumerate(x):
    #         name = os.path.join(save_path, str(i).zfill(2) + '.jpg')
    #         plot_a_img(x_i, track, path, r, filename=name, axis_state='on', tick_state='off', fig_size=(size_x, size_y), append=append)
    # else:
    #     plot_pose_seq(x, save_path, traj_real=path, range=r, traj_flag=track_points, axis_state='on', fig_size=(size_x, size_y), tick_state='on', append=append)


def visualize_mul_seq(x, save_path, path=None, id_list=None, form='.gif', track_points=11, append=''):
    """
    :param x: (n,T,22,2)
    :param path: (n,T,2)
    :param save_path:
    :param id_list: (n,)
    :param form: ".jpg" ".png"
    :param track_points:
    :param append: "real" "fake"
    :return:
    """
    assert x.ndim == 4 and x.shape[-1] == 2 and x.shape[-2] == 22
    if path is not None:
        assert path.ndim == 3 and path.shape[0] == x.shape[0]
    if id_list is not None:
        assert len(x) == len(id_list)

    for i, x_i in enumerate(tqdm(x)):
        if id_list is None:
            num_str = i
        else:
            num_str = id_list[i]
        name = os.path.join(save_path, str(num_str).zfill(2)+'_'+append)

        # if form in ['.jpg']:
        #     name = os.path.join(save_path, str(num_str).zfill(2)+'_'+append)
        #     if not os.path.exists(name):
        #         os.mkdir(name)
        # elif form in ['.gif']:
        #     name = os.path.join(save_path, str(num_str).zfill(2) + '_' + append + form)
        # else:
        #     raise EOFError

        if path is None:
            path_i = None
        else:
            path_i = path[i]

        visualize_seq(x_i, track_points, path_i, name, form, append)

