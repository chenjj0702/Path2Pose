import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path


def plot_frame(x):
    # x (24,n_dim)
    assert x.ndim == 2 and x.shape[0] == 22 and x.shape[1] == 2

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
    return dots, lines, line_list


def plot_pose_seq(X, save_name, traj_real=None, **kwargs):
    params = dict(interv=50,
                  range=None,
                  backend='Agg',
                  fps=17,
                  fig_size=(6, 6),
                  fig_dpi=100,
                  saved_dpi=100,
                  axis_state='on',
                  tick_state='on',
                  frame_state='on',
                  traj_flag=None)

    for key, val in kwargs.items():
        params[key] = val

    matplotlib.use(params['backend'])  # set matplotlib backend

    plt.rcParams['figure.figsize'] = params['fig_size']
    plt.rcParams['figure.dpi'] = params['fig_dpi']  # !!! figure size
    plt.rcParams['savefig.dpi'] = params['saved_dpi']  # !!! figure saved size
    fig = plt.figure()
    plt.axis(params['axis_state'])
    if params['tick_state'] == 'off':
        plt.xticks([])
        plt.yticks([])

    # figure size
    if params['range'] is not None:
        plt.axis(params['range'])
    else:
        minx, maxx = np.min(X[:, :, 0]), np.max(X[:, :, 0])
        miny, maxy = np.min(X[:, :, 1]), np.max(X[:, :, 1])
        deltx = maxx - minx
        delty = maxy - miny
        delt = max(deltx, delty)
        plt.axis([minx, minx + delt, miny, miny + delt])

    # plot traj
    if params['traj_flag'] is not None:
        n = int(params['traj_flag'])
        plt.plot(X[:, n, 0], X[:, n, 1], 'go', markersize=3)

    if traj_real is not None:
        plt.plot(traj_real[:, 0], traj_real[:, 1], 'r-', lw=2, alpha=0.5)

    # plot 1 frame
    dots, lines, pair_list = plot_frame(X[0])

    def update(t):
        t = int(t)
        for i_, dot in enumerate(dots):
            dot.set_xdata(X[t, i_, 0])
            dot.set_ydata(X[t, i_, 1])

        for i_, line in enumerate(lines):
            line.set_xdata(X[t, pair_list[i_], 0])
            line.set_ydata(X[t, pair_list[i_], 1])
        return dots, lines

    ani = animation.FuncAnimation(fig, func=update, frames=len(X), interval=params['interv'])
    mywriter = FFMpegWriter(fps=params['fps'])
    ani.save(save_name, writer=mywriter)
    plt.close('all')
