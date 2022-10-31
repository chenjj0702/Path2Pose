import numpy as np


def enh(data, enh_num=8):
    b, t, d_pose = data.shape
    data = data.reshape(b, t, 22, 2)
    angles = np.linspace(0, np.pi * 2, num=enh_num, endpoint=False)

    d_list = []
    for ag in angles:
        w = np.array([[np.cos(ag), np.sin(ag)], [-np.sin(ag), np.cos(ag)]])
        v2 = data.reshape(-1, 2)  # (N,2)
        vw = np.matmul(v2, w).reshape(data.shape)
        vw = vw.reshape(b, t, d_pose)
        d_list.append(vw)
    d_list = np.concatenate(d_list, axis=0)
    return d_list
