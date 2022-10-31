import numpy as np


def enh(data_dict, enh_num=8):
    out_dict = {}
    angles = np.linspace(0, np.pi * 2, num=enh_num, endpoint=False)

    for k_, v_ in data_dict.items():
        d_list = []
        for ag in angles:
            w = np.array([[np.cos(ag), np.sin(ag)], [-np.sin(ag), np.cos(ag)]])
            v2 = v_.reshape(-1, 2)  # (N,2)
            vw = np.matmul(v2, w).reshape(v_.shape)
            d_list.append(vw)
        d_list = np.concatenate(d_list, axis=0)
        out_dict[k_] = d_list.astype(np.float32)
    return out_dict
