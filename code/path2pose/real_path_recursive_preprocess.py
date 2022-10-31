"""
基于145帧的长路径生成循环生成的预处理数据
"""
import numpy as np
from tools.visualize import visualize_mul_seq

root_dir = 'D:/Data/DLSpace/database/'
npz_name = root_dir + 'pubic_larva_refine_pose_head_enh1_win145_step40_test0.npz'
data_load = np.load(npz_name, allow_pickle=True)
real_145 = data_load['full_dict'].item()['train']

n_points = 22
dim = 2
base_points = 11
form = '.jpg'
obj_dir = root_dir

# ids = [100, 50]
#
# real_plot = real_145[ids]
# path_plot = path_145[ids]
# real_plot = real_plot.reshape(
#     (real_plot.shape[0], real_plot.shape[1], n_points, dim))
# visualize_mul_seq(real_plot, path_plot, obj_dir, ids, track_points=base_points, append='real', form=form)
#
# # fake
# fake_plot = fake_145[ids]
# path_plot = path_145[ids]
# fake_plot = fake_plot.reshape(
#     (fake_plot.shape[0], fake_plot.shape[1], n_points, dim))
# visualize_mul_seq(fake_plot, path_plot, obj_dir, ids, track_points=base_points, append='fake', form=form)

# 分段
n_pre = 5
n_step = 35
point = 11
out_dict = {}
pre_pose = []
pose_list = []
path_list = []
for i in range(len(real_145)):
    real_i = real_145[i].reshape(145, 22, 2)

    pose_list_i = []
    traj_list_i = []
    start_i = n_pre
    end_i = n_pre + n_step
    while end_i <= len(real_i):
        traj_list_i.append(real_i[start_i - 1: end_i, point, :])
        pose_list_i.append(real_i[start_i: end_i])

        start_i = start_i + n_step
        end_i = end_i + n_step

    pre_pose.append(real_i[:n_pre])
    pose_list.append(np.array(pose_list_i)[:4])
    path_list.append(np.array(traj_list_i)[:4])

out_dict['pose0'] = pre_pose
out_dict['pose'] = pose_list
out_dict['path'] = path_list

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.figure()
y0 = path_list[0]
for yo_i in y0:
    plt.plot(yo_i[:, 0], yo_i[:, 1], '-o')
plt.show()


save_file = obj_dir + f'recursive_public.npz'  # !!!
np.savez(save_file,
         cat_dict=out_dict,
         )