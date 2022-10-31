"""
synthesize long pose sequence by recursively generating and joining short sequences
plot long pose sequence and save data as npz
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tools.visualize_recursive import visualize_mul_seq


def get_long(bone, args, sub_dir, hist, guide_list, plot=False):
    gen = bone.gen
    gen.eval()
    device = bone.device

    hist = torch.from_numpy(hist)[:5].reshape(5, args.n_points, args.dim).float().unsqueeze(0).to(device)  # (1,5,22,2)
    out_list = [hist.detach().cpu().numpy()]
    path_list = []
    for i, path_i in enumerate(guide_list):
        path_i = torch.from_numpy(path_i).float().to(device)
        pose_shift = hist[0, 0, args.base_points, :]
        hist = hist - pose_shift
        guide = path_i[1:] - pose_shift
        guide = guide.unsqueeze(0).float().to(device)  # (1,35,2)

        with torch.no_grad():
            hist = hist.reshape(hist.size()[0], hist.size()[1], -1)
            output = gen(guide, hist)  # (1,35,44)
        output = output.reshape(output.size()[0], output.size()[1], args.n_points, args.dim)  # (1,35,22,2)
        output = output + pose_shift

        out_list.append(output.detach().cpu().numpy())
        path_list.append(path_i.detach().cpu().numpy())

        hist = output[:, -5:, :, :]

    out = np.concatenate(out_list, 1)  # (1,L,22,2)
    out = out.reshape((out.shape[0], out.shape[1], -1))  # (1,L,44)
    path = np.concatenate(path_list, 0)
    path = path[np.newaxis, :, :]  # (1,L,2)

    ids = [0]
    if plot is True:
        obj_dir = Path(args.obj_dir)
        obj_dir = obj_dir / sub_dir
        if not obj_dir.exists():
            obj_dir.mkdir(parents=True)
        form = '.jpg'
        fake_plot = out[ids]
        path_plot = path[ids]
        fake_plot = fake_plot.reshape(
            (fake_plot.shape[0], fake_plot.shape[1], args.n_points, args.dim))
        visualize_mul_seq(fake_plot, path_plot, obj_dir, ids, track_points=args.base_points, append='full', form=form)

    return out.squeeze()


def recursive(bone, args, init_list, guide_list, real_set_list, plot_id=None):
    out_list = []
    for i in tqdm(range(len(guide_list))):
        hist_i = init_list[i].reshape(5, 44)
        guide_i = guide_list[i]
        title = str(i).zfill(3) + '_syn'
        if i in plot_id:
            plot_flag = True
        else:
            plot_flag = False
        out = get_long(bone, args, title, hist_i, guide_i, plot=plot_flag)
        out_list.append(out)

        if plot_flag is True:
            truth_seg = real_set_list[i]
            truth_1 = [x for x in truth_seg]
            truth_1 = np.concatenate(truth_1, axis=0)
            truth_1 = truth_1[np.newaxis, :]  # (1,140,22,2)

            obj_tmp = Path(args.obj_dir)
            sub_dir = str(i).zfill(3) + '_real'
            obj_fin = obj_tmp / sub_dir
            if not obj_fin.exists():
                obj_fin.mkdir(parents=True)
            visualize_mul_seq(truth_1, None, obj_fin, [0], track_points=args.base_points, append='full',
                              form='.jpg')

    out = np.array(out_list)

    fake_seg_list = []
    start_i = 5
    end_i = start_i + 35
    while end_i <= out.shape[1]:
        tmp = out[:, start_i: end_i].reshape(out.shape[0], 35, 22, 2)
        fake_seg_list.append(tmp)

        start_i += 35
        end_i += 35
    fake_seg_list = np.stack(fake_seg_list, axis=1)  # (n,4,35,22,2)
    real_set_list = np.array(real_set_list)

    save_name = os.path.join(args.obj_dir, 'long_pose_sequence.npz')
    np.savez(save_name, real_seg=real_set_list, fake_seg=fake_seg_list,
             describe='The long sequence without initial poses are divided into 4 segments without coordinates adjustment. '
                      'The first pose are not at the coordinate origin')
