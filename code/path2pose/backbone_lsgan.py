import os
import math
import time
import datetime
import shutil
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools.visualize import visualize_mul_seq
from tools.dir_tools import Log, save_args

matplotlib.use('Agg')


def plot_disc_dist(dist_dict):
    epoch_i, dir_path = dist_dict['epoch'], dist_dict['dir']
    del dist_dict['epoch'], dist_dict['dir']

    for k, v in dist_dict.items():
        real, fake = v[0], v[1]
        real, fake = real.reshape(-1), fake.reshape(-1)
        minv = np.min(np.concatenate((real, fake)))
        maxv = np.max(np.concatenate((real, fake)))
        plt.figure()
        plt.hist(real, range=[minv, maxv], bins=20, color='red', label='real')
        plt.hist(fake, range=[minv, maxv], bins=20, color='cyan', label='fake')
        plt.legend()
        name_ = os.path.join(dir_path, str(epoch_i).zfill(5) + k + '.png')
        plt.savefig(name_)
        plt.close()


class Backbone_LSGAN:
    def __init__(self, args, gen, disc, device):
        self.args = args
        self.device = device
        self.epoch_start = 1

        """ build model """
        self.gen = gen
        self.disc = disc
        self.criterion = nn.MSELoss()

        if args.mode == 'train':
            t = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            args.save_dir = Path(args.save_dir) / (args.description + '_' + t)
            if not args.save_dir.exists():
                args.save_dir.mkdir()
            args.save_dir = str(args.save_dir)
            args.checkpoints_dir = os.path.join(args.save_dir, 'model')
            if not os.path.exists(args.checkpoints_dir):
                os.mkdir(args.checkpoints_dir)
            args.val_dir = os.path.join(args.save_dir, 'val')
            if not os.path.exists(args.val_dir):
                os.mkdir(args.val_dir)

            save_args(args)
            print(args, sep='; ')

        else:
            check_point_load = torch.load(args.load_model, map_location=self.device)

            if args.mode == 'reuse':
                self.epoch_start = check_point_load['epoch'] + 1
                self.args.save_dir = check_point_load['args'].save_dir
                self.args.checkpoints_dir = check_point_load['args'].checkpoints_dir
                self.args.val_dir = check_point_load['args'].val_dir
            elif args.mode in ['test', 'recursive']:
                self.args.obj_dir = Path(check_point_load['args'].save_dir) / args.mode / str(check_point_load['epoch'])
                if not self.args.obj_dir.exists():
                    self.args.obj_dir.mkdir(parents=True)
                self.args.obj_dir = str(self.args.obj_dir)
            else:
                raise EOFError

            args_load = vars(check_point_load['args'])
            args_new = vars(self.args)
            for i, (k, v) in enumerate(args_load.items()):
                if k not in args_new.keys():
                    print('args_new has no key: ', k)
                    continue
                if args_load[k] != args_new[k]:
                    print(f'args_new[{k}]={args_new[k]} , load_args[{k}={args_load[k]}]')
                    continue
            input('press any key to continue ...')

            self.gen.load_state_dict(check_point_load['gen'])
            if args.mode == 'reuse':
                self.disc.load_state_dict(check_point_load['disc'])

        if args.mode in ['train', 'reuse']:
            """ logger """
            self.log_train = Log(args, log_type='train', items=args.log_train_items)
            self.log_val = Log(args, log_type='val', items=args.log_val_items)
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            """ optimizer """
            self.optimizer_G = torch.optim.Adam(filter(lambda x: x.requires_grad, self.gen.parameters()),
                                                lr=args.lr_g, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.disc.parameters()),
                                                lr=args.lr_d, betas=(0.5, 0.999))
            """ lr """
            ratio_lr = lambda x: math.exp(-0.0005 * x) * (1 - args.lrf) + args.lrf
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=ratio_lr)
            self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=ratio_lr)

        self.gen = self.gen.to(device)
        if args.mode in ['train', 'reuse']:
            self.disc = self.disc.to(device)

    def train(self, train_loader, val_loader):
        disc_loss_fun = nn.MSELoss()
        for epoch_i in tqdm(range(self.epoch_start, self.epoch_start + self.args.epoch)):
            self.gen.train()
            self.disc.train()

            start_time = time.time()
            loss_list = []
            res_real, res_fake = [], []
            for batch_i, batch in enumerate(train_loader):
                pose_glb, path_glb = map(lambda x: x.to(self.device), batch)
                if len(pose_glb) < self.args.batch_size:
                    continue

                """ train disc """
                guide_path = path_glb[:, 5:]  # (b,T,2)
                gold_pose = pose_glb[:, 5:]  # (b,t,44)
                hist_pose = pose_glb[:, :5]

                output = self.gen(guide_path, hist_pose)
                res_real = self.disc(gold_pose, path_glb, hist_pose)
                res_fake = self.disc(output.detach(), path_glb, hist_pose)
                label_real = torch.ones_like(res_real).to(self.device) * 0.9
                label_fake = torch.ones_like(res_fake).to(self.device) * 0.1
                disc_loss_real = disc_loss_fun(res_real, label_real)
                disc_loss_fake = disc_loss_fun(res_fake, label_fake)
                loss_disc = 0.5 * (disc_loss_real + disc_loss_fake)

                self.optimizer_D.zero_grad()
                loss_disc.backward()
                self.optimizer_D.step()

                """ train gen """
                res_g_fake = self.disc(output, path_glb, hist_pose)
                g_loss_disc = disc_loss_fun(res_g_fake, label_real)
                fake_path = self.move(output, self.args.base_points)
                g_loss_path = self.criterion(fake_path, guide_path)

                g_loss = g_loss_disc + g_loss_path * self.args.lambda_recon

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                loss_list.append([int(epoch_i),
                                  np.float16(loss_disc.detach().cpu().numpy()),
                                  np.float16(g_loss_disc.detach().cpu().numpy()),
                                  np.float16(g_loss_path.detach().cpu().numpy()),
                                  ])

            self.writer.add_scalar(os.path.join('lr', 'generator'), self.optimizer_G.state_dict()['param_groups'][0]['lr'], epoch_i)
            self.writer.add_scalar(os.path.join('lr', 'discriminator'), self.optimizer_D.state_dict()['param_groups'][0]['lr'], epoch_i)
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

            end_time = time.time()
            # log
            loss_list = np.array(loss_list).mean(0)
            log_dict = {self.args.log_train_items[x]: loss_list[x] for x in range(len(loss_list))}
            self.log_train.update(log_dict)
            print('time used ', end_time - start_time)
            print(log_dict)

            self.writer.add_scalar(os.path.join('train', 'loss_disc'), loss_list[1], epoch_i)
            self.writer.add_scalar(os.path.join('train', 'g_loss_disc'), loss_list[2], epoch_i)
            self.writer.add_scalar(os.path.join('train', 'g_loss_path'), loss_list[3], epoch_i)

            # save model
            if epoch_i % self.args.check_epochs == 0 or epoch_i == 1:
                checkpoint = {'epoch': epoch_i,
                              'args': self.args,
                              'gen': self.gen.state_dict(),
                              'disc': self.disc.state_dict(),
                              'opt_g': self.optimizer_G.state_dict(),
                              'opt_dis': self.optimizer_D.state_dict(),
                              }
                filename = os.path.join(self.args.checkpoints_dir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)

            if epoch_i % self.args.validation_epochs == 0 or epoch_i == 1:
                self.val(val_loader, self.log_val, epoch_i)

            # plot disc distribute
            if epoch_i % 100 == 0 or epoch_i == 1:
                dist_dir = os.path.join(self.args.val_dir, 'disc_dist')
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                dist_dict = {'dir': dist_dir,
                             'epoch': epoch_i,
                             'disc_full': (res_real.detach().cpu().numpy(), res_fake.detach().cpu().numpy()),
                             }
                plot_disc_dist(dist_dict)

        self.writer.close()

    def val(self, val_loader, log, epoch_i):
        self.gen.eval()
        self.disc.eval()

        cond_list = []
        out_list = []
        real_list = []
        loss_list = []
        for batch_i, batch in enumerate(val_loader):
            # prepare data
            pose_glb, path_glb = map(lambda x: x.to(self.device), batch)
            guide_path = path_glb[:, 5:]  # (b,T,2)
            gold_pose = pose_glb[:, 5:]  # (b,t,44)
            hist_pose = pose_glb[:, :5]

            with torch.no_grad():
                output = self.gen(guide_path, hist_pose)
                res_ = self.disc(output, path_glb, hist_pose)

            valid_label = torch.ones_like(res_).to(self.device) * 0.9
            loss_disc = nn.MSELoss()(res_, valid_label)
            fake_path = self.move(output, self.args.base_points)
            loss_path = self.criterion(fake_path, guide_path)

            output = torch.cat((hist_pose, output), dim=1)
            gold_seq = torch.cat((hist_pose, gold_pose), dim=1)
            out_list.append(output.detach().cpu().numpy())
            real_list.append(gold_seq.detach().cpu().numpy())
            cond_list.append(guide_path.detach().cpu().numpy())

            loss_list.append([epoch_i,
                              loss_disc.detach().cpu().numpy(),
                              loss_path.detach().cpu().numpy(),
                              ])

        """ log in excel """
        loss_list = np.stack(loss_list).mean(0)
        log_dict = {self.args.log_val_items[x]: loss_list[x] for x in range(len(loss_list))}
        log.update(log_dict)

        self.writer.add_scalar(os.path.join('val', 'loss_disc'), loss_list[1], epoch_i)
        self.writer.add_scalar(os.path.join('val', 'loss_path'), loss_list[2], epoch_i)

        """ visualize """
        if epoch_i % self.args.visual_epochs == 0 or epoch_i == 1:
            val_dir = os.path.join(self.args.val_dir, 'epoch' + str(epoch_i))
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            out = np.concatenate(out_list, axis=0)  # (n,len_seq,d_pos)
            real = np.concatenate(real_list, axis=0)
            path = np.concatenate(cond_list, axis=0)  # (n,len_seq,2)
            ids = np.arange(0, len(out), int(len(out) / self.args.visual_num))[:self.args.visual_num]

            # save
            npz_name = val_dir + '/test_results.npz'
            np.savez(npz_name, real=real, fake=out, path=path, plot_id=ids)

            # plot fake
            out_plot = out[ids]
            path_plot = path[ids]
            out_plot = out_plot.reshape((out_plot.shape[0], out_plot.shape[1], self.args.n_points, self.args.dim))
            visualize_mul_seq(out_plot, val_dir, path_plot, ids, track_points=self.args.base_points, append='fake')

            if epoch_i == 1:
                real_plot = real[ids]
                path_plot = path[ids]
                real_plot = real_plot.reshape(
                    (real_plot.shape[0], real_plot.shape[1], self.args.n_points, self.args.dim))
                visualize_mul_seq(real_plot, val_dir, path_plot, ids, track_points=self.args.base_points, append='real')

    def move(self, data, points):
        data = data.reshape(data.size(0), data.size(1), self.args.n_points, self.args.dim)
        if isinstance(points, int):
            out = data[:, :, points, :]
        elif isinstance(points, list):
            out = data[:, :, points, :].mean(2)
        else:
            raise EOFError
        return out
