"""
1. train
    python main.py --mode train --save_dir ../results/train/ --train_npz ../database/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz
2. test
    python main.py --mode test --load_model ../results/train/AttnCnNet/model/epoch_20000.pt --train_npz ../database/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz
3. synthesize long pose sequence
    python main.py --mode recursive --load_model ../results/train/AttnCnNet/model/epoch_20000.pt --recursive_npz ../database/recursive_public.npz
"""
import os
import random
import torch
import argparse
import numpy as np
from pathlib import Path
from tools.dataset import LarvaDataset, load_data
from test_model import test
from recursive_new import recursive

""" backbone """
from backbone_lsgan import Backbone_LSGAN

""" generator """
from models.generator_attn_cn import Gen_attn_cn
from models.generator_gcn import Gen_gcn
from models.generator_rnn import Gen_rnn
from models.discriminator import Disc_cnn_40


## parameters
parser = argparse.ArgumentParser()
parser.add_argument('--description', type=str, default='AttnCnNet')  # using description as root directory name
parser.add_argument('--mode', type=str, default='train')  # ['train', 'reuse', 'test', 'recursive']

parser.add_argument('--save_dir', metavar='PATH', default='../../results/')
parser.add_argument('--load_model', type=str, default='../../results/AttnCnNet/model/epoch_20000.pt')  # load model for reuse, test, recursive

parser.add_argument('--train_npz', type=str, default='../../dataset/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz')  # data of short sequence for traning
parser.add_argument('--recursive_npz', type=str, default='../../dataset/recursive_public.npz')  # data for long pose synthesis

parser.add_argument('--gpu_ids', type=int, default=0, help='gpu id to use')  # gpu-id

parser.add_argument('--log_train_items', type=str, nargs='+', default=['epoch', 'd_loss', 'g_loss_disc', 'g_loss_path'])
parser.add_argument('--log_val_items', type=str, nargs='+', default=['epoch', 'loss_disc', 'loss_path'])

parser.add_argument('--epoch', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--check_epochs', type=int, metavar='N', default=1000)
parser.add_argument('--validation_epochs', type=int, metavar='N', default=100)
parser.add_argument('--visual_epochs', type=int, metavar='N', default=1000)
parser.add_argument('--visual_num', type=int, metavar='N', default=10)
parser.add_argument('--seed', type=int, default=1234)  # random seed

parser.add_argument('--base_points', type=int, nargs='+', default=11)  # base point of displacement; 11-head, 0-tail

parser.add_argument('--dim', type=int, default=2)  # dimension
parser.add_argument('--n_points', type=int, default=22)  # number of key points

""" choose model """
parser.add_argument('--generator', type=str, default='AttnCn')  # ['AttnCn','Gcn','Rnn']

""" generator """
parser.add_argument('--gen_bn', type=bool, default=True)
parser.add_argument('--gen_dp', type=float, default=0)

parser.add_argument('--d_rnn_out', type=int, default=100)
parser.add_argument('--d_rnn_hidden', type=int, default=100)

parser.add_argument('--d_mix_hidden', type=int, default=256)
parser.add_argument('--d_mix_out', type=int, default=100)
parser.add_argument('--d_noise', type=int, default=2)
parser.add_argument('--noise_lambda', type=float, default=0.1)

parser.add_argument('--enc_d_model', type=int, default=200)
parser.add_argument('--enc_d_k', type=int, default=64)
parser.add_argument('--enc_d_v', type=int, default=64)
parser.add_argument('--enc_d_inner', type=int, default=512)
parser.add_argument('--enc_n_layers', type=int, default=2)
parser.add_argument('--enc_n_heads', type=int, default=8)
parser.add_argument('--enc_dropout', type=float, default=0.1)

# decoder-attncn
parser.add_argument('--dec_d_model', type=int, default=100)
parser.add_argument('--dec_d_k', type=int, default=64)
parser.add_argument('--dec_d_v', type=int, default=64)
parser.add_argument('--dec_d_inner', type=int, default=128)
parser.add_argument('--dec_n_heads', type=int, default=4)
parser.add_argument('--dec_t_hidden', type=int, default=64)
parser.add_argument('--dec_t_kernel', type=int, default=5)
parser.add_argument('--dec_dropout', type=float, default=0)

# decoder-rnn
parser.add_argument('--dec_d_hidden', type=int, default=200)

""" discriminator """
parser.add_argument('--disc_sn', type=bool, default=True)
parser.add_argument('--disc_dp', type=float, default=0)

parser.add_argument('--lambda_recon', type=float, default=0.1)  # gain of path reconstruction loss
parser.add_argument('--lr_g', type=float, default=1e-4)  # lr
parser.add_argument('--lr_d', type=float, default=5e-4)  # lr
parser.add_argument('--lrf', type=float, default=0.5)  # exponential damping
parser.add_argument('--seq_max_len', type=int, default=100)

args = parser.parse_args()
args.d_pose = args.dim * args.n_points

## setup
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
print('CUDA_VISIBLE: ', args.gpu_ids)
device = torch.device('cuda')

## dataloader
if args.mode in ['train', 'reuse', 'test']:
    train_data, test_data = load_data(args.train_npz, args.base_points)
    # from enhance import enh
    # test_data = enh(test_data)

    train_loader = torch.utils.data.DataLoader(LarvaDataset(train_data),
                                               num_workers=0, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(LarvaDataset(test_data),
                                              num_workers=0, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=False)
else:
    data_load = np.load(args.recursive_npz, allow_pickle=True)
    data_dict = data_load['cat_dict'].item()

    init_list = data_dict['pose0']
    guide_list = data_dict['path']
    truth_list = data_dict['pose']

## model
gen_dict = {
    'AttnCn': Gen_attn_cn,
    'Gcn': Gen_gcn,
    'Rnn': Gen_rnn,
}
gen = gen_dict[args.generator](args, device=device)
disc = Disc_cnn_40(args, device=device)
print(gen._get_name(), ' - params num is ', sum(p.numel() for p in gen.parameters()))
print(disc._get_name(), ' - params num is ', sum(p.numel() for p in disc.parameters()))

bone = Backbone_LSGAN(args, gen, disc, device)

""" train """
if args.mode in ['train', 'reuse']:
    bone.train(train_loader, test_loader)
elif args.mode in ['test']:
    test(bone, test_loader, test_data, plot_num=10, form='.jpg')
elif args.mode in ['recursive']:
    plot_id = [0, 100, 200, 300]  # index of sample to be plotted
    recursive(bone, args, init_list, guide_list, truth_list, plot_id=plot_id)
