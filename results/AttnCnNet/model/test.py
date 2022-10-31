import torch
import argparse

pt = torch.load('epoch_20000.pt')
pt_dict = vars(pt['args'])
pt_dict['description'] = 'AttnCnNet'
pt_dict['train_npz'] = '../../dataset/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz'
pt_dict['recursive_npz'] = '../../dataset/recursive_public.npz'
pt_dict['save_dir'] = '../../results/AttnCnNet'
pt_dict['checkpoints_dir'] = '../../results/AttnCnNet/model'
pt_dict['val_dir'] = '../../results/AttnCnNet/val'
pt_dict['lambda_recon'] = 0.1

torch.save(pt, 'epoch_20000_2.pt')
