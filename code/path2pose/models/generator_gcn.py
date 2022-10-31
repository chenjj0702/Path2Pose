"""

"""
import sys

sys.path.append('models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Encoder
from models.gcn import st_gcn
from models.graph import Graph


##
def format_trans(X):
    """ (batch,timesteps,22,c) <-> (batch,timesteps,c,11,2)   """
    if len(X.size()) == 4 and X.size(-2) == 22:
        batch, T, n_points, c = X.size()
        x1 = X.transpose(2, 3)  # (batch,T,c,22)
        x1 = x1.reshape(batch, T, c, 2, 11)  # (batch,T,c,2,11)
        tmp1 = x1[:, :, :, 0, :]  # 0 - 10 (batch,T,c,11)
        tmp2 = torch.flip(x1[:, :, :, 1, :], dims=[-1])  # 21 - 11 (batch,T,c,11)
        out = torch.stack((tmp1, tmp2), -1)  # (batch,T,c,11,2)

    elif len(X.size()) == 5 and X.size(-1) == 2 and X.size(-2) == 11:
        batch, T, c, _, _ = X.size()
        tmp1 = X[:, :, :, :, 0]  # 0 - 10 (batch,T,c,11)
        tmp2 = torch.flip(X[:, :, :, :, 1], dims=[-1])  # 11 - 21 (batch,T,c,11)
        out = torch.cat((tmp1, tmp2), dim=-1)  # (b,t,c,22)
        out = out.transpose(-1, -2)
    else:
        print('fun-format_trans: input dim is wrong')
        raise EOFError
    return out


##
class PoseEncoderRnn(nn.Module):
    def __init__(self, device, d_input, d_output, d_hidden, bn=True):
        super(PoseEncoderRnn, self).__init__()

        self.dense1 = nn.Linear(d_input, d_hidden)
        self.rnn1 = nn.LSTM(d_hidden, d_hidden, batch_first=True)
        self.rnn2 = nn.LSTM(d_hidden, d_hidden, batch_first=True)
        self.dense2 = nn.Linear(d_hidden, d_output)

        self.bn = bn
        self.norm1 = nn.BatchNorm1d(d_hidden)
        self.norm2 = nn.BatchNorm1d(d_hidden)
        self.norm3 = nn.BatchNorm1d(d_hidden)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        # x -> (n,t,44)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        assert len(x.size()) == 3
        out = self.dense1(x)  # (b,t,d_hidden)
        if self.bn:
            out = out.permute(0, 2, 1)
            out = self.norm1(out)
            out = out.permute(0, 2, 1)
        out = self.act(out)

        out, (h, v) = self.rnn1(out)  # (b,t,d_hidden)
        if self.bn:
            out = out.permute(0, 2, 1)
            out = self.norm2(out)
            out = out.permute(0, 2, 1)

        out, _ = self.rnn2(out, (h, v))
        if self.bn:
            out = out.permute(0, 2, 1)
            out = self.norm3(out)
            out = out.permute(0, 2, 1)
        out = out[:, -1, :]  # (b,d_hidden)

        out = self.dense2(out)
        return out


class PoseEncoderCnn(nn.Module):
    def __init__(self, device, d_input, d_output):
        super(PoseEncoderCnn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(d_input, 64, kernel_size=2, stride=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 128, kernel_size=(4, 4, 1), stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, d_output, kernel_size=(1, 4, 1), stride=1),
        )

    def forward(self, x):
        out = self.net(x)  # (b,d_out,1,1,1)
        return out.squeeze()


##
class Mix(nn.Module):
    def __init__(self, d_input, d_output, d_hidden, bn=True):
        super(Mix, self).__init__()
        self.bn = bn

        self.l1 = nn.Linear(d_input, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_output)
        self.act = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(d_hidden)
        self.norm2 = nn.BatchNorm1d(d_output)

        if d_input == d_output:
            self.res = nn.Identity()
        else:
            self.res = nn.Linear(d_input, d_output)

    def forward(self, x):
        out = self.l1(x)  # (b,t,d_hidden)
        if self.bn:
            out = out.permute(0, 2, 1).contiguous()  # (b,d_hidden,t)
            out = self.norm1(out)
            out = out.permute(0, 2, 1).contiguous()  # (b,t,d_hidden)
        out = self.res(x) + self.l2(self.act(out))  # (b,t,d_out)

        if self.bn:
            out = out.permute(0, 2, 1).contiguous()  # (b,d_out,t)
            out = self.norm2(out)
            out = out.permute(0, 2, 1).contiguous()  # (b,t,d_out)

        out = self.act(out)
        return out


##
class GCN_Decoder(nn.Module):
    def __init__(self, device, d_input, dropout=0, bn=True):
        super(GCN_Decoder, self).__init__()

        self.bn = bn

        self.emb = nn.Sequential(
            nn.Linear(d_input, 128),
        )

        self.gcn1 = st_gcn(in_channels=128, out_channels=64, kernel_size=(3, 2),
                           dropout=dropout, residual=True, bn=True)

        self.gcn2 = st_gcn(in_channels=64, out_channels=32, kernel_size=(3, 2),
                           dropout=dropout, residual=True, bn=True)

        self.norm = nn.BatchNorm2d(128)
        self.act = nn.LeakyReLU()

    def forward(self, x, A):
        b, t, v, c = x.size()

        out = x.reshape(b, -1, c)  # (b,tv,c)
        out = self.emb(out).reshape(b, t, v, -1).permute(0, 3, 1, 2).contiguous()  # (b,256,t,v)
        if self.bn:
            out = self.norm(out)
        out = self.act(out)

        out = self.gcn1(out, A)
        out = self.gcn2(out, A)

        return out


## transformer without scheduled sampling
class Gen_gcn(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.device = device
        self.args = args

        d_mix_input = args.dim + args.d_rnn_out + args.d_noise

        self.pos_enc = PoseEncoderRnn(device, args.d_pose, args.d_rnn_out, args.d_rnn_hidden, args.gen_bn)

        self.mix = Mix(d_mix_input, args.d_mix_out, args.d_mix_hidden, args.gen_bn)

        self.encoder = Encoder(device, args.d_mix_out, args.enc_d_model, args.enc_d_k, args.enc_d_v, args.enc_d_inner,
                               args.enc_n_heads, args.enc_n_layers, args.enc_dropout, args.seq_max_len)

        self.decoder = GCN_Decoder(device=device,
                                   d_input=args.enc_d_model+args.d_rnn_out+args.dim,
                                   dropout=args.gen_dp,
                                   bn=args.gen_bn)

        A = Graph(args.n_points,
                  [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
                   (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
                   (20, 21), (21, 0), (1, 21), (2, 20), (3, 19), (4, 18), (5, 17), (6, 16), (7, 15), (8, 14),
                   (9, 13), (10, 12)], 1, strategy='distance', max_hop=1).A
        self.A = torch.tensor(A).float().to(device)

        self.bias_linear = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    # dynamic auto-condition + self-attention mask
    def forward(self, cond_seq, tgt_seq):
        # cond_seq (b,T,d_music)
        # tgt_seq (b,t,d_pose)
        bsz, seq_len, _ = cond_seq.size()

        # pre encoder
        c_pose = self.pos_enc(tgt_seq)  # (b,d_pre)
        c_pose = c_pose.unsqueeze(1).repeat(1, seq_len, 1)  # (b,T,d_pre)

        # mix
        dec_inp = torch.cat((cond_seq, c_pose), dim=-1)  # (b,T,d_music+d_pre)
        if self.args.d_noise:
            noise = torch.randn(bsz, seq_len, self.args.d_noise).to(self.device) * self.args.noise_lambda
            dec_inp = torch.cat((dec_inp, noise), dim=-1)  # (b,t,d)
        out = self.mix(dec_inp)  # (b,T,d_mix)

        # encoder
        enc_hidden, *_ = self.encoder(out)  # (b,T,d_model)
        enc_hidden = torch.cat((enc_hidden, c_pose), dim=-1)  # (b,t,d_model+d_pre)
        enc_hidden = enc_hidden.unsqueeze(2).repeat(1, 1, self.args.n_points, 1)  # (b,T,v,d_model+d_pre)
        last_pose = tgt_seq[:, [-1], :].repeat(1, seq_len, 1).reshape(bsz, seq_len, self.args.n_points, self.args.dim)  # (b,T,v,2)
        out = torch.cat((enc_hidden, last_pose), dim=-1)  # (b,T,v,d_model+d_pre+dim)

        # decoder
        out = self.decoder(out, self.A)  # (b,2,t,v)

        # post
        out = out.permute(0, 2, 3, 1).contiguous()  # (b,t,v,2)
        out = self.bias_linear(out)  # (b,t,24,2)
        out = out.reshape(bsz, seq_len, self.args.d_pose)  # (b,t,44)

        return out
