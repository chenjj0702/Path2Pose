"""

"""
import sys

sys.path.append('models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Encoder
from models.attn_cn import st_attn_cn


##
class PoseEncoder(nn.Module):
    def __init__(self, device, d_input, d_output, d_hidden, bn=True):
        super(PoseEncoder, self).__init__()

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
class Attn_CN_Decoder(nn.Module):
    def __init__(self, device, d_input, d_output, s_d_model, s_d_k, s_d_v, s_d_inner, s_n_heads,
                 t_hidden=64, t_kernel=5, t_stride=1,
                 dropout=0, bn=True):
        super(Attn_CN_Decoder, self).__init__()
        self.bn = bn

        self.emb = nn.Sequential(
            nn.Linear(d_input, 256),
        )

        self.st1 = st_attn_cn(device, 256, 64, s_d_model, s_d_k, s_d_v, s_d_inner, s_n_heads, t_hidden, t_kernel, t_stride,
                              dropout=dropout, bn=bn, residual=True)

        self.st2 = st_attn_cn(device, 64, d_output, s_d_model, s_d_k, s_d_v, s_d_inner, s_n_heads, t_hidden, t_kernel, t_stride,
                              dropout=dropout, bn=bn, residual=True)

        self.act = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(256)

    def forward(self, x):
        b, t, v, c = x.size()

        out = x.reshape(b, -1, c)  # (b,tv,c)
        out = self.emb(out).reshape(b, t, v, -1).permute(0, 3, 1, 2).contiguous()  # (b,256,t,v)
        if self.bn:
            out = self.norm(out)
        out = self.act(out)  # (b,256,t,v)

        out = self.st1(out)  # (b,64,t,v)

        out = self.st2(out)  # (b,32,t,v)

        return out


## transformer without scheduled sampling
class Gen_attn_cn(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.device = device
        self.args = args

        d_mix_input = args.dim + args.d_rnn_out + args.d_noise

        self.pos_enc = PoseEncoder(device, args.d_pose, args.d_rnn_out, args.d_rnn_hidden, args.gen_bn)

        self.mix = Mix(d_mix_input, args.d_mix_out, args.d_mix_hidden, args.gen_bn)

        self.encoder = Encoder(device, args.d_mix_out, args.enc_d_model, args.enc_d_k, args.enc_d_v, args.enc_d_inner,
                               args.enc_n_heads, args.enc_n_layers, args.enc_dropout, args.seq_max_len)

        self.decoder = Attn_CN_Decoder(device, args.enc_d_model+args.d_rnn_out+args.dim, 32,
                                       args.dec_d_model, args.dec_d_k, args.dec_d_v, args.dec_d_inner, args.dec_n_heads,
                                       args.dec_t_hidden, args.dec_t_kernel, 1,
                                       args.dec_dropout, args.gen_bn)

        self.bias_linear = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, gui_path, hist_pose):
        # gui_path (b,T,d_path)
        # hist_pose (b,T,d_pose)
        bsz, seq_len, _ = gui_path.size()

        # hist encoder
        c_pose = self.pos_enc(hist_pose)
        c_pose = c_pose.unsqueeze(1).repeat(1, seq_len, 1)

        # mix
        dec_inp = torch.cat((gui_path, c_pose), dim=-1)
        if self.args.d_noise:
            noise = torch.randn(bsz, seq_len, self.args.d_noise).to(self.device) * self.args.noise_lambda
            dec_inp = torch.cat((dec_inp, noise), dim=-1)
        out = self.mix(dec_inp)

        # encoder
        enc_hidden, *_ = self.encoder(out)
        enc_hidden = torch.cat((enc_hidden, c_pose), dim=-1)
        enc_hidden = enc_hidden.unsqueeze(2).repeat(1, 1, self.args.n_points, 1)
        last_pose = hist_pose[:, [-1], :].repeat(1, seq_len, 1).reshape(bsz, seq_len, self.args.n_points, self.args.dim)
        out = torch.cat((enc_hidden, last_pose), dim=-1)

        # decoder
        out = self.decoder(out)

        # post
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.bias_linear(out)
        out = out.reshape(bsz, seq_len, self.args.d_pose)

        return out
