"""

"""
import sys

sys.path.append('models')
import numpy as np
import torch
import torch.nn as nn
from models.attention import Encoder


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
class RNN_Decoder(nn.Module):
    def __init__(self, device, d_input, d_hidden, bn):
        super(RNN_Decoder, self).__init__()
        self.bn = bn
        self.dense1 = nn.Linear(d_input, d_hidden)
        self.rnn1 = nn.LSTM(d_hidden, d_hidden, batch_first=True)
        self.rnn2 = nn.LSTM(d_hidden, d_hidden, batch_first=True)
        self.norm1 = nn.BatchNorm1d(d_hidden)
        self.norm2 = nn.BatchNorm1d(d_hidden)
        self.norm3 = nn.BatchNorm1d(d_hidden)

        self.dense2 = nn.Sequential(nn.Linear(d_hidden, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 44),
                                    nn.Tanh()
                                    )

    def forward(self, x):
        # x -> (b,t,d)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        b, t, _ = x.size()

        out = self.dense1(x)  # (b,t,d_hidden)
        if self.bn:
            out = out.permute(0, 2, 1)  # (b,c,t)
            out = self.norm1(out)
            out = out.permute(0, 2, 1)

        out, (h, c) = self.rnn1(out)
        if self.bn:
            out = out.permute(0, 2, 1)  # (b,c,t)
            out = self.norm2(out)
            out = out.permute(0, 2, 1)

        out, _ = self.rnn2(out, (h, c))
        if self.bn:
            out = out.permute(0, 2, 1)
            out = self.norm3(out)
            out = out.permute(0, 2, 1)

        out = self.dense2(out)  # (b,t,44)
        out = out.reshape(b, t, 22, 2)  # (b,t,22,2)
        out = out.permute(0, 3, 1, 2)  # (b,2,t,22)
        return out


## transformer without scheduled sampling
class Gen_rnn(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.device = device
        self.args = args

        d_mix_input = args.dim + args.d_rnn_out + args.d_noise

        self.pos_enc = PoseEncoder(device, args.d_pose, args.d_rnn_out, args.d_rnn_hidden, args.gen_bn)

        self.mix = Mix(d_mix_input, args.d_mix_out, args.d_mix_hidden, args.gen_bn)

        self.encoder = Encoder(device, args.d_mix_out, args.enc_d_model, args.enc_d_k, args.enc_d_v, args.enc_d_inner,
                               args.enc_n_heads, args.enc_n_layers, args.enc_dropout, args.seq_max_len)

        self.decoder = RNN_Decoder(device, args.enc_d_model+args.d_rnn_out+args.d_pose, args.dec_d_hidden, args.gen_bn)

        self.bias_linear = nn.Linear(args.dim, args.dim)

    # dynamic auto-condition + self-attention mask
    def forward(self, cond_seq, tgt_seq):
        # cond_seq (b,T,d_music)
        # tgt_seq (b,T,d_pose)
        bsz, seq_len, _ = cond_seq.size()

        # pre encoder
        c_pose = self.pos_enc(tgt_seq)  # (b,d_pre)
        c_pose = c_pose.unsqueeze(1).repeat(1, seq_len, 1)  # (b,t,d_pre)

        # mix
        dec_inp = torch.cat((cond_seq, c_pose), dim=-1)  # (b,t,d_music+d_pre)
        if self.args.d_noise:
            noise = torch.randn(bsz, seq_len, self.args.d_noise).to(self.device) * self.args.noise_lambda
            dec_inp = torch.cat((dec_inp, noise), dim=-1)  # (b,t,d)
        out = self.mix(dec_inp)  # (b,t,d_mix)

        # encoder
        enc_hidden, *_ = self.encoder(out)  # (b,t,d_model)
        last_pose = tgt_seq[:, [-1], :].repeat(1, seq_len, 1)  # (b,T,v,2)
        out = torch.cat((enc_hidden, c_pose, last_pose), dim=-1)  # (b,t,d_pre+d_model+d_pose)

        # decoder
        out = self.decoder(out)  # (b,2,t,v)

        # post
        out = out.permute(0, 2, 3, 1).contiguous()  # (b,t,v,2)
        out = self.bias_linear(out)  # (b,t,24,2)
        out = out.reshape(bsz, seq_len, self.args.d_pose)  # (b,t,44)

        return out
