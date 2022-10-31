import torch.nn as nn
from attention import Encoder


# class Temp_cnn(nn.Module):
#     def __init__(self, t_k, bn):
#         super(Temp_cnn, self).__init__()
#         self.cn1 = nn.Conv2d()


class st_attn_cn(nn.Module):
    def __init__(self, device, d_input, d_output, s_d_model, s_d_k, s_d_v, s_d_inner, s_n_heads, d_t_hidden, t_kernel, t_stride=1,
                 dropout=0.1, bn=True, residual=True):
        super(st_attn_cn, self).__init__()

        assert t_kernel % 2 == 1
        padding = ((t_kernel - 1) // 2, 0)  # 保证时间点不变

        self.device = device
        self.bn = bn

        if not residual:
            self.residual = lambda x: 0.0
        elif (d_input == d_output) and (t_stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(d_input, d_output, kernel_size=1, stride=1)

        self.spatial = Encoder(device, d_input, s_d_model, s_d_k, s_d_v, s_d_inner, s_n_heads, 1, dropout, max_len=1000)

        self.temporal1 = nn.Conv2d(s_d_model, d_t_hidden, (t_kernel, 1), (t_stride, 1), padding, padding_mode='reflect')
        self.temporal2 = nn.Conv2d(d_t_hidden, d_t_hidden, (t_kernel, 1), (t_stride, 1), padding, padding_mode='reflect')

        self.dense = nn.Conv2d(d_t_hidden, d_output, 1, 1)

        self.norm1 = nn.BatchNorm2d(d_t_hidden)
        self.norm2 = nn.BatchNorm2d(d_t_hidden)
        self.norm3 = nn.BatchNorm2d(d_output)

        self.Lact = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, t, v = x.size()

        res = self.residual(x)  # (b,d_out,t,v)

        x = x.permute(0, 2, 3, 1).contiguous()  # (b,t,v,c)
        out = x.reshape(b * t, v, c)  # (bt,v,c)
        out, _ = self.spatial(out)  # (bt,v,s_d_model)
        out = out.reshape(b, t, v, -1).permute(0, 3, 1, 2)  # (b,s_d_model,t,v)

        out = self.temporal1(out)  # (b,d_t_hidden,t,v)
        if self.bn:
            out = self.norm1(out)
        out = self.Lact(out)

        out = self.temporal2(out)
        if self.bn:
            out = self.norm2(out)
        out = self.Lact(out)  # (b,d_t_hidden,t,v)

        out = self.dense(out) + res  # (b,d_out,t,v)
        if self.bn:
            out = self.norm3(out)
        out = self.dropout(self.Lact(out))

        return out
