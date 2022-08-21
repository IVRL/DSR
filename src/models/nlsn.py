import os
import torch
import torch.nn as nn
from .rcan import CALayer, RCAB, ResidualGroup, MeanShift, default_conv, Upsampler
from .attention import NonLocalAttention, NonLocalSparseAttention

__all__ = ['nlsn']

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    

class NLSN(nn.Module):
    def __init__(self, 
                 scale,
                 pretrained,
                 n_resblocks=32,
                 n_feats=256,
                 res_scale=0.1,
                 n_hashes=4,
                 chunk_size=144,
                 conv=default_conv):
        super(NLSN, self).__init__()

        kernel_size = 3 
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 1
        n_colors = 3
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [NonLocalSparseAttention(
            channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale)]         

        for i in range(n_resblocks):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i+1)%8==0:
                m_body.append(NonLocalSparseAttention(
                    channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        if pretrained:
            self.load_pretrained()

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
    
    
    def load_pretrained(self, map_location=None, strict=False):
        
        # pretrained_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                             'pretrained',
        #                                             'NLSN_x4.pt'
        #                                             ))
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)
        # print('load from x4 pre-trained model')
        own_state = self.state_dict()
        state_dict =  torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'pretrained',
                                                    'NLSN_x4.pt'
                                                    ))
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
        print('load from x4 pre-trained model')
        

def nlsn(scale, pretrained=False):
    model = NLSN(scale=4, pretrained=pretrained)
    
    return model