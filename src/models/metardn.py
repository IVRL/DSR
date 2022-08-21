import os
import torch
import torch.nn as nn
import math

from .rdn import RDB

__all__ = ['metardn', 'input_matrix_wpn', 'metardn_altitude']

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Pos2Weight(nn.Module):
    def __init__(self,inC, nfeature=3, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(nfeature,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
        
    def forward(self,x):
        output = self.meta_block(x)
        return output

class MetaRDN(nn.Module):
    def __init__(self,
                 scale,
                 nfeature=3,
                 G0 = 64,
                 D=16,
                 C=8,
                 G=64,
                 n_colors=3,
                 RDNkSize=3,
                 rgb_range=255
                 ):
        super(MetaRDN, self).__init__()
        
        
        self.scale = scale
        kSize = RDNkSize

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        self.D = D
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        ## position to weight
        self.P2W = Pos2Weight(inC=G0, nfeature=nfeature)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, meta_mat):
        
        N = x.size(0)
        x *= 255.

        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        
        local_weight = self.P2W(meta_mat)   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)

        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
        
        local_weight = local_weight.reshape(x.size(0), x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(0, 2,4,1,3,5,6)
        local_weight = local_weight.reshape(x.size(0), scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.reshape(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.reshape(x.size(0),3, scale_int*x.size(2), scale_int*x.size(3))
        out = self.add_mean(out)

        return out/255
    

        
def metardn(scale, pretrained=False):
    model =  MetaRDN(scale=scale)
    if pretrained:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                     'pretrained',
                                                     'metasr_1000.pt')
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
        model.load_state_dict(pretrained_dict, strict=False)
        print(f'load from pre-trained model: {os.path.basename(path)}')
    return model

def metardn_altitude(scale, pretrained=False):
    model =  MetaRDN(scale=scale, nfeature=1)
    if pretrained:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                     'pretrained',
                                                     'metasr_1000.pt')
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
        model.load_state_dict(pretrained_dict, strict=False)
        print(f'load from pre-trained model: {os.path.basename(path)}')
    return model
        
    
def input_matrix_wpn(inH, inW, scale, altitude=1.0, only_altitude=False):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    
    outH, outW = int(scale*inH), int(scale*inW)
        
    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH,  scale_int, 1, dtype=torch.bool)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int, dtype=torch.bool)
    
    scale_mat = torch.zeros(1,1)
    scale_mat[0,0] = 1.0/scale
    scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)
    
    altitude_mat = torch.zeros(1,1)
    altitude_mat[0,0] = altitude
    altitude_mat = torch.cat([altitude_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)


    ####projection  coordinate  and caculate the offset 
    h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag,  0] = True
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = True
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
    mask_mat = mask_mat.eq(2)
    pos_mat = pos_mat.contiguous().view(-1,2)
   
    pos_mat = torch.cat((scale_mat.view(-1,1), pos_mat),1)
    
    if only_altitude:
        return altitude_mat.view(-1,1), mask_mat
    else:
        pos_mat = torch.cat((altitude_mat.view(-1,1), pos_mat),1)
        return pos_mat, mask_mat  ## outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
    
    