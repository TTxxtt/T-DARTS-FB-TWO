#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""

import torch
import torch.nn as nn
from torchsummary import summary
import math
from collections import OrderedDict
import sys
from utils import random_choice, find_choice_index
# from einops.layers.torch import Rearrange

current_module = sys.modules[__name__]

debug = False

class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = (0, self.C1 // 2 ), bias =False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding = 0, bias = False, max_norm = 1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4), stride = 4),
                nn.Dropout(p = dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding = (0, 22//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = 0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8), stride = 8),
                nn.Dropout(p = dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2,
                 dropoutP = 0.25, F1=8, D = 2,
                 C1 = 125, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.firstBlocks(x)
        f = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x, f


def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

# %% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#%% support of mixConv2d
import torch.nn.functional as F

from typing import Tuple, Optional

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size, tuple):
            padding = (0,padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass = 2, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = torch.squeeze(x.permute((0,4,2,3,1)), dim = 4)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        f = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(f)
        return x, f

#%% FBMSNet_MixConv
class FBMSNet(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
        # input_size: channel x datapoint
        super(FBMSNet, self).__init__()
        self.strideFactor = 4

        self.mixConv2d = nn.Sequential(
            MixedConv2d(in_channels=9, out_channels=num_Feat, kernel_size=[(1,15),(1,31),(1,63),(1,125)],
                         stride=1, padding='', dilation=1, depthwise=False,),
            nn.BatchNorm2d(num_Feat),)
        self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat*dilatability, nChan=int(nChan))

        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        size = self.get_size(nChan, nTime)

        self.fc = self.LastBlock(size[1],nClass)

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        y = self.mixConv2d(x)
        x = self.scb(y)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x) 
        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        return c, f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        x = self.mixConv2d(data)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()

class ConvBn(nn.Module):
	def __init__(self, inplanes, outplanes, k, dilation, padding):
		super(ConvBn, self).__init__()
		
		self.op = nn.Sequential(
			nn.Conv2d(inplanes, outplanes, kernel_size=(1, k), stride=1, padding=padding, dilation=dilation, bias=False),
			# nn.BatchNorm2d(outplanes),
		)
	
	def forward(self, x):
		return self.op(x)

class Cell(nn.Module):
    def __init__(self, F1, shadow_bn):
        super(Cell, self).__init__()
        self.F1 = F1
        self.shadow_bn = shadow_bn

        self.nodes = nn.ModuleList([])
        for i in range(2):
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,1), (0,7)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,2), (0,14)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,4), (0,28)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,8), (0,56)))
         
        self.bn_list = nn.ModuleList([])
        if self.shadow_bn:
            for j in range(2):
                self.bn_list.append(nn.BatchNorm2d(self.F1))
        else:
            self.bn = nn.BatchNorm2d(self.F1)
        
    def forward(self, x, choice):
        path_ids = choice       # eg.[0, 2, 3]
        x_list = []
        
        for i, id in enumerate(path_ids):
            x_list.append(self.nodes[(len(path_ids)-1)*4+id](x))                
        # out = sum(x_list) 
        out = torch.cat(x_list, 1)

        if self.shadow_bn:
            out = self.bn_list[len(path_ids)-1](out)
        else:
            out = self.bn(out)
        return out

#%% FBMSNet_Inception
class SuperNet(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,sampling_rate=250, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
        # input_size: channel x datapoint
        super(SuperNet, self).__init__()
        self.strideFactor = 8
        
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.LowFreqCell = Cell(F1=num_Feat//3, shadow_bn=True)
        self.MidFreqCell = Cell(F1=num_Feat//3, shadow_bn=True)
        self.HighFreqCell = Cell(F1=num_Feat//3, shadow_bn=True)
        self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat*dilatability, nChan=int(nChan))
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # size = int(num_Feat*dilatability*self.strideFactor)
        size = int(num_Feat*dilatability*self.strideFactor)

        self.fc = self.LastBlock(size,nClass)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()  
    
    def forward(self, x, choice):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        
        x = torch.split(x,3,1)
        out1 = self.LowFreqCell(x[0], choice['Low'])
        out2 = self.MidFreqCell(x[1], choice['Mid'])
        out3 = self.HighFreqCell(x[2], choice['High'])
        out = torch.cat([out1,out2,out3], dim=1)
        # out = self.TemporalCell(x, choice['op'])
        out = self.scb(out)
        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[3] / self.strideFactor)])
        out = self.temporalLayer(out)
        f = torch.flatten(out, start_dim=1)
        c = self.fc(f)
        
        return c,f



class FBNASCell(nn.Module):
    def __init__(self, F1, shadow_bn, choice):
        super(FBNASCell, self).__init__()
        self.F1 = F1
        self.shadow_bn = shadow_bn
        self.opt_choice = choice
        self.nodes = nn.ModuleList([])

        for i in range(2):
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,1), (0,7)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,2), (0,14)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,4), (0,28)))
            self.nodes.append(ConvBn(3, self.F1//(i+1), 15, (1,8), (0,56)))
            
        self.bn_list = nn.ModuleList([])
        if self.shadow_bn:
            for j in range(2):
                self.bn_list.append(nn.BatchNorm2d(self.F1))
        else:
            self.bn = nn.BatchNorm2d(self.F1)
        
    def forward(self, x):
        path_ids = self.opt_choice      # eg.[0, 2, 3]
        # op_ids = self.opt_choice['op']   # eg.[1, 1, 2]
        x_list = []

        for i, id in enumerate(path_ids):
            x_list.append(self.nodes[(len(path_ids)-1)*4+id](x))            
        # out = sum(x_list)
        out = torch.cat(x_list, 1)
        if self.shadow_bn:
            out = self.bn_list[len(path_ids)-1](out)
        else:
            out = self.bn(out)
        return out

#%% FBMSNet_Inception
class FBNASNet(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
        
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,sampling_rate=250, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, opt_choice=None, dropoutP=0.5, *args, **kwargs):
        # input_size: channel x datapoint
        super(FBNASNet, self).__init__()
        self.strideFactor = 8
        
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.LowFreqCell = FBNASCell(F1=num_Feat//3, shadow_bn=True, choice=opt_choice['Low'])
        self.MidFreqCell = FBNASCell(F1=num_Feat//3, shadow_bn=True, choice=opt_choice['Mid'])
        self.HighFreqCell = FBNASCell(F1=num_Feat//3, shadow_bn=True, choice=opt_choice['High'])
        self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat*dilatability, nChan=int(nChan))
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # size = int(num_Feat*dilatability*self.strideFactor)
        size = int(num_Feat*dilatability*self.strideFactor)

        self.fc = self.LastBlock(size,nClass)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()  
                        
    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = torch.split(x,3,1)
        out1 = self.LowFreqCell(x[0])
        out2 = self.MidFreqCell(x[1])
        out3 = self.HighFreqCell(x[2])
        out = torch.cat([out1,out2,out3], dim=1)
        # out = self.TemporalCell(x)
        out = self.scb(out)
        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[3] / self.strideFactor)])
        out = self.temporalLayer(out)
        f = torch.flatten(out, start_dim=1)
        c = self.fc(f)
        return c,f

#%% __main__
if __name__ == "__main__":

    # net = FBMSNet_Inception(nChan=22, nTime=512).cuda()
    # summary(net, (9, 22, 512))

    # net = FBMSNet(nChan=22, nTime=512).cuda()
    # summary(net, (9, 22, 512))
    opt_choice = random_choice(2)
    net = FBNASNet(nChan=22, nTime=512,opt_choice=opt_choice).cuda()
    summary(net, (9, 22, 512))