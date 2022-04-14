#%%

from __future__ import absolute_import
import os
from datetime import datetime as dt
start =  dt.now()
root_dir = os.getcwd()

#stock_index ='SP500'

from argparse import Namespace
from pathlib import Path
from torch.nn import DataParallel
import argparse
import glob
import socket
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import logging
from collections import namedtuple
from glob import glob
from graphviz import Digraph
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from torchvision import transforms #, datasets
import argparse
import copy
import datetime
import FinanceDataReader as fdr
import gc
import glob
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_finance
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pytorch_lightning as pl
import re
import seaborn as sns
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils ####### as 추가
import torchvision
import torchvision.datasets as dset
import utils #폴더
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat') # normal_bottleneck reduce_bottleneck')
#Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# Genotype이라는 클래스 만들어줌

DARTS = Genotype(
    normal = [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat = [2, 3, 4, 5], #########
    reduce = [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat = [2, 3, 4, 5])

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
from datetime import timedelta

import argparse

parser = argparse.ArgumentParser("CNN_DARTS")#####

parser.add_argument('-seq', '--seq_length', type=int, default=30)
parser.add_argument('-step', '--step_length', type=int, default=1)
parser.add_argument('--unrolled', type=bool, default=False)
parser.add_argument('-aux', '--auxiliary', type= bool, default=False)
parser.add_argument('-aux_w', '--auxiliary_weight', type=float, default=0.1)
parser.add_argument('--drop_path_prob', type=float, default=0.2)
parser.add_argument('--init_C', type=int, default=16)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.025) ##smaller to avoid eror?.025
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0003)
parser.add_argument('--grad_clip', type=int, default=4) #5
parser.add_argument('-bs', '--batch_size', type=int, default=32) ###################배치사이즈 16
parser.add_argument('-sep', '--search_epoch_size', type=int, default=20)
parser.add_argument('-ep', '--epoch_size', type=int, default=50) ##약칭
#parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')#추가
parser.add_argument('--exppath', type=Path, default=Path('outputs_ind'), help='experiment name') ##parser.add_argument('--exp_path', type=Path, default=root_dir +'outputs/') #default=Path('exp'), help='experiment name')
#parser.add_argument('--save', type=Path, default=root_dir +'outputs_ind/')
parser.add_argument('--checkpoint_path', type=Path, help='path to checkpoint for restart')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use') ##추가
parser.add_argument('-date', '--outputdate', type=str, help='로드할 파일있는 폴더명 반드시 입력')
parser.add_argument('-col', '--coltype', type=str, help='로드할 차트 데이터 타입 반드시 입력')
parser.add_argument('-drawf', '--draw_fnc', type=str, help='반드시 입력')
parser.add_argument( '--index', type=str, help='반드시 입력')
parser.add_argument( '--dl', type=bool, default=False)


args = parser.parse_args()

index =args.index

output_save_folder_path = root_dir + '/outputs/'
output_path = os.path.join(output_save_folder_path, time.strftime('%Y%m%d_%H_%M', time.localtime(time.time())))

chart_dir ='/nas3/mink/NAS/feature_fusion/charts/' +index+ '_' +args.draw_fnc+'/'  +args.coltype # /미포함

if not os.path.exists(output_save_folder_path):
    os.mkdir(output_save_folder_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

import logging

def setup_logger(logger_name, log_file, path, level=logging.INFO):
    l = logging.getLogger(logger_name)
    logging.basicConfig(stream=sys.stdout, datefmt='%Y%m%d_%H_%M')

    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(os.path.join(path, log_file)) #, mode='w'
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

setup_logger('log1','log.txt',output_path)
setup_logger('log2','log_search.txt',root_dir )
log1 = logging.getLogger('log1')
log2 = logging.getLogger('log2')

aaaa=output_path.split('/')[-1]

log1.info(f'{aaaa}')
log2.info(f'{aaaa}' )

log1.info(f'{args}')
log2.info(f'{args}')

class StockChartDataset_candle(object): #####

    def __init__(self, dir_path, transforms):
        self.dir_path = dir_path
        self.transforms = transforms

        df = pd.read_csv(dir_path + 'candlestick_target.csv')  ##########

        self.imgs = df.filename.tolist()
        self.log_target = df.target.tolist()

    def __getitem__(self, idx):
        # Load images
        img_path = os.path.join(self.dir_path, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        target = torch.tensor([self.log_target[idx]])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = StockChartDataset_candle(chart_dir + '/train/candlestick/',
                                       transforms = transform)
valid_dataset = StockChartDataset_candle(chart_dir + '/val/candlestick/',
                                     transforms = transform)
test_dataset = StockChartDataset_candle(chart_dir + '/test/candlestick/',
                                      transforms = transform)
####################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0,drop_last=args.dl) ######배치사이즈 1->64->1 2.12
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch_size,  shuffle = True,  num_workers = 0,drop_last=args.dl)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size  = args.batch_size, shuffle = False,  num_workers = 0,drop_last=args.dl)

x, y = next(iter(train_loader)) #빼면안되
#x.size(), y.size()
x, y = next(iter(valid_loader))
#x.size(), y.size()
x, y = next(iter(test_loader))

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# Genotype이라는 클래스 만들어줌

DARTS = Genotype(
    normal = [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat = [2, 3, 4, 5], #########
    reduce = [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat = [2, 3, 4, 5])


# ## Operation full 버전 setting


PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(kernel_size=3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'sep_conv_7_7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, affine)
}


def criterion(input,target): #####################################
    return F.mse_loss(input,target)

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride  =  stride

    def forward(self, x):
        if self.stride == 2:
            x = x[:,:,::self.stride, ::self.stride]
        return x.mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine = True):
        super(FactorizedReduce, self).__init__()

        self.relu  =  nn.ReLU()
        self.conv1  =  nn.Conv2d(C_in, C_out//2, kernel_size = 1, stride = 2, bias = False)
        self.conv2  =  nn.Conv2d(C_in, C_out//2, kernel_size = 1, stride = 2, bias = False)
        self.bn  =  nn.BatchNorm2d(C_out, affine = affine)

    def forward(self, x):
        x  =  self.relu(x)
        x1  =  self.conv1(x)
        x2  =  self.conv2(x[:,:,1:,1:])
        x  =  torch.cat([x1, x2], dim = 1)
        x  =  self.bn(x)
        return x


class SepConv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine = True):
        super(SepConv, self).__init__()
        self.ops  =  nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_C, in_C, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_C, bias = False),
            nn.Conv2d(in_C, in_C, kernel_size = 1, bias = False),
            nn.BatchNorm2d(in_C, affine = affine),
            nn.Conv2d(in_C, in_C, kernel_size = kernel_size, stride = 1, padding = padding, groups = in_C, bias = False),
            nn.Conv2d(in_C, out_C, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_C, affine = affine)
        )

    def forward(self, x):
        return self.ops(x)


class DilConv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine = True):
        super(DilConv, self).__init__()
        self.ops  =  nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_C, in_C, kernel_size = kernel_size, stride = stride, padding = padding, dilation = 2, groups = in_C, bias = False),
            nn.Conv2d(in_C, out_C, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_C, affine = affine)
        )

    def forward(self, x):
        return self.ops(x)


class ReLUConvBN(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine = True):
        super(ReLUConvBN, self).__init__()
        self.ops  =  nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_C, out_C, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(out_C, affine = affine)
        )

    def forward(self, x):
        return self.ops(x)



class SearchNetwork(nn.Module):
    def __init__(self, C, n_classes, n_layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3):
        super(SearchNetwork, self).__init__()
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.criterion = criterion
        self.device = device
        self.steps = steps
        self.multiplier = multiplier

        curr_C = C * stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, curr_C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(curr_C)
        )

        prev_prev_C, prev_C, curr_C = curr_C, curr_C, C
        self.cells = nn.ModuleList()
        prev_reduction = False
        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers//3]:
                curr_C *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(steps, multiplier, prev_prev_C, prev_C, curr_C, reduction, prev_reduction)
            prev_reduction = reduction
            self.cells.append(cell)
            prev_prev_C, prev_C = prev_C, multiplier * curr_C

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prev_C, n_classes)
        )

        k = sum(1 for i in range(steps) for j in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops) * 1e-3)
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops) * 1e-3)


    def arch_parameters(self):
        return [self.alpha_normal, self.alpha_reduce]


    def new(self):
        new_model = Network(self.C, self.n_classes, self.n_layers, self.criterion, self.device).to(self.device)

        if use_DataParallel:

            for x, y in zip(new_model.module.arch_parameters(), self.module.arch_parameters()):
                x.data.copy_(y.data)

        else:
            for x, y in zip(new_model.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)

        return new_model


    def loss(self, inputs, targets):  ######################
        logits = self(inputs) #?????????????##############################

        return F.mse_loss(logits, targets) #long추가취소


    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

        logits = self.classifier(s1)
        return logits


    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0

            for i in range(self.steps): # for each node
                end = start + n
                W = weights[start:end].copy() # [2, 8], [3, 8], ...
                edges = sorted(range(i + 2), # i+2 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) # get strongest ops
                                               if k != PRIMITIVES.index('none'))
                               )[:2] # only has two inputs
                for j in edges: # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])): # get strongest ops for current input j->i
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j)) # save ops and input node
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype

class MixedOp(nn.Module): ###가중 평균 해주는 클래스
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(
                    op,
                    nn.BatchNorm2d(C, affine = False)
                )
            self.ops.append(op)

    def forward(self, x, weights):
        out = sum([w * l(x) for w, l in zip(weights, self.ops)])
        return out

class SearchCell(nn.Module):
    def __init__(self, steps, multiplier, prev_prev_C, prev_C, curr_C, reduction, prev_reduction):
        super(SearchCell, self).__init__()
        self.steps = steps
        self.multiplier = multiplier
        self.reduction = reduction

        if prev_reduction:   ###########################
            self.preprocess0 = FactorizedReduce(prev_prev_C, curr_C, affine = False)

        else:
            self.preprocess0 = ReLUConvBN(prev_prev_C, curr_C, kernel_size = 1, stride = 1,
                                    padding = 0, affine = False)


        self.preprocess1 = ReLUConvBN(prev_C, curr_C, kernel_size = 1, stride = 1, padding = 0, affine = False)

        self.layers = nn.ModuleList()

        for i in range(steps):

            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1    ###############

                op = MixedOp(curr_C, stride)  #########
                self.layers.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self.steps):
            s = sum([self.layers[offset + j](h, weights[offset + j])for j, h in enumerate(states)])
            offset +=  len(states)
            states.append(s)

        return torch.cat(states[-self.multiplier:], dim = 1)

# ## Architecture

def concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architecture:

    def __init__(self, model,criterion, lr = args.lr, momentum = args.momentum, weight_decay = 1e-3):
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.criterion =criterion
        self.optimizer = torch.optim.Adam(self.model.module.arch_parameters() if use_DataParallel else self.model.arch_parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = weight_decay)


    def compute_unrolled_model(self, inputs, targets, eta, optimizer):
        loss = self.model.loss(inputs, targets) #long()붙임
        theta = concat(self.model.parameters()).detach() # theta: torch.Size([1930618])

        try:
            # fetch momentum data from theta optimizer
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)

        except:
            moment = torch.zeros_like(theta)

        dtheta = concat(torch.autograd.grad(loss, self.model.parameters() if use_DataParallel else self.model.module.parameters())).data
        theta = theta.sub(eta, moment + dtheta + self.weight_decay * theta)
        unrolled_model = self.construct_model_from_theta(theta) # construct a new model

        return unrolled_model

    def step(self, train_inputs, train_targets, valid_inputs, valid_targets, eta, optimizer, unrolled):
        self.optimizer.zero_grad()

        if unrolled:
            self.backward_step_unrolled(train_inputs, train_targets, valid_inputs, valid_targets, eta, optimizer)
        else:
            self.backward_step(valid_inputs, valid_targets)
        self.optimizer.step()

    def backward_step(self, valid_inputs, valid_targets):
        #loss = self.model.loss(valid_inputs, valid_targets)

        logits = self.model(valid_inputs)
        loss = self.criterion(logits, valid_targets)

        # both alpha and theta require grad but only alpha optimizer will
        # step in current phase.

        loss.backward()

    def backward_step_unrolled(self, train_inputs, train_targets, valid_inputs, valid_targets, eta, optimizer):
        unrolled_model = self.compute_unrolled_model(train_inputs, train_targets, eta, optimizer)
        unrolled_loss = unrolled_model.loss(valid_inputs, valid_targets) #long()추가
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6

        dalpha = [v.grad for v in unrolled_model.module.parameters()] if use_DataParallel else [v.grad for v in unrolled_model.parameters()]

        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, train_inputs, train_targets)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

  # write updated alpha into original model
        if use_DataParallel:
            for v, g in zip(self.model.module.parameters(), dalpha):
                if v.grad is None:
                    v.grad = g.data
                else:
                    v.grad.data.copy_(g.data)
        else:
            for v, g in zip(self.model.parameters(), dalpha):
                if v.grad is None:
                    v.grad = g.data
                else:
                    v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        model_new = self.model.module.new() if use_DataParallel else self.model.new()
        model_dict = self.model.module.state_dict() if use_DataParallel else self.model.state_dict()

        params, offset = {}, 0

        for k, v in self.model.named_parameters():
            v_length = v.numel()

            # restore theta[] value to original shape
            name = k[7:] if use_DataParallel else k #?????????????????
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset +=  v_length

        assert offset ==  len(theta)

        model_dict.update(params)
        model_new.load_state_dict(model_dict)

        return model_new.cuda()

    def hessian_vector_product(self, vector, inputs, targets, r = 1e-2):
        R = r / concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            # w+ = w + R * v
            p.data.add_(R, v)
        loss = self.model.loss(inputs, targets) #long 추가
        # gradient with respect to alpha
        grads_p = autograd.grad(loss, self.model.module.parameters() if use_DataParallel else self.model.parameters())


        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(2 * R, v)
        loss = self.model.loss(inputs, targets) #long추가
        grads_n = torch.autograd.grad(loss, self.model.module.parameters() if use_DataParallel else self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(R, v)

        h = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # h len: 2 h0 torch.Size([14, 8])
        # print('h len:', len(h), 'h0', h[0].shape)

        return h


class Cell(nn.Module):
    def __init__(self, genotype, prev_prev_C, prev_C, C, reduction, prev_reduction):
        super(Cell, self).__init__()

        if prev_reduction:
            self.preprocess0 = FactorizedReduce(prev_prev_C, C)

        else:
            self.preprocess0 = ReLUConvBN(prev_prev_C, C, kernel_size = 1, stride = 1, padding = 0)
        self.preprocess1 = ReLUConvBN(prev_C, C, kernel_size = 1, stride = 1, padding = 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat

        else:
#             print('koos, type(genotype) :::', type(genotype))
#             print('koos, genotype :::', genotype)
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self.steps = len(op_names) // 2
        self.indices = indices
        self.concat = concat
        self.multiplier = len(concat)
        self.ops = nn.ModuleList()

        for name, idx in zip(op_names, indices):
            stride = 2 if reduction and idx < 2 else 1
            op = OPS[name](C, stride, True)
            self.ops.append(op)


    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self.steps):

            h1 = states[self.indices[2 * i]]
            h2 = states[self.indices[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob)

            s = h1 + h2
            states.append(s)

        return torch.cat([states[i] for i in self.concat], dim = 1)

def drop_path(x, drop_prob):

    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = x.bernoulli(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)

    return x

#class AuxiliaryHead(nn.Module):
  #  def __init__(self, aux_C, n_classes):
    #    super(AuxiliaryHead, self).__init__()
      #  self.layers = nn.Sequential(

         #   nn.ReLU(),
           # nn.AdaptiveAvgPool2d(2),
           # nn.Conv2d(aux_C, 128, kernel_size = 1, stride = 1, padding = 0, bias = False),
           # nn.BatchNorm2d(128),
          #  nn.ReLU(),
        #    nn.Conv2d(128, 768, kernel_size = 2, stride = 1, padding = 0, bias = False),
          #  nn.BatchNorm2d(768),
          #  nn.ReLU(),
         #   nn.Flatten(),
        #    nn.Linear(768, n_classes)
       # )

  #  def forward(self, x):
      #  return self.layers(x)

class Network(nn.Module):
    def __init__(self, C, n_classes, n_layers, auxiliary, genotype, stem_multiplier = 3):##aux
        super(Network, self).__init__()
        self.n_layers = n_layers
        self.auxiliary = auxiliary

        curr_C = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, curr_C, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(curr_C)
        )

        prev_prev_C, prev_C, curr_C = curr_C, curr_C, C
        self.cells = nn.ModuleList()
        prev_reduction = False

        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers //3]:
                curr_C *=  2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, prev_prev_C, prev_C, curr_C, reduction, prev_reduction)
            prev_reduction = reduction
            self.cells.append(cell)
            prev_prev_C, prev_C = prev_C, cell.multiplier * curr_C

           # if i ==  2 * n_layers // 3:
               # aux_C = prev_C

        #if auxiliary:
        #    self.aux_head = AuxiliaryHead(aux_C, n_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prev_C, n_classes)
        )

    def forward(self, x):
      #  aux_logits = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, args.drop_path_prob)
           # if i ==  2 * self.n_layers // 3:
            #    pass#   if self.auxiliary and self.training:
                 #   aux_logits = self.aux_head(s1)
        logits = self.classifier(s1)

        return logits #, aux_logits

## Digraph

def plot(genotype, filename):

    g = Digraph(
        format = 'png',
        edge_attr = dict(fontsize = '20', fontname = "times"),
        node_attr = dict(style = 'filled', shape = 'rect', align = 'center', fontsize = '20', height = '0.5', width = '0.5',
                       penwidth = '2', fontname = "times"),
        engine = 'dot')

    g.body.extend(['rankdir = LR'])

    g.node("c_{k-2}", fillcolor = 'darkseagreen2')
    g.node("c_{k-1}", fillcolor = 'darkseagreen2')

    assert len(genotype) % 2 ==  0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor = 'lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j ==  0:
                u = "c_{k-2}"
            elif j ==  1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, target = op, fillcolor = "gray")

    g.node("c_{k}", fillcolor = 'palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor = "gray")

    g.render(filename, view = False)


for idx, (image, target) in enumerate(train_loader):

    image = Variable(image).cuda()#추가3.13
    target = Variable(target).cuda(non_blocking=True)#추가3.13

    target = target.view(-1)   #
    #target = np.argmax(target,axis=1)
   # print(idx, target.shape ) #정수 출력


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f'{self.name:10s} {self.avg:.3f}'
        return fmtstr


class ProgressMeter(object):
    def __init__(self, meters, loader_length, prefix=""):

        self.meters = [AverageMeter(i) for i in meters]
        self.loader_length = loader_length
        self.prefix = prefix  ##

    def reset(self):
        for m in self.meters:
            m.reset()

    def update(self, values, n=1):
        for m, v in zip(self.meters, values):

            m.update(v, n)
            self.__setattr__(m.name, m.avg)

    def display(self, batch_idx, postfix=""):
        batch_info = f'[{batch_idx+1:03d}/{self.loader_length:03d}]'
        msg = [self.prefix + ' ' + batch_info]
        msg += [str(meter) for meter in self.meters]
        msg = ' | '.join(msg)

        sys.stdout.write('\r')
        sys.stdout.write(msg + postfix)
        sys.stdout.flush()


class SearchTrainer(object):
    def __init__(self, model, architecture, criterion, optimizer, scheduler, device): ###################
        self.model = model
        self.architecture = architecture
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        #self.epoch = args.search_epoch_size #####
        self.least_loss = 10000 #받는 변수가 아니지만 클래스 내에서 사용하는변수
        #self.best_epoch = 0

    def train(self, train_loader, valid_loader, criterion, optimizer):##발리드 데이터는 왜????
##################
        losses = utils.AverageMeter()
        lr = self.scheduler.get_last_lr()[0] ######

        #valid_iter = iter(valid_loader) #???????????????

        with tqdm(train_loader) as progress:
            for step, (x, target) in enumerate(progress):
                progress.set_description_str(f'Step {step}') ##epoch변경
                batchsz = x.size(0)
                model.train()
                # [b, 3, 32, 32], [40]
                x, target = x.to(device), target.to(device, non_blocking=True)
                x_search, target_search = next(iter(valid_loader))  # [b, 3, 32, 32], [b] #######??????????????????????????????????????????????????/
                x_search, target_search = x_search.to(device), target_search.to(device, non_blocking=True)
                # 1. update alpha
                self.architecture.step(x, target, x_search, target_search, lr, self.optimizer, unrolled=args.unrolled)

                logits = self.model(x)
                loss = criterion(logits, target)

                # 2. update weight
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.module.parameters() if use_DataParallel else self.model.parameters(),
                                        args.grad_clip)
                self.optimizer.step()

                losses.update(loss.item(), batchsz)

                #if step % 100 == 0:
                   # log1.info(f'Step:{step:03} loss:{losses.avg }')

            return losses.avg

    def validate(self, val_loader, criterion):
        self.model.eval()
        #loss_value = []
        least_loss = 10000
        self.epoch = args.search_epoch_size

        losses = utils.AverageMeter()
        self.model.eval()

        with torch.no_grad():
            with tqdm(valid_loader) as progress:
                for step, (x, target) in enumerate(progress):

                    progress.set_description_str(f'Valid epoch {epoch}')
                    x, target = x.to(device), target.cuda(non_blocking=True)
                    batchsz = x.size(0)
                    logits = self.model(x)
                    loss = criterion(logits, target)

                    losses.update(loss.item(), batchsz)
                    progress.set_postfix_str(f'loss: {losses.avg}')

                   # if step % 100 == 0:
                    #    log1.info(f'Validation: {step:03} {losses.avg}')

                if loss < self.least_loss: #######################################

             #   self.best_epoch = epoch
                    self.least_loss = loss
                    genotype = self.model.module.genotype() if use_DataParallel else self.model.genotype()

                    ckpt = {
                        'genotype': self.model.module.genotype() if use_DataParallel else self.model.genotype(),
                     #   'best_epoch': self.best_epoch,
                        'least_loss': self.least_loss,
                        'model_state_dict': self.model.module.state_dict() if use_DataParallel else self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}

                    torch.save(ckpt, output_path + '/search_ckpt.pt')

            #PLOT
                    plot(genotype.normal, output_path + f'/{epoch+1:02d}_normal')
                    plot(genotype.reduce, output_path + f'/{epoch+1:02d}_reduce')


        return losses.avg

    def test(self, test_loader, criterion):

        ckpt = self.model.load_state_dict(torch.load(output_path + '/search_ckpt.pt'), strict= False) # 변경
        self.model.eval()
        loss_value = []

        with torch.no_grad():

            for idx, (image, target) in enumerate(test_loader):
                image = Variable(image).cuda()#추가3.13
                target = Variable(target).cuda(non_blocking=True)#추가3.13

                target = target.view(-1)

                pred = self.model(image)
                loss = self.criterion(pred, target)
                loss = loss.item()
                loss_value.append(loss)


                if idx % 1 == 0:
                    print("num:" ,idx,"/test mse loss:" ,loss)


        return(sum(loss_value)/len(test_loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SearchNetwork(args.init_C, 1, args.n_layers, criterion, device).to(device)############################

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay) #
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.search_epoch_size, eta_min = 1e-3)
#아키텍처 인풋
use_cuda = torch.cuda.is_available()#추가
device = torch.device('cuda' if use_cuda else 'cpu') #추가
use_DataParallel = torch.cuda.device_count() > 1 #추가

if use_DataParallel:
    print('use Data Parallel')
    model = nn.parallel.DataParallel(model)
    model = model.cuda()
    module = model.module
    #torch.cuda.manual_seed_all(args.seed)
else:
    model = model.to(device)
    module = model

architecture = Architecture(model, criterion)

trainer = SearchTrainer(model, architecture, criterion, optimizer, scheduler, device) #class로 인스턴스 만듬?###

torch.cuda.empty_cache()

train_loss_value = []
valid_loss_value = []

for epoch in tqdm(range(args.search_epoch_size), desc='Total Progress'):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        log1.info(f'\n >> Epoch: {epoch+1} lr: {lr}')
        gen = module.genotype()
        log1.info(f'Genotype: {gen}')

        train_obj = trainer.train(train_loader, valid_loader,criterion, optimizer) ## Search Trainer 클래스의 train
        train_loss_value.append(train_obj)#함수

        log1.info(f'Train mse: {train_obj}')
        valid_obj =trainer.validate(valid_loader, criterion)
        valid_loss_value.append(valid_obj)

        log1.info(f'Valid mse: {valid_obj}')

        gen = module.genotype()
        gen_path = output_path + '/genotype.json'
        utils.save_genotype(gen, gen_path)

        log1.info(f'Result genotype: {gen}')


#for ep in range(args.search_epoch_size):
    #print('TRAIN:', ep+1, '/',args.search_epoch_size,'-' * 65)###

    #train_loss = trainer.train(train_loader, valid_loader,  model, criterion, optimizer, lr=args.lr, epoch=args.search_epoch_size+1) ## Search Trainer 클래스의 train 함수
    #log1.info(f'train mse: {train_loss}')

  #  print('VALIDATION:',ep+1, '/',args.search_epoch_size,'-' * 65)###

    #valid_loss=trainer.validate(valid_loader, model,criterion,epoch+1)

   # log1.info(f'valid mse: {valid_loss}')

    #log1.info(f'Result genotype: {gen}')

#test_loss_value=[]

for ep in range(1): #

    print('TEST:',ep+1, '/',args.search_epoch_size,'-' * 65)###

    mse1=trainer.test(test_loader, criterion)

    log1.info(f'args again = {args}') #추가
    log1.info(f'Test mse')#{idx:03}
    log1.info(f'{mse1}')#{idx:03}
    log2.info(f'Test mse')#{idx:03}
    log2.info(f'{mse1}')#{idx:03}

    end = dt.now()
    elapsed=end-start

    log1.info(f'Took {elapsed.days}d{elapsed.seconds // 3600}h{elapsed.seconds // 60 % 60}m{elapsed.seconds % 60}s')
    log2.info(f'Took {elapsed.days}d{elapsed.seconds // 3600}h{elapsed.seconds // 60 % 60}m{elapsed.seconds % 60}s')
import matplotlib.pyplot as plt

#%%

import matplotlib.pyplot as plt #
fig = plt.figure()

plt.plot(train_loss_value, marker='.',c='red', linewidth = 1, target ="Train Set Loss")
plt.plot(valid_loss_value, marker='.',c='blue', linewidth = 1, target ="Val Set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xtarget('epoch')
plt.show()
# %%


#fig = plt.figure()
#plt.plot(test_loss_value, marker='.', c='green', linewidth = 1, target ="Test Set Loss")

#plt.legend(loc='upper right')
#plt.grid()
#plt.xtarget('Epoch')
#plt.ytarget('Loss')
#plt.show()

#print('Normal Cell')  ##################
#display(Image.open(output_path + f'/{ep+1:02d}_normal.png'))
#print('Reduction Cell')
#display(Image.open(output_path + f'/{ep+1:02d}_reduce.png'))
