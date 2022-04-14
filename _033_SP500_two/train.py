#%%
#
#
import os
root_dir = os.getcwd()



from datetime import datetime as dt
from argparse import Namespace
from pathlib import Path
from torch.nn import DataParallel
import argparse
import glob
import logging
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
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_finance
import numpy as np
import os
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

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
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

start =  dt.now()
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



output_save_folder_path1 = root_dir + 'outputs_ind/'
output_path1 = os.path.join(output_save_folder_path1, time.strftime('%Y%m%d_%H_%M', time.localtime(time.time())))

args.exppath=output_save_folder_path1
chart_dir ='/nas3/mink/NAS/feature_fusion/charts/' +index+ '_' +args.draw_fnc+'/'  +args.coltype # /미포함

if not os.path.exists(output_save_folder_path1):
    os.mkdir(output_save_folder_path1)
if not os.path.exists(output_path1):
    os.mkdir(output_path1)


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
###fh = logging.FileHandler(os.path.join(output_path, 'log.txt'))
#fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)

setup_logger('log3','log.txt',output_path1)
setup_logger('log4','log_final.txt',root_dir )
log3 = logging.getLogger('log3')
log4 = logging.getLogger('log4')

aaa=output_path1.split('/')[-1]

log3.info(f'{aaa}')
log4.info(f'{aaa}' )




retrieve_path = root_dir +'/outputs/' + args.outputdate

ckpt = torch.load(retrieve_path + '/search_ckpt.pt') # 기존에 Search 한 모델 가져오기
#ckpt = torch.load(output_path + '/search_ckpt.pt')


#log3.info(f'-----------------------------------------------------------------------------------------------------------')
log3.info(f'{args}')
log4.info(f'{args}')
class StockChartDataset_candle(object):

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
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, affine), ##
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, affine)  ###
}


def criterion(input,target): ##############
    loss = nn.MSELoss()
    loss1 = loss(input,target)#######################
    return loss1

#def criterion2(input,target): ##############확인필요
   # loss = F.l1_loss() ##
   # loss1 = loss(input,target)#######################
  #  return loss1


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
        assert C_out % 2 == 0
        self.relu  =  nn.ReLU(inplace=False)
        self.conv_1  =  nn.Conv2d(C_in, C_out//2, 1, stride = 2, padding=0,bias = False)
        self.conv_2  =  nn.Conv2d(C_in, C_out//2, 1, stride = 2,  padding=0,bias = False)
        self.bn  =  nn.BatchNorm2d(C_out, affine = affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine = True):
        super(DilConv, self).__init__()
        self.ops  =  nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_C, in_C, kernel_size = kernel_size, stride = stride, padding = padding, dilation = 2, groups = in_C, bias = False),
            nn.Conv2d(in_C, out_C, kernel_size = 1, padding=0, bias = False),
            nn.BatchNorm2d(out_C, affine = affine)
        )

    def forward(self, x):
        return self.ops(x)


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
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
            nn.Linear(prev_C, n_classes) ###############################################################
        )

        k = sum(1 for i in range(steps) for j in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops) * 1e-3)
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops) * 1e-3)


    def arch_parameters(self):
        return [self.alpha_normal, self.alpha_reduce]


    def new(self):
        new_model = Network(self.C, self.n_classes, self.n_layers, self.criterion, self.device).to(self.device)
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



def concat(xs):
    return torch.cat([x.view(-1) for x in xs])##################################



class Architecture(object):

    def __init__(self, model, lr = 3e-4, momentum = 0.9, weight_decay = 1e-3):
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = weight_decay)


    def compute_unrolled_model(self, inputs, targets, eta, optimizer):
        loss = self.model.loss(inputs, targets) #long()붙임
        theta = concat(self.model.parameters()).detach() # theta: torch.Size([1930618])

        try:
            # fetch momentum data from theta optimizer
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)

        except:
            moment = torch.zeros_like(theta)

        dtheta = concat(torch.autograd.grad(loss, self.model.parameters())).data
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
        loss = self.model.loss(valid_inputs, valid_targets) #long()추가
        loss.backward()

    def backward_step_unrolled(self, train_inputs, train_targets, valid_inputs, valid_targets, eta, optimizer):
        unrolled_model = self.compute_unrolled_model(train_inputs, train_targets, eta, optimizer)
        unrolled_loss = unrolled_model.loss(valid_inputs, valid_targets) #long()추가
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6

        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, train_inputs, train_targets)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0

        for k, v in self.model.named_parameters():
            v_length = v.numel()

            # restore theta[] value to original shape

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
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())


        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(2 * R, v)
        loss = self.model.loss(inputs, targets) #long추가
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(R, v)

        h = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # h len: 2 h0 torch.Size([14, 8])
        # print('h len:', len(h), 'h0', h[0].shape)

        return h




class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev,C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._indices = indices
       # self._compile(C, op_names, indices, concat, reduction)###

    #def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]


    def forward(self, s0, s1, drop_prob): ###
        s0 = self.preprocess0(s0) ##dimension 문제
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(self._steps):

            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob)

            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim = 1)




def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class AuxiliaryHead(nn.Module):

    def __init__(self, aux_C, n_classes):
        super(AuxiliaryHead, self).__init__()
        self.layers = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(aux_C, 128, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, n_classes)
        )

    def forward(self, x):
        return self.layers(x)



class Network(nn.Module): ###modol.py의  NetworkCIFAR 가져옴
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(Network, self).__init__()
        self.layers = layers
        self.auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C

        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )


        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C


        self.cells = nn.ModuleList()

        reduction_prev = False####

        for i in range(layers):
            if i in [layers//3, 2*layers //3]:
                C_curr *=  2
                reduction = True

            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction

            self.cells += [cell]

            #cell_multiplier = 3 ###

            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr ###############################

            if i ==  2 * layers // 3:
                C_to_aux = C_prev

       # if auxiliary:
          #  self.aux_head = AuxiliaryHead(C_to_aux, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1) ##
        self.classifier = nn.Linear(C_prev, num_classes) ##

    def forward(self, x):

        aux_pred = None

        s0 =s1=self.stem(x)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, args.drop_path_prob)

           # if i ==  2 * self.layers // 3:

             #   if self.auxiliary and self.training:

                 #   aux_pred = self.aux_head(s1)

        out = self.global_pooling(s1)
        pred = self.classifier(out.view(out.size(0), -1))


        return pred #, aux_pred   ########################################


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



# def accuracy(logits, targets):
#     _, pred = logits.max(1)
#     acc = pred.eq(targets).float().mean().item()
#     return acc



for idx, (image, target) in enumerate(train_loader):

    image = Variable(image).cuda()#추가3.13
    target = Variable(target).cuda(non_blocking=True)#추가3.13

    target = target.view(-1)   #
    #target = np.argmax(target,axis=1)
    #print(idx, target.shape ) #정수 출력


# ## Device

# In[37]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'



output_path1 #??????




class Trainer_1(object):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.least_loss = 10000
        self.best_epoch=0


    def train(self, train_loader, criterion, optimizer):
        losses = utils.AverageMeter()#progress = ProgressMeter(["train_loss", "train_acc"], len(train_loader), prefix = f'EPOCH {epoch:03d}')
        self.model.train()
      #  self.model.drop_path_prob = args.drop_path_prob * epoch / args.epoch_size  #####

        lr = self.scheduler.get_last_lr()[0] ######

        start_time = time.time()
        loss_value = []

        for idx, (image, target) in enumerate(train_loader):

            image = Variable(image).cuda()#추가3.13
            target = Variable(target).cuda(non_blocking=True)#추가3.13

            target = target.view(-1)    # 1가 있는차원 삭제

            pred = self.model(image)

            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip) ####

            self.optimizer.step()

            loss = loss.item()

            loss_value.append(loss) #progress.update([loss, acc], n = image.size(0))
            mse_r= (sum(loss_value)/len(loss_value)) #
            if idx % 100 == 0:
              #  print("data_num:" ,idx,"/train mse loss:" ,loss)


                log3.info(f'Step:{idx:03} loss:{mse_r }')

        self.scheduler: self.scheduler.step()
        finish_time = time.time()
        epoch_time = finish_time - start_time




        return (sum(loss_value)/len(loss_value))

    def validate(self, val_loader, criterion):
       # progress = ProgressMeter(["val_loss", "val_acc"], len(val_loader), prefix = f'VALID {epoch:03d}')
        self.model.eval()
        loss_value = []
        least_loss = 10000

        with torch.no_grad():
            for idx, (image, target) in enumerate(val_loader):

                image = Variable(image).cuda()#추가3.13
                target = Variable(target).cuda(non_blocking=True)#추가3.13

                target = target.view(-1)    # 1가 있는차원 삭제
                pred, *_ = self.model(image)
                loss = self.criterion(pred, target)
                loss = loss.item()
                loss_value.append(loss) #progress.update([loss, acc], n = image.size(0))
                mse_rr=(sum(loss_value)/len(loss_value))
                if idx % 100 == 0:
                   # print("data_num:" ,idx,"/validation mse loss:" ,loss)
                    log3.info(f'>> Validation: {idx:03} {mse_rr}')
            if loss < self.least_loss: #######################################

             #   self.best_epoch = epoch
                self.least_loss = loss
                ckpt = {
                    #'genotype': self.model.genotype(),
                    #   'best_epoch': self.best_epoch,
                    'least_loss': self.least_loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }

                torch.save(ckpt, output_path1 + '/ckpt.pt')


        return(sum(loss_value)/len(loss_value))

    def test(self, test_loader, criterion):
        ckpt = self.model.load_state_dict(torch.load(output_path1 + '/ckpt.pt'), strict= False) # 변경

        self.model.eval()
        loss_value = []
      #  mae_value = []

        with torch.no_grad():

            for idx, (image, target) in enumerate(test_loader):
                image = Variable(image).cuda()#추가3.13
                target = Variable(target).cuda(non_blocking=True)#추가3.13

                target = target.view(-1)

                pred, * _ = self.model(image)  #3.20변경
                loss = self.criterion(pred, target) #확인 필요
                loss = loss.item()
                loss_value.append(loss)

                #mae = criterion2(pred,target) #확인필요

               # mae= mae.item()
                #mae_value.append(mae)

               # mape = (target - pred).abs() / target.abs()
               # mape = mape.item()
               # mape_value.append(mape)
                mse= (sum(loss_value)/len(loss_value))
              #  mae= (sum(mae_value)/len(mae_value))
                if idx % 1 == 0:

                    log3.info(f'>> Test mse : {idx:03} {mse}')

        # Logging to TensorBoard by default
       #         self.log('mse', loss)#
             #   self.log('mae', mae)#
              #  self.log('mape', mape)#

              #  if idx == 20:
              #      break
        return (sum(loss_value)/len(loss_value)) #, (sum(mae_value)/len(mae_value)) #, (sum(mape_value)/len(mape_value))

genotype = ckpt['genotype']  ###########
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#criterion = F.mse_loss().to(device)

model = Network(args.init_C, 1, args.n_layers, args.auxiliary, genotype).to(device)

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_size, eta_min = 1e-3)

trainer_1 = Trainer_1(model, criterion, optimizer, scheduler, device)

train_loss_value = []
valid_loss_value = []

for ep in range(args.epoch_size):

    print('TRAIN:', ep+1, '/',args.epoch_size,'-' * 65)###
    train_loss_value.append(trainer_1.train(train_loader, criterion, optimizer))

    print('VALIDATION:',ep+1, '/',args.epoch_size,'-' * 65)###
    valid_loss_value.append(trainer_1.validate(valid_loader, criterion))

#
test_loss_value=[]
#test_mae_value = []
#test_mape_value = []

for ep in range(1):

    print('TEST:',ep+1, '/',args.epoch_size,'-' * 65)###

    mse1=trainer_1.test(test_loader, criterion)

    log3.info(f'args again = {args}') #추가
    log3.info(f'Test mse')#{idx:03}
    log3.info(f'{mse1}')#{idx:03}
    log4.info(f'Test mse')#{idx:03}
    log4.info(f'{mse1}')#{idx:03}

    end = dt.now()
    elapsed=end-start

    log3.info(f'Took {elapsed.days}d{elapsed.seconds // 3600}h{elapsed.seconds // 60 % 60}m{elapsed.seconds % 60}s')
    log4.info(f'Took {elapsed.days}d{elapsed.seconds // 3600}h{elapsed.seconds // 60 % 60}m{elapsed.seconds % 60}s')
    log4.info(f'----------------------------------------------------Finished!--------------------------------------------------------------')

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
