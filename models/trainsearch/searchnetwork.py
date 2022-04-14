###결국 여기서 트레인 다함
"""
train_darts_model 클래스
SearchNetwork 클래스
Found_SearchNet 클래스 """

import models.centmodal.centermodel as center  #

import models.auxmodal.scheduler as sc
import models.auxmodal.aux as aux        #ReShapeInputLayer 함수


import models.trainsearch.train_score.train_sc as tr         # tr.train_mmimdb_track_mse

from models.trainsearch.plotgeno import Plotter

from .darts.fusionnet import FusionNetwork
from .darts.model import Found_FusionNetwork

from .darts.architect import Architect

import torch
import torch.nn as nn
import torch.optim as op


def train_darts_model(dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.MSELoss()
    # model to train
    model = SearchNetwork(args, criterion)  ########################
    params = model.central_params()

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)

    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer) ######

    plotter = Plotter(args)

    #결국에 여기서 다 트레인
    least_mse, best_genotype = tr.train_track_mse(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device,
                                            num_epochs=args.epochs,
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args,
                                            #mse_type=
                                            init_mse=1, th_mse2=0.0001) #기준이 되는

    return least_mse, best_genotype

class SearchNetwork(nn.Module): ###
    def __init__(self, args, criterion):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.image_net = center.GP_VGG(args)  ###이미지 모달넷정의
        self.digit_net = center.SimpleRNN(args, num_hidden=100, number_input_feats=1) ##digit 모달넷정의number_input_feats=100)


        self.reshape_layers = self.create_reshape_layers(args) ########?????????????????
        print(self.reshape_layers)
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        self._criterion = criterion

        #####

        self.fusion_net = FusionNetwork( steps=self.steps, multiplier=self.multiplier, num_input_nodes=self.num_input_nodes,
                                    num_keep_edges=self.num_keep_edges,  args=self.args,  criterion=self.criterion)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,     args.num_outputs) ##????? C * L * multiplier


    def create_reshape_layers(self, args): ######aux의 Reshape Layer 함수 불러와서 append.... layer 수
        C_ins =  [480, 480, 480, 480, 60, 60, 120,120,120,120,120]###??
       # C_ins = [480, 480, 480,  60, 60, 120,120,120,120]
        #C_ins = [512, 512, 512, 512, 64, 64,128,128,128,128,128] #######ValueError: in_channels must be divisible by groups(총 인풋 수가 6이니 6개인듯) # [512, 512, 512, 512, 64, 128] #채널??
        reshape_layers = nn.ModuleList() ###########
      #  print(reshape_layers)
        for i in range(len(C_ins)): #range(6)
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))

        print("reshape_layers",reshape_layers)
        return reshape_layers

    def reshape_input_features(self, input_features): ######
        ret = []
        print("len of self reshape layers",len(self.reshape_layers)) #9 -> 11
        print("length of input features...",len(input_features)) #11-->9??????

        for i, input_feature in enumerate(input_features):

            reshaped_feature = self.reshape_layers[i](input_feature)  # IndexError: index 1~6 is out of range
            ret.append(reshaped_feature)

        return ret

    def forward(self, tensor_tuple):
        digit, image = tensor_tuple

        # apply net on input image
        image_features = self.image_net(image) # 4개??
        print(image_features)
        image_features = image_features[0:-1]

        # apply net on input skeleton
        digit_features = self.digit_net(padded_input = digit, input_lengths =30) ##self.digit_net(digit)#
        print(digit_features)
        digit_features = digit_features[0:-1]

        input_features = list(image_features) + list(digit_features) ########list component 합함
        print("num. input_features:",input_features,"im_feats", list(image_features),"digitfeats: ", list(digit_features))


        input_features = self.reshape_input_features(input_features) ######피처 두개 더한거를 reshape하여라

        out = self.fusion_net(input_features)

        out = self.central_classifier(out) ####
        return out

    def genotype(self):
        return self.fusion_net.genotype()

    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()}, ##key가 같은 딕서너리?
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, target):
        logits = self(input_features)
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self.fusion_net.arch_parameters()

class Found_SearchNet(nn.Module): ###찾아진 모델 스트럭처
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.image_net = center.GP_VGG(args)  ###이미지 모달 가져오기
        self.digit_net = center.SimpleRNN(args, num_hidden=100, number_input_feats=1) ##digit 모달 가져오기 number_input_feats=100>>1

        self._genotype = genotype

        self.reshape_layers = self.create_reshape_layers(args)

        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges

        self.criterion = criterion



        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,  criterion=self.criterion,   genotype=self._genotype) ####

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,   args.num_outputs)

    def create_reshape_layers(self, args):
      #  C_ins =  [480, 480, 480,  60, 60, 120,120,120,120]
        C_ins =  [480, 480, 480, 480, 60, 60, 120,120,120,120,120]###?????????
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes)) #11

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
            else:
                # here the reshape layers is not used, so we set it to ReLU to make it have no parameters
                reshape_layers.append(nn.ReLU())

        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):
        digit, image = tensor_tuple

        # apply net on input image
        image_features = self.imagenet(image) #list
        image_features = image_features[0:-1] #list

        # apply net on input skeleton
        digit_features = self.digit_net(digit)
        digit_features = digit_features #[0:-1]

        input_features = list(image_features) + list(digit_features) #######
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, target):
        logits = self(input_features)
        return self._criterion(logits, target)
