import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import models.auxmodal.aux as aux
import numpy as np
from torchvision import models as tmodels


class SimpleRNN(nn.Module):

    def __init__(self, args, num_hidden=100, number_input_feats=1):#100>1
        super(SimpleRNN, self).__init__()

        self.num_hidden = num_hidden

        # e1
        # self.embedding1 = nn.GRU(number_input_feats, num_hidden, batch_first=True, num_layers=2, dropout=0.6)
        self.embedding1 = nn.GRU(number_input_feats, num_hidden, batch_first=True, num_layers=1)

        # e2
        #self.embedding2 = nn.GRU(num_hidden, num_hidden, batch_first=True, num_layers=1)

        # The linear layer that maps from hidden state space to output space
        self.hid2val = nn.Linear(num_hidden, args.num_outputs)

    def forward(self, padded_input, input_lengths=30):
        print("length of input",len(padded_input))
        padded_output1, _ = self.embedding1(padded_input)
        padded_output1 = nn.functional.dropout(padded_output1, 0.666)

        #padded_output2, _ = self.embedding2(padded_output1)

     #   vals = []
       # for i, s in enumerate(padded_output2):
       #     vals.append(s[input_lengths[i] - 1]) ###############################
      #  vals = torch.stack(vals, dim=0)

        val_space = self.hid2val(padded_output1)
        print("rnn out shape",val_space.shape)
        # val_space = F.sigmoid(self.hid2val(padded_output1[:,-1,:]))

        return val_space ##마지막 아웃풋 하나만?????????????왜 여러개나와


class GP_VGG(nn.Module):
    def __init__(self, args):
        super(GP_VGG, self).__init__()

        # self.vgg = tmodels.vgg19(pretrained='imagenet')
        vgg = list(tmodels.vgg19(pretrained='imagenet').features)  #featured의 리스트



        self.vgg = nn.ModuleList(vgg)

        self.gp1 = aux.GlobalPooling2D()
        self.gp2 = aux.GlobalPooling2D()
        self.gp3 = aux.GlobalPooling2D()
        self.gp4 = aux.GlobalPooling2D()

       # self.bn4 = nn.BatchNorm1d(116)  # only used for classifier
        self.classifier0 = nn.Linear(512,116)
        self.classifier = nn.Linear(116, args.num_outputs) # num_outputs = 1->32 (regression) 최종아웃풋아님 rnn에서 아웃풋                  29*4=116
        self.conv2d=nn.Conv2d(512,116,kernel_size=1)
    def forward(self, x):
        print("cnn in shape", x.shape)
        for i_l, layer in enumerate(self.vgg): #nn.ModuleList(vgg)
            print(self.vgg)
            x = layer(x)

            if i_l == 20:  #20번째모듈?레이어
                # out_1 = self.gp1(x)
                out_1 =self.conv2d(x)#####################변경
               # self.conv2d(x)

            if i_l == 26:
                # out_2 = self.gp2(x)
                out_2 =self.conv2d(x)###########################변경

            if i_l == 33:
                # out_3 = self.gp3(x)
                out_3  =self.conv2d(x)###########################변경

            if i_l == 36:
                out_4 = self.conv2d(x)
                print(x.shape)#([8, 512, 3, 3])
                tmp_4 = self.gp4(x)# mat1 and mat2 shapes cannot be multiplied (8x512 and 116x32)
             #   bn_4 = self.bn4(tmp_4)#running_mean should contain 512 elements not 116

        out =  self.classifier0( tmp_4)
        print(out.shape)#([8, 512, 3, 3])
        out = self.classifier(out)
        print(out.shape)#([8, 512, 3, 3])
      #  out=self.conv2d(out)###########################변경

        # print()
        # print(out_4, out)
        print("cnn out shape", out.shape)

        return out_1, out_2, out_3, out_4,   out  ###returns 5 outputs

#####################################
######################################
####FUSING  모델 -baselines#############

class SomeCentralNet(nn.Module): #  digit_first_hidden,
    def __init__(self, args, digit_first_hidden, image_channels):
        super(SomeCentralNet, self).__init__()

        self.args = args
        self.image_net = GP_VGG(args)
        self.digit_net = SimpleRNN(args, num_hidden=100, number_input_feats=1)

        self.alpha1_feat1 = nn.Parameter(torch.rand(1))
        self.alpha2_feat1 = nn.Parameter(torch.rand(1))

        self.alpha1_feat2 = nn.Parameter(torch.rand(1))
        self.alpha2_feat2 = nn.Parameter(torch.rand(1))

        self.alpha_conv1 = nn.Parameter(torch.rand(1))
        self.alpha_conv2 = nn.Parameter(torch.rand(1))

        # self.central_conv1 = nn.Conv1d(1, args.channels, kernel_size=3, padding=1, bias=False)
        # self.central_conv2 = nn.Conv1d(1, args.channels, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        nodes = args.channels
        inunits = 512

        if self.args.fusetype == 'cat':
            nodes *= 2
            inunits *= 2

        self.central_conv1 = nn.Linear(inunits, args.channels)
        self.central_conv2 = nn.Linear(inunits, args.channels)

        self.gp1 = aux.GlobalPooling1D()
        self.gp2 = aux.GlobalPooling1D()

        self.bnc1 = nn.BatchNorm1d(args.channels)
        self.bnc2 = nn.BatchNorm1d(args.channels)

        self.central_classifier = nn.Linear(nodes, args.num_outputs)

    def central_params(self):

        central_parameters = [
            {'params': self.central_conv1.parameters()},
            {'params': self.central_conv2.parameters()},
            {'params': self.alpha1_feat1},
            {'params': self.alpha2_feat1},
            {'params': self.alpha1_feat2},
            {'params': self.alpha2_feat2},
            {'params': self.alpha_conv1},
            {'params': self.alpha_conv2},
            {'params': self.central_classifier.parameters()}]

        return central_parameters

    def forward(self, digit, image):

        im_gp1, im_gp2, im_gp3, im_gp4, im_out = self.image_net(image)

        im_gp1 = self.bn1(im_gp1)
        im_gp2 = self.bn2(im_gp2)
        im_gp3 = self.bn3(im_gp3)
        im_gp4 = self.bn4(im_gp4)

        t_o1, t_o2, t_out = self.digit_net(digit)

        one = A.Variable(torch.ones(1))  # ugly hack to improve
        if image.is_cuda:
            one = one.cuda()

        if self.args.fusingmix == '11,24':
            fuse1 = self._fuse_features(t_o1, im_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)

        elif self.args.fusingmix == '13,24':
            fuse1 = self._fuse_features(t_o1, im_gp3, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)

        elif self.args.fusingmix == '12,24':
            fuse1 = self._fuse_features(t_o1, im_gp2, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        else:
            raise ValueError(
                'self.args.fusingmix {} fusion combinantion is not implemented'.format(self.args.fusingmix))

        cc1 = F.relu(self.central_conv1(fuse1))
        cc2 = F.relu(self.central_conv2(fuse2))

        # cc1 = F.relu(self.central_conv1(fuse1.unsqueeze(1)))
        # cc2 = F.relu(self.central_conv2(fuse2.unsqueeze(1)))

        # fuse3 = self.gp1(cc)
        # cc2 = self.gp2(cc2)

        fuse3 = self._fuse_features(cc1, cc2, self.alpha_conv1, self.alpha_conv2, self.args.fusetype) #fusetype 컨켓etc

        fusion_out = self.central_classifier(fuse3)

        return t_out, im_out, fusion_out

    def _fuse_features(self, ft1,ft2, a1, a2, fusetype):

        ft1sz = ft1.size()
        ft2sz = ft2.size()

        dif = ft1sz[1] - ft2sz[1]

        if fusetype == 'cat':  ##피처 두개 사이즈를 비교 해서 lateral padding을 양수로..조정  그리고 컨켓 순서를 큰거를 먼저 작은거에  옆쪽 패딩 가로 패딩?
            if dif > 0: #피처 1 길이가 더 크다면
                fuse = torch.cat((ft1, self._lateral_padding(ft2, dif)), 1)
            elif dif < 0:
                fuse = torch.cat((self._lateral_padding(ft1, -dif), ft2), 1)
            else:
                fuse = torch.cat((ft1,ft2), 1)

        elif fusetype == 'wsum': ###weighted sum = 성능 더 안 좋음
            if dif > 0:
                fuse = ft1 * a1.expand_as(ft1) + self._lateral_padding(ft2, dif) * a2.expand_as(ft1)
            elif dif < 0:
                fuse = self._lateral_padding(ft1, -dif) * a1.expand_as(ft2) + ft2 * a2.expand_as(ft2)
            else:
                fuse = ft1 * a1.expand_as(ft1) + ft2 * a1.expand_as(ft2)

        return fuse

    def _lateral_padding(self, inputs, pad=0):
        sz = inputs.size()
        padding = A.Variable(torch.zeros(sz[0], pad), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded


class FusNet(nn.Module):  #simple Baseline
    def __init__(self, args, digit_first_hidden, image_channels):
        super(FusNet, self).__init__()

        self.image_net = GP_VGG(args)  ###이미지 모달 가져오기
        self.digit_net = SimpleRNN(args, num_hidden=100, number_input_feats=1) ##digit 모달 가져오기

        self.classifier = nn.Linear(int(512 + 2 * digit_first_hidden), args.num_outputs)

        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, digit, image):
        im_gp1, im_gp2, im_gp3, im_gp4, im_out = self.image_net(image)
        d_out = self.digit_net(digit)

        im_gp4 = self.bn4(im_gp4)

        multimodal_feat = torch.cat((d_out, im_gp4), 1) ####
        out = self.classifier(multimodal_feat)

        return out

    def central_params(self):
        central_parameters = [
            {'params': self.classifier.parameters()}
        ]

        return central_parameters





# -------------- GRAPH --------------
class Maxout(nn.Module):
    def __init__(self, d, m, k):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(dim=max_dim)
        return m

####
class MaxOut_MLP(nn.Module):

    def __init__(self, args, first_hidden=64, number_input_feats=300): #######
        super(MaxOut_MLP, self).__init__()

        self.op1 = Maxout(number_input_feats, first_hidden, 5)  ###
        self.op2 = nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(0.5))
        self.op3 = Maxout(first_hidden, first_hidden * 2, 5)
        self.op4 = nn.Sequential(nn.BatchNorm1d(first_hidden * 2), nn.Dropout(0.5))

        # The linear layer that maps from hidden state space to output space
        self.hid2val = nn.Linear(first_hidden * 2, args.num_outputs)

    def forward(self, x):
        o1 = self.op1(x)
        o2 = self.op2(o1)
        o3 = self.op3(o2)
        o4 = self.op4(o3)
        o5 = self.hid2val(o4)

        return o1, o3, o5  ### o2,o4는 안 쓰고 이렇게 세개만 쓰겠다.



class SimpleVTNet(nn.Module): #baseline
    def __init__(self, args, digit_first_hidden, image_channels):
        super(SimpleVTNet, self).__init__()

        self.image_net = GP_LeNet_Deeper(args, image_channels)
        self.digit_net = MaxOut_MLP(args, digit_first_hidden)

        self.classifier = nn.Linear(int(16 * args.channels + 2 * digit_first_hidden), args.num_outputs)

    def forward(self, digit, image):
        im_gp1, im_gp2, im_gp3, im_gp4, im_gp5, im_out = self.image_net(image)
        t_o1, t_o2, t_out = self.digit_net(digit)

        multimodal_feat = torch.cat((t_o2, im_gp5), 1)
        out = self.classifier(multimodal_feat)

        return out

    def central_params(self):
        central_parameters = [
            {'params': self.classifier.parameters()}
        ]

        return central_parameters


# %%



################################################################
####FUSING  메인 모델 ###########################################

class VGGVTNet(nn.Module):
    def __init__(self, args, digit_first_hidden, image_channels):
        super(VGGVTNet, self).__init__()

        self.image_net = GP_VGG(args)  ###이미지 모달 가져오기
        self.digit_net = MaxOut_MLP(args, digit_first_hidden) ###텍스트 모달 가져오기

        self.classifier = nn.Linear(int(512 + 2 * digit_first_hidden), args.num_outputs)

        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, digit, image):
        im_gp1, im_gp2, im_gp3, im_gp4, im_out = self.image_net(image)
        t_o1, t_o2, t_out = self.digit_net(digit)

        im_gp4 = self.bn4(im_gp4)

        multimodal_feat = torch.cat((t_o2, im_gp4), 1)
        out = self.classifier(multimodal_feat)

        return out

    def central_params(self):
        central_parameters = [
            {'params': self.classifier.parameters()}
        ]

        return central_parameters


# %%


    # %%

"""
class SimpleVT_CentralNet(nn.Module):
    def __init__(self, args, digit_first_hidden, image_channels):
        super(SimpleVT_CentralNet, self).__init__()

        self.args = args
        self.image_net = GP_LeNet_Deeper(args, image_channels)
        self.digit_net = MaxOut_MLP(args, digit_first_hidden)  ###여기서 maxout mlp 가져옴

        self.alpha1_feat1 = nn.Parameter(torch.rand(1))
        self.alpha2_feat1 = nn.Parameter(torch.rand(1))

        self.alpha1_feat2 = nn.Parameter(torch.rand(1))
        self.alpha2_feat2 = nn.Parameter(torch.rand(1))

        self.alpha_conv1 = nn.Parameter(torch.rand(1))
        self.alpha_conv2 = nn.Parameter(torch.rand(1))

        self.central_conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.central_conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        if self.args.fusingmix == '11,25' or self.args.fusingmix == '13,25':
            nodes = 384

        if self.args.fusingmix == '11,23':
            nodes = 256

        if self.args.fusetype == 'cat':
            nodes *= 2

        self.central_classifier = nn.Linear(nodes, args.num_outputs)

    def central_params(self):

        central_parameters = [
            {'params': self.central_conv1.parameters()},
            {'params': self.central_conv2.parameters()},
            {'params': self.alpha1_feat1},
            {'params': self.alpha2_feat1},
            {'params': self.alpha1_feat2},
            {'params': self.alpha2_feat2},
            {'params': self.alpha_conv1},
            {'params': self.alpha_conv2},
            {'params': self.central_classifier.parameters()}]

        return central_parameters

    def forward(self, digit, image):

        im_gp1, im_gp2, im_gp3, im_gp4, im_gp5, im_out = self.image_net(image)
        t_o1, t_o2, t_out = self.digit_net(digit)

        one = A.Variable(torch.ones(1))  # ugly hack to improve
        if image.is_cuda:
            one = one.cuda()

        if self.args.fusingmix == '11,23':
            fuse1 = self._fuse_features(t_o1, im_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp3, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        elif self.args.fusingmix == '11,25':
            fuse1 = self._fuse_features(t_o1, im_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp5, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        elif self.args.fusingmix == '13,25':
            fuse1 = self._fuse_features(t_o1, im_gp2, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp5, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        else:
            raise ValueError(
                'self.args.fusingmix {} fusion combinantion is not implemented'.format(self.args.fusingmix))

        cc1 = F.relu(self.central_conv1(fuse1.unsqueeze(1)))
        cc1 = self._fuse_features(cc1[:, 0, :], fuse2, self.alpha_conv1, one, 'wsum')

        cc2 = F.relu(self.central_conv2(cc1.unsqueeze(1)))

        fusion_out = self.central_classifier(cc2[:, 0, :])

        return t_out, im_out, fusion_out

    def _fuse_features(self, ft1,ft2, a1, a2, fusetype):

        ft1sz = ft1.size()
        ft2sz = ft2.size()

        dif = ft1sz[1] - ft2sz[1]

        if fusetype == 'cat':
            if dif > 0:
                fuse = torch.cat((ft1, self._lateral_padding(ft2, dif)), 1)
            elif dif < 0:
                fuse = torch.cat((self._lateral_padding(ft1, -dif), ft2), 1)
            else:
                fuse = torch.cat((ft1,ft2), 1)

        elif fusetype == 'wsum':
            if dif > 0:
                fuse = ft1 * a1.expand_as(ft1) + self._lateral_padding(ft2, dif) * a2.expand_as(ft1)
            elif dif < 0:
                fuse = self._lateral_padding(ft1, -dif) * a1.expand_as(ft2) + ft2 * a2.expand_as(ft2)
            else:
                fuse = ft1 * a1.expand_as(ft1) + ft2 * a1.expand_as(ft2)

        return fuse

    def _lateral_padding(self, inputs, pad=0):
        sz = inputs.size()
        padding = A.Variable(torch.zeros(sz[0], pad), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded

    # %% """

class VGGT_CentralNetV2(nn.Module):
    def __init__(self, args, digit_first_hidden, image_channels):
        super(VGGT_CentralNetV2, self).__init__()

        self.args = args
        self.image_net = GP_VGG(args)
        self.digit_net = MaxOut_MLP(args, digit_first_hidden)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        nodes = args.channels
        inunits = 512  #####################################################뭘말하는

        self.alpha1_feat1 = nn.Parameter(torch.from_numpy(np.zeros((1, inunits), np.float32))) #학습가능한 파라미터??
        self.alpha2_feat1 = nn.Parameter(torch.from_numpy(np.zeros((1, inunits), np.float32))) #alpha 는 연결 강도인데
        self.alpha1_feat2 = nn.Parameter(torch.from_numpy(np.zeros((1, inunits), np.float32)))
        self.alpha2_feat2 = nn.Parameter(torch.from_numpy(np.zeros((1, inunits), np.float32)))

        if self.args.fusetype == 'cat':
            nodes *= 2 ##컨켓을 했다면 node 수 2배로 , 들어가는 in unit도 2배로
            inunits *= 2

        self.alpha_conv1 = nn.Parameter(torch.from_numpy(np.zeros((1, args.channels), np.float32)))
        self.alpha_conv2 = nn.Parameter(torch.from_numpy(np.zeros((1, args.channels), np.float32)))

        self.central_conv1 = nn.Linear(inunits, args.channels)
        self.central_conv2 = nn.Linear(inunits, args.channels)

        self.gp1 = aux.GlobalPooling1D()
        self.gp2 = aux.GlobalPooling1D()

        self.bnc1 = nn.BatchNorm1d(args.channels)
        self.bnc2 = nn.BatchNorm1d(args.channels)

        self.central_classifier = nn.Linear(nodes, args.num_outputs) ###합친 모델의 마지막 부분?

    def central_params(self):

        central_parameters = [  ###이게 뭐하는?
            {'params': self.central_conv1.parameters()}, ######??
            {'params': self.central_conv2.parameters()},
            {'params': self.alpha1_feat1},
            {'params': self.alpha2_feat1},
            {'params': self.alpha1_feat2},
            {'params': self.alpha2_feat2},
            {'params': self.alpha_conv1},
            {'params': self.alpha_conv2},
            {'params': self.central_classifier.parameters()}]

        return central_parameters

    def forward(self, digit, image): ####본격 아키텍처 빌드 부분

        im_gp1, im_gp2, im_gp3, im_gp4, im_out = self.image_net(image)  #이미지를 이미지넷에 통과 시킨 결과.... 5개 줄이 나온다.
            #중간에서 나오는 4개는 global pooling을 통해 차원수 통일시킴

        im_gp1 = self.bn1(im_gp1) ## 각각 배치놈
        im_gp2 = self.bn2(im_gp2)
        im_gp3 = self.bn3(im_gp3)
        im_gp4 = self.bn4(im_gp4)


        ######

        t_o1, t_o2, t_out = self.digit_net(digit)  ###텍스트를 텍스트넷에 통과시킨 결과... 중간 노드 2개 + final output

        one = A.Variable(torch.ones(1))  # ugly hack to improve
        if image.is_cuda:
            one = one.cuda()

        if self.args.fusingmix == '11,24':  #####???????????
            fuse1 = self._fuse_features(t_o1, im_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        elif self.args.fusingmix == '13,24':
            fuse1 = self._fuse_features(t_o1, im_gp3, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        elif self.args.fusingmix == '12,24':
            fuse1 = self._fuse_features(t_o1, im_gp2, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(t_o2, im_gp4, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
        else:
            raise ValueError(
                'self.args.fusingmix {} fusion combinantion is not implemented'.format(self.args.fusingmix))

        cc1 = F.relu(self.central_conv1(fuse1))
        cc2 = F.relu(self.central_conv2(fuse2))

        # cc1 = F.relu(self.central_conv1(fuse1.unsqueeze(1)))
        # cc2 = F.relu(self.central_conv2(fuse2.unsqueeze(1)))

        # fuse3 = self.gp1(cc)
        # cc2 = self.gp2(cc2)

        fuse3 = self._fuse_features(cc1, cc2, self.alpha_conv1, self.alpha_conv2, self.args.fusetype)

        fusion_out = self.central_classifier(fuse3)

        return t_out, im_out, fusion_out

    def _fuse_features(self, ft1,ft2, a1, a2, fusetype):

        ft1sz = ft1.size()  ##피처 1 사이즈
        ft2sz = ft2.size()

        bsz = ft1sz[0]
        dif = ft1sz[1] - ft2sz[1] #피처 1 2 사이즈의 차이

        if fusetype == 'cat':
            if dif > 0:
                in1 = ft1 * F.sigmoid(a1.expand(bsz, -1))
                in2 = self._lateral_padding(ft2, dif) * F.sigmoid(a2.expand(bsz, -1))

            elif dif < 0:
                in1 = self._lateral_padding(ft1, -dif) * F.sigmoid(a1.expand(bsz, -1))  #####
                in2 = ft2 * F.sigmoid(a2.expand(bsz, -1))
            else:
                in1 = ft1 * F.sigmoid(a1.expand(bsz, -1))
                in2 = ft2 * F.sigmoid(a2.expand(bsz, -1))

            fuse = torch.cat((in1, in2), 1)
        elif fusetype == 'wsum':
            if dif > 0:
                in1 = ft1 * F.sigmoid(a1.expand(bsz, -1))
                in2 = self._lateral_padding(ft2, dif) * F.sigmoid(a2.expand(bsz, -1))

            elif dif < 0:
                in1 = self._lateral_padding(ft1, -dif) * F.sigmoid(a1.expand(bsz, -1))
                in2 = ft2 * F.sigmoid(a2.expand(bsz, -1))
            else:
                in1 = ft1 * F.sigmoid(a1.expand_as(ft1))
                in2 = ft2 * F.sigmoid(a2.expand_as(ft2))

            fuse = in1 + in2

        return fuse

    def _lateral_padding(self, inputs, pad=0):
        sz = inputs.size()
        padding = A.Variable(torch.zeros(sz[0], pad), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded

    # %%


class WeightedCrossEntropyWithLogits(nn.Module):

    def __init__(self, pos_weight):
        super(WeightedCrossEntropyWithLogits, self).__init__()
        self.w = pos_weight

    def forward(self, logits, targets):
        q = [self.w] * logits.size()[0]
        q = torch.from_numpy(np.asarray(q, np.float32)).to(logits.device)

        x = F.sigmoid(logits)
        z = targets

        L = q * z * -torch.log(x) + (1 - z) * -torch.log(1 - x)
        # l = (1 + (q - 1) * z)
        # L = (1 - z) * x + l * (torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(-x, 0)[0])

        totloss = torch.mean(torch.mean(L))
        return totloss



# %%
class GP_LeNet_Deeper(nn.Module):
    def __init__(self, args, in_channels):
        super(GP_LeNet_Deeper, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, args.channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.channels))
        self.gp1 = aux.GlobalPooling2D()

        self.conv2 = nn.Conv2d(args.channels, 2 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * args.channels))
        self.gp2 = aux.GlobalPooling2D()

        self.conv3 = nn.Conv2d(2 * args.channels, 4 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(4 * args.channels))
        self.gp3 = aux.GlobalPooling2D()

        self.conv4 = nn.Conv2d(4 * args.channels, 8 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(8 * args.channels))
        self.gp4 = aux.GlobalPooling2D()

        self.conv5 = nn.Conv2d(8 * args.channels, 16 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(16 * args.channels))
        self.gp5 = aux.GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(16 * args.channels), args.num_outputs)
        )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)
        gp1 = self.gp1(out)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out4 = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out4, 2)
        gp4 = self.gp4(out4)

        out5 = F.relu(self.bn5(self.conv5(out)))
        out = F.max_pool2d(out5, 2)
        gp5 = self.gp5(out5)

        out = self.classifier(gp5)

        return gp1, gp2, gp3, gp4, gp5, out

##""" #RNN은 왜하는거?
# %%



