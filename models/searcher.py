""" class Searcher """
from datasets import stock as stock_data
import models.trainsearch.searchnetwork as tsearcher
from torch.utils.data import DataLoader


import os
root_dir = os.getcwd()

import pandas as pd
import numpy as np
import torch
#from datasets import stock as stock_data

import torchvision.transforms as transforms
from PIL import Image

class Searcher():
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger

        #self.df = pd.read_csv('/nas3/mink/NAS/feature_fusion/charts/index/sp_stscale.csv')#args.dir_path + 'candlestick_target.csv')

        train_df=pd.read_csv(args.dir_path+'/train/candlestick_target.csv')
        val_df =pd.read_csv(args.dir_path+'/val/candlestick_target.csv')
        test_df =pd.read_csv(args.dir_path+'/test/candlestick_target.csv')


        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_training = stock_data.FusionDataset(df=train_df, dir_path2=args.dir_path+'/train/', transforms=transformer, stage='train', feat_dim=300, args=args)
        dataset_val = stock_data.FusionDataset(df=val_df, dir_path2=args.dir_path+'/val/', transforms=transformer, stage='val', feat_dim=300, args=args)
        dataset_test = stock_data.FusionDataset(df=test_df, dir_path2=args.dir_path+'/test/', transforms=transformer, stage='test', feat_dim=300, args=args)

        datasets = {'train': dataset_training, 'val': dataset_val, 'test': dataset_test}
        self.dataloaders = {
            x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                          drop_last=False) for x in ['train', 'val', 'test']}

    def search(self):
        least_mse, best_genotype = tsearcher.train_darts_model(self.dataloaders, self.args, self.device, self.logger) ###self.dataloaders 디렉토리
        return least_mse, best_genotype





###추가

class FusionDataset(object):
    def __init__(self, df,args,  transform): #history_len, step_len, dir_path,
        self.args = args
        self.dir_path = args.dir_path
        self.transforms = transforms
        df_imgs_path = pd.read_csv(args.dir_path + '/'+ 'candlestick_target.csv') # /추가
        self.imgs = df_imgs_path.filename.tolist()
        # self.log_target = df.target.tolist()

        self.history_len = args.history_len
        self.step_len = args.step_len

        df['close_log'] = np.log(df['Close'] / df['Close'].shift(1))
        # df['vol_log'] = np.log(df['Volume'] / df['Volume'].shift(1))
        df['target_log'] = np.log(df['Close'].shift(-self.step_len) / df['Close'])
        history = []
        target = []

        for i in range(args.history_len, df.shape[0], args.step_len):

            history.append(df[i-args.history_len+1: i]['close_log'].values)
            target.append(df.loc[i-1, 'target_log'])

        history, target = np.array(history), np.array(target)
        history = np.reshape(history, (history.shape[0], history.shape[1], 1))

        self.history = history
        self.target = target

    def __getitem__(self, idx):
        history = torch.tensor(self.history[idx], dtype=torch.float)
        # Load images
        img_path = os.path.join(self.dir_path, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        target = torch.tensor([self.target[idx]], dtype=torch.float)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, history, target ###

    def __len__(self):
        return self.history.shape[0]####################################################################

