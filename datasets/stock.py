from unicodedata import digit
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
#import re
#import unicodedata
from PIL import Image #
import string
import argparse
from IPython import embed

#glove = []  # {w: vectors[word2idx[w]] for w in words}
#all_letters = string.ascii_letters + " .,;'"
fdim = 0


#지워도됨
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, digit, target = sample['image'], sample['digit'], sample['target']

        return {'image':torch.from_numpy(image.astype(np.float32)),
                # 'digit': digit,
                'digit':digit , # torch.from_numpy(digit.astype(np.float32)), #
                'target': target, #torch.from_numpy(target.astype(np.float32)), #
                'digitlen': sample['digitlen']}

class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, stdevs):
        self.mean_vector, self.stdevs = mean_vector, stdevs

    def __call__(self, sample):
        image = sample['image']
        return {'image': self._normalize(image, self.mean_vector, self.stdevs),
                'digit': sample['digit'],
                'target': sample['target'], 'digitlen': sample['digitlen']}

    def _normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self._is_tensor_image(tensor):
            print(tensor.size())
            raise TypeError('tensor is not a torch image. Its size is {}.'.format(tensor.size()))
        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def _is_tensor_image(self, img):
        return torch.is_tensor(img) and img.ndimension() == 3




class FusionDataset(object):
    def __init__(self, df, dir_path2, transforms=None,   stage='train',  feat_dim=100,args=None): #dffffff

        if stage == 'train':
            self.len_data = 1728
        elif stage == 'test':
            self.len_data = 221
        elif stage == 'val':
            self.len_data =472

        self.args=args
        self.dir_path=dir_path2

        self.transforms = transforms
        self.root_dir = dir_path2
        self.stage = stage
       # print(self.root_dir)  #/nas3/mink/charts/fus_SP500_bar_area/s_trix-s_Volume/train/

        df_imgs_path = pd.read_csv(dir_path2 + 'candlestick_target.csv')

        self.imgs = df_imgs_path.filename.tolist()

        self.digit_len = args.history_len
        self.step_len = args.step_len
        self.df = df #추가

        digit= []
        target = []



        print(self.df.shape) #0,3 ????
        for i in range(self.digit_len, self.df.shape[0] - self.step_len): #0~ 29까지
            print(i)
            digit.append(df[i - self.digit_len + 1: i]['close_log'].values) #-29에서 30까지??
            target.append(df.loc[i-1, 'target'])  ####target
            #print(digit)

        digit, target = np.array(digit), np.array(target)#
        #print(digit.shape)
        #print(target.shape)
        digit = np.reshape(digit, (digit.shape[0], digit.shape[1], 1))  ###dataset이 인식안되고 있음

        self.digit=digit #########추가

        self.target=target ###########추가

        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.digit_len

    def __getitem__(self, idx): ###index
        digit = torch.tensor(self.digit[idx], dtype=torch.float)
        # Load images
        img_path = os.path.join(self.dir_path+'/candlestick', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        digitlen = digit.shape[0]

        target = torch.tensor([self.target[idx]], dtype=torch.float) ############

        #    if self.transforms:
        #       sample = self.transforms(sample) #없으면 dfault collate error

        sample = {'image': img, 'digit': digit, 'target': target, 'digitlen': digitlen}

        return img, digit, target

##########################



def collate_imdb(list_samples):
    global fdim
    max_digit_len = 0
    for sample in list_samples:
        L = len(sample['digit'])
        if max_digit_len < L:
            max_digit_len = L

    list_images = len(list_samples) * [None]
    list_digit = len(list_samples) * [None]
    list_target = len(list_samples) * [None]
    list_digitlen = len(list_samples) * [None]

    for i, sample in enumerate(list_samples):
        digit_sample_len = len(sample['digit'])

        digit_i = sample['digit'].astype(np.float32)
        padding = np.asarray([fdim * [-10.0]] * (max_digit_len - digit_sample_len), np.float32)

        list_images[i] = sample['image']
        list_target[i] = sample['target']
        if padding.shape[0] > 0:
            list_digit[i] = torch.from_numpy(np.concatenate((digit_i, padding), 0))
        else:
            list_digit[i] = torch.from_numpy(digit_i)
        list_digitlen[i] = sample['digitlen']

    images = torch.transpose(torch.stack(list_images), 1, 3)
    digit = torch.stack(list_digit)
    target = torch.stack(list_target)

    return {'image': images, 'digit': digit, 'target': target, 'digitlen': list_digitlen}

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description='Modality optimization.')
        parser.add_argument('--datadir', type=str, help='data directory',
                            default='~~~')
        #parser.add_argument('--small_dataset', action='store_true', default=False, help='dataset scale')
        parser.add_argument('--average_digit', action='store_true', default=False, help='averaging digit features')
        return parser.parse_args("")
    args = parse_args()

    dataset = STDATA(root_dir=args.datadir,
        transform=None,
        stage='val',
        feat_dim=300,
        args=args) ####

    embed()
