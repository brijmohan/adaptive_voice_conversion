'''
Generates TSNE scatter plot for a fixed number of speakers from spk2utt file
'''

import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE
from utils import *
from functools import reduce
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms
import librosa 

from os.path import join, isdir


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_UTT_PER_SPK = 20
X = []

MAX_MALE = 5
MAX_FEMALE = 5

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class Inferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)
        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.args.model}')
        self.model.load_state_dict(torch.load(f'{self.args.model}',
                                              map_location=torch.device(device_name)))
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance_spk(self, uttpath):
        x, _ = get_spectrograms(uttpath)
        x = torch.from_numpy(self.normalize(x)).cuda()
        x = self.utt_make_frames(x)
        emb = self.model.get_speaker_embeddings(x)
        emb = emb.detach().cpu().numpy()
        return emb

    def inference_one_utterance_content(self, uttpath):
        x, _ = get_spectrograms(uttpath)
        x = torch.from_numpy(self.normalize(x)).cuda()
        x = self.utt_make_frames(x)
        emb = self.model.get_content_repr(x)
        emb = emb.detach().cpu().numpy()
        emb = np.mean(emb, axis=2)
        return emb

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def inference_from_path(self):
        S, C = [], []

        with open(self.args.source) as f:
            for line in f.read().splitlines():
                sp = line.split()[0]
                S.append(self.inference_one_utterance_spk(sp))
                C.append(self.inference_one_utterance_content(sp))

        S = np.concatenate(S, axis=0)
        C = np.concatenate(C, axis=0)

        print("Speaker embeddings:", S.shape)
        print("Content embeddings:", C.shape)

        np.save(self.args.output+"_content.npy", C)
        np.save(self.args.output+"_spk.npy", S)
        
        return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', help='attr file path')
    parser.add_argument('-config', '-c', help='config file path')
    parser.add_argument('-model', '-m', help='model path')
    parser.add_argument('-source', '-s', help='filelist with paths')
    parser.add_argument('-output', '-o', help='Output numpy object with same sequence of vectors as filelist')
    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)
    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()
