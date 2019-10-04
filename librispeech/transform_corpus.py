import torch
import os 
import yaml
from model import AE
from utils import *
from collections import defaultdict
from argparse import ArgumentParser, Namespace
#from scipy.io.wavfile import write
import soundfile as sf
import random
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms

from os.path import join
import glob
import random

class CorpusConvertor(object):
    def __init__(self, config, args):
        self.config = config
        print(config)
        self.args = args
        print(args)

        self.build_model()
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

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        sf.write(output_path, wav_data, self.args.sample_rate)
        return

    def convert_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        #x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def convert_from_path(self, srcpath, tframes, opath):
        src_mel, _ = get_spectrograms(srcpath)
        #tar_mel, _ = get_spectrograms(trgpath)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        #tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        conv_wav, conv_mel = self.convert_one_utterance(src_mel, tframes)
        self.write_wav_to_file(conv_wav, opath)
        return

    def get_mel_frames(self, apath):
        tar_mel_frames, _ = get_spectrograms(apath)
        tar_mel_frames = torch.from_numpy(self.normalize(tar_mel_frames)).cuda()
        tar_mel_frames = self.utt_make_frames(tar_mel_frames)
        return tar_mel_frames


    def main():
        input_paths = self.args.input_paths
        output_dir = self.args.output
        stgy = self.args.strategy

        os.makedirs(output_dir, exist_ok=True)

        # Read all audio files with paths
        print("Reading all audio files...")
        all_inp_files = []
        for p in input_paths:
            all_inp_files.extend(glob.glob(join(p, '*/*/*.flac')))

        # Create speaker mapping from path
        print("Creating speaker mapping from path...")
        spk2utt = defaultdict(lambda: [])
        for fpath in all_inp_files:
            sp = fpath.split('/')
            spk2utt[sp[-3]].append(fpath)

        print("Total number of speakers:", len(spk2utt))

        # Select targets
        print("Selecting targets based on strategy", stgy)
        spklist = spk2utt.keys()
        if stgy == "1":
            tgt_spk = random.choice(spklist)
            tgt_utt = random.choice(spk2utt[tgt_spk])
            with open(join(output_dir, 'targets'), 'w') as f:
                f.write(tgt_utt+'\n')
            tgt_utt = self.get_mel_frames(tgt_utt)
        elif stgy == "2":
            tgt_spks = random.sample(spklist, 100)
            tgt_map = {}
            with open(join(output_dir, 'targets'), 'w') as f:
                for spk in spklist:
                    cspk = random.choice(tgt_spks)
                    cutt = random.choice(spk2utt[cspk])
                    f.write(spk + " -> " + cutt + "\n")
                    tgt_map[spk] = self.get_mel_frames(cutt)
        elif stgy == "3":
            tgt_spks = random.sample(spklist, 100)
            tgt_map = {}
            with open(join(output_dir, 'targets'), 'w') as f:
                for spk, uttlist in spk2utt.items():
                    for utt in uttlist:
                        cspk = random.choice(tgt_spks)
                        cutt = random.choice(spk2utt[cspk])
                        f.write(utt + " -> " + cutt + "\n")
                        tgt_map[utt] = self.get_mel_frames(cutt)

        # Converting files
        print("Converting files...")
        for spk, uttlist in spk2utt.items():
            for utt in uttlist:
                sp = utt.split('/')
                utt_dir = join(output_dir, *sp[-4:-1])
                os.makedirs(utt_dir, exist_ok=True)
                utt_opath = join(output_dir, *sp[-4:])

                if stgy == "1":
                    self.convert_from_path(utt, tgt_utt, utt_opath)
                elif stgy == "2":
                    self.convert_from_path(utt, tgt_map[spk], utt_opath)
                elif stgy == "3":
                    self.convert_from_path(utt, tgt_map[utt], utt_opath)
        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', help='attr file path')
    parser.add_argument('-config', '-c', help='config file path')
    parser.add_argument('-model', '-m', help='model path')
    parser.add_argument('-strategy', '-s', help='Strategy for conversion 1, 2, 3')
    parser.add_argument('-output', '-o', help='Output directory, exact copy of source directories will be created')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=16000, type=int)
    parser.add_argument('input_paths', type=str, nargs='+', help='Input directories')
    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)
    convertor = CorpusConvertor(config=config, args=args)
    convertor.main()
