import os
# import g711
import random
import soundfile
import numpy as np

import torch
from torch.utils.data import Dataset

from torch  import Tensor
from numpy  import ndarray
from typing import List, Tuple


def load_wave(filepath:str, duration:float=None) -> ndarray:
    wave, _ = soundfile.read(filepath, dtype='float32')

    if duration is not None:
        wave_len = wave.shape[0]
        clip_len = int(duration * 16000)

        if wave_len <= clip_len:
            n_concat = (clip_len // wave_len) + 1
            wave     = np.concatenate([wave] * n_concat, axis=0)
            wave_len = wave.shape[0]

        offset = random.randint(0, wave_len - clip_len)
        wave   = wave[offset:offset+clip_len]

    return wave


class WaveMixer:
    def __init__(self, mix_ratio:List[float]=[0.01, 0.3], hop_length:int=128) -> None:
        self.mix_ratio  = mix_ratio
        self.hop_length = hop_length

    def concate(self, ref_wave:ndarray, cat_path:str) -> Tuple[ndarray, List[int]]:
        ratio    = random.uniform(*self.mix_ratio)
        cat_wave = load_wave(cat_path, len(ref_wave) / 16000 * ratio)

        offset = random.randint(0, len(ref_wave) - len(cat_wave))
        ref_wave[offset:offset+len(cat_wave)] = cat_wave

        position = [offset // self.hop_length, (offset + len(cat_wave) - 1) // self.hop_length]
        position = list(map(lambda p: max(0,  p - 1), position))

        return ref_wave, position


class WaveAugmentor():
    def __init__(self, noise_dir:str, rir_dir:str) -> None:
        self.noise_types   = ['noise', 'speech', 'music']
        self.noise_snr     = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.noise_pathset = {'noise': [], 'speech': [], 'music': []}
        
        for noise_type in self.noise_types:
            for r, _, fs in os.walk(os.path.join(noise_dir, noise_type)):
                self.noise_pathset[noise_type] += [ os.path.join(r, f) for f in fs if f.endswith('.wav') ]

        self.rir_paths = []
        for r, _, fs in os.walk(rir_dir):
            self.rir_paths += [ os.path.join(r, f) for f in fs if f.endswith('.wav') ]
        
        # self.codec_types = ['alaw', 'ulaw']

    def add_noise(self, clean_wave:ndarray) -> ndarray:
        noise_type = random.choice(self.noise_types)
        noise_path = random.choice(self.noise_pathset[noise_type])
        noise_wave = load_wave(noise_path, len(clean_wave) / 16000)

        clean_db = 10 * np.log10(np.mean(clean_wave ** 2) + 1e-9)
        noise_db = 10 * np.log10(np.mean(noise_wave ** 2) + 1e-9)
        snr      = random.uniform(*self.noise_snr[noise_type])

        noisy_wave = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise_wave + clean_wave
        if max(abs(noisy_wave)) > 1.:
            noisy_wave /= max(abs(noisy_wave))

        return noisy_wave
        
    def add_reverb(self, wave:ndarray) -> ndarray:
        rir_path = random.choice(self.rir_paths)

        rir, _ = soundfile.read(rir_path, dtype='float32')
        rir = rir / np.sqrt(np.sum(rir ** 2))

        return np.convolve(wave, rir, mode='full')[:len(wave)]

    # def add_codec(self, wave:ndarray) -> ndarray:
    #     codec = random.choice(self.codec_types)

    #     if codec == 'alaw':
    #         wave = g711.decode_alaw(g711.encode_alaw(wave))
    #     elif codec == 'ulaw':
    #         wave = g711.decode_ulaw(g711.encode_ulaw(wave))

    #     return wave


class AddDataset(Dataset):
    def __init__(self, data_dir:str, data_list:str, duration:float=None, mix:bool=False, mix_ratio:List[float]=[0.01, 0.3], \
            hop_length:int=128, augment:bool=False, noise_dir:str=None, rir_dir:str=None) -> None:
        super(AddDataset, self).__init__()

        with open(data_list) as infile:
            lines = infile.readlines()

        pathset = {'fake': [], 'genuine': []}
        for line in lines:
            name, cls = line.strip().split()
            pathset[cls] += [ os.path.join(data_dir, name) ]

        if mix:
            self.paths     = pathset['genuine']
            self.labels    = [1] * len(pathset['genuine'])            
            
            self.cat_paths = pathset['fake'] + pathset['genuine']
            self.mixer     = WaveMixer(mix_ratio, hop_length)
        else:
            self.paths  = pathset['fake'] + pathset['genuine']
            self.labels = [0] * len(pathset['fake']) + [1] * len(pathset['genuine'])            

        if augment:
            self.augmentor = WaveAugmentor(noise_dir, rir_dir)

        self.mix      = mix
        self.augment  = augment
        self.duration = duration

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        path  = self.paths[idx]
        wave  = load_wave(path, self.duration)
        label = self.labels[idx]
        posit = [0, 0]

        if self.mix:
            label = random.choice([0, 1])
            if label == 0:
                cat_path = path
                while cat_path == path:
                    cat_path = random.choice(self.cat_paths)
                wave, posit = self.mixer.concate(wave, cat_path)

        if self.augment:
            # aug_type = random.choice(['none', 'noise', 'reverb', 'codec'])
            aug_type = random.choice(['none', 'noise', 'reverb'])
            if aug_type == 'noise':
                wave = self.augmentor.add_noise(wave)
            elif aug_type == 'reverb':
                wave = self.augmentor.add_reverb(wave)
            # elif aug_type == 'codec':
            #     wave = self.augmentor.add_codec(wave)
        
        wave  = torch.tensor(wave , dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        posit = torch.tensor(posit, dtype=torch.int64)
        
        return wave, label, posit

    def __len__(self):
        return len(self.paths)


class EvalDataset(Dataset):
    def __init__(self, data_dir:str, duration:float=None) -> None:
        super(EvalDataset, self).__init__()
        
        self.names = []
        self.paths = []
        for name in os.listdir(data_dir):
            if name.endswith('.wav'):
                self.names += [ name ]
                self.paths += [ os.path.join(data_dir, name) ]

        self.duration = duration

    def __getitem__(self, idx) -> Tuple[Tensor, str]:
        name = self.names[idx]
        path = self.paths[idx]
        wave = load_wave(path, self.duration)

        return torch.FloatTensor(wave), name

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dataset = AddDataset( 
        data_dir ='/home/hckuo/Project/ADD/data/ADD/',
        data_list='/home/hckuo/Project/ADD/data/ADD/label/train_dev_vocoder_list.txt',
        duration =4.,
        mix=True,
        mix_ratio=[0.01, 0.3],
        hop_length=128,
        augment=True,
        noise_dir='/mnt/Dataset/musan',
        rir_dir='/mnt/Dataset/simulated_rirs_16k'
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    for i, (batch_x, batch_y, batch_pos) in enumerate(loader):
        print(i ,batch_x.size(), batch_y.size(), batch_pos.size())

    
    dataset = AddDataset( 
        data_dir ='/home/hckuo/Project/ADD/data/ADD/',
        data_list='/home/hckuo/Project/ADD/data/ADD/label/track2adp_list.txt',
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    for i, (batch_x, batch_y, batch_pos) in enumerate(loader):
        print(i ,batch_x.size(), batch_y.size(), batch_pos.size())