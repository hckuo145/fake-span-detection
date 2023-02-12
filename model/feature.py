import s3prl.hub as hub

import torch
import torch.nn as nn
import torchaudio

from torch import Tensor


class Spectrogram(nn.Module):

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> None:
        super(Spectrogram, self).__init__()
        
        self.torchfb = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hamming_window
        )
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
        
        return x


class MelSpectrogram(nn.Module):
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        hop_length: int = 160,
    ) -> None:
        super(MelSpectrogram, self).__init__()

        self.torchfb = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window_fn=torch.hamming_window, 
            n_mels=n_mels
        )
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
        
        return x


class LFCC(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 128,
        n_lfcc: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> None:
        super(LFCC, self).__init__()
        
        self.torchfb = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                'n_fft'     : n_fft,
                'hop_length': hop_length,
                'window_fn' : torch.hamming_window
                }
            )
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
        
        return x


class MFCC(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 80,
        n_fft: int = 512,
        n_mels: int = 80,
        hop_length: int = 160,
    ) -> None:
        super(MFCC, self).__init__()
        
        self.torchfb = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft'     : n_fft,
                'hop_length': hop_length,
                'window_fn' : torch.hamming_window,
                'n_mels'    : n_mels
            }
        )
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
        
        return x


class Wav2Vec(nn.Module):
    def __init__(self):
        super(Wav2Vec, self).__init__()
        
        # wav2vec_large
        self.extractor = hub.wav2vec()

    @torch.no_grad()
    def forward(self, wavs):
        return self.extractor(list(wavs))['c']


class Wav2Vec2(nn.Module):
    def __init__(self):
        super(Wav2Vec2, self).__init__()
        
        # wav2vec_base_960
        self.extractor = hub.wav2vec2()

    @torch.no_grad()
    def forward(self, wavs):
        return self.extractor(list(wavs))['last_hidden_state'].permute(0, 2, 1).contiguous()


class WavLM(nn.Module):
    def __init__(self):
        super(WavLM, self).__init__()
        
        # wavlm_base_plus
        self.extractor = hub.wavlm()

    @torch.no_grad()
    def forward(self, wavs):
        return self.extractor(list(wavs))


class HuBERT(nn.Module):
    def __init__(self):
        super(HuBERT, self).__init__()
        
        # hubert_base
        self.extractor = hub.hubert()
    
    @torch.no_grad()
    def forward(self, wavs):
        return self.extractor(list(wavs))



if __name__ == '__main__':
    wavs = torch.randn(2, 16000 * 4 - 10)

    frontend = MelSpectrogram(
        n_fft     = 512,
        n_mels    = 80,
        hop_length= 128
    )
    feats = frontend(wavs)
    print(feats.size())

    # frontend = Wav2Vec()
    # feats = frontend(wavs)
    # print(feats.size())

    frontend = Wav2Vec2()
    feats = frontend(wavs)
    print(feats.size())

    # frontend = WavLM()
    # feats = frontend(wavs)

    # frontend = HuBERT()
    # feats = frontend(wavs)