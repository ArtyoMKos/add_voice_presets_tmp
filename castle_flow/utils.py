import random
from typing import Optional, Union

import torch
import torchaudio
import torchvision
from audiokit.utils.audio import mel_spectrogram_torch

from castle_flow.autoencoder.config import AutoEncoderDatasetConfig
from castle_flow.flow_decoder.config import FlowDecoderDatasetConfig


def get_rand_slice_start(length: int, segment_size: int) -> int:
    max_start_idx = length - segment_size
    start_idx = random.randint(0, max_start_idx)
    return start_idx


def get_context_mask(
    sequence: torch.Tensor,
    drop_prob: float,
    min_seq_ratio: float,
    max_seq_ratio: float,
    left: bool = False
) -> torch.Tensor:
    drop = torch.bernoulli(torch.tensor(drop_prob))
    if drop:
        return torch.zeros_like(sequence)
    r = random.uniform(min_seq_ratio, max_seq_ratio)
    seq_len = sequence.shape[-1]
    mask_len = round(r * seq_len)
    mask = torch.ones_like(sequence)
    ss = -mask_len if left else random.randint(0, seq_len - mask_len)
    ee = None if left else ss + mask_len
    mask[..., ss:ee] = 0.0
    return mask


def get_mel(
    audio: torch.Tensor,
    config: Union[AutoEncoderDatasetConfig, FlowDecoderDatasetConfig]
) -> torch.Tensor:
    return mel_spectrogram_torch(
        y=audio, # [batch_size, n_samples]
        n_fft=config.filter_length,
        num_mels=config.n_mels,
        sampling_rate=config.sampling_rate,
        hop_size=config.hop_length,
        win_size=config.win_length,
        fmin=config.mel_fmin,
        fmax=config.mel_fmax,
        center=False,
        padding=True
    )

# Create a learning rate scheduler to implement the linear warmup and decay
def lr_lambda_wrapper(max_step, warmup_step):
    def lr_lambda(current_step):
        if current_step <= warmup_step:
            return current_step / warmup_step
        else:
            return 1 - (current_step - warmup_step) / (max_step - warmup_step)

    return lr_lambda


def vsr_transform(
    mel: torch.FloatTensor, 
    height=None, 
    min_r=0.85, # 68 for 80 mels
    max_r=1.15 # 92 for 80 mels
) -> torch.FloatTensor:
    if height is None:
        r = random.uniform(min_r, max_r)
        height = round(mel.size(-2)*r)

    tgt = torchvision.transforms.functional.resize(
        mel, 
        (height, mel.size(-1)), 
        antialias=True
    )
    if height >= mel.size(-2):
        return tgt[:, :mel.size(-2), :]
    else:
        silence = tgt[:,-1:,:].repeat(1,mel.size(-2)-height,1) 
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


def vsr_transform_batch(
    mels: torch.FloatTensor,
    mel_lengths: torch.LongTensor,
    heights=None,
    min_r=0.85,
    max_r=1.15
) -> torch.FloatTensor:
    mels_vsr = mels.clone()
    for i in range(mels.size(0)):
        height = None
        if heights is not None:
            height = heights[i]
        mel = mels[i][:, :mel_lengths[i]]
        mel = mel[None, :, :]
        mels_vsr[i][:, :mel_lengths[i]] = vsr_transform(
            mel,
            height=height, 
            min_r=min_r, 
            max_r=max_r
        )
    return mels_vsr


def get_context_mask(
    sequence: torch.Tensor,
    drop_prob: float,
    min_seq_ratio: float,
    max_seq_ratio: float,
    left: bool = False
):
    drop = torch.bernoulli(torch.tensor(drop_prob))
    if drop:
        return torch.zeros_like(sequence)
    r = random.uniform(min_seq_ratio, max_seq_ratio)
    seq_len = sequence.shape[-1]
    mask_len = round(r * seq_len)
    mask = torch.ones_like(sequence)
    ss = -mask_len if left else random.randint(0, seq_len - mask_len)
    ee = None if left else ss + mask_len
    mask[..., ss:ee] = 0.0
    return mask


def load_audio(
    audio_path: str,
    chunk_size: Optional[int] = None,
    cut_left_chunk: bool = False,
    cut_right_chunk: bool = False,
    cut_random_chunk: bool = False,
    device: str = 'cpu',
) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(audio_path, format=audio_path.split('.')[-1])
    audio = audio.mean(dim=0)
    
    if chunk_size is not None:
        n_samples = audio.shape[-1]
        n_samples_in_chunk = min(int(chunk_size * sr), n_samples)
        if cut_left_chunk:
            audio = audio[ :n_samples_in_chunk]
        elif cut_right_chunk:
            audio = audio[-n_samples_in_chunk:]
        elif cut_random_chunk:
            start = random.randrange(0, n_samples - n_samples_in_chunk)
            end = start + n_samples_in_chunk
            audio = audio[start: end]
    
    return audio.to(device), sr
