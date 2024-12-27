import math
import os

import torch
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from audiokit.utils import load_metadata

from castle_flow.utils import get_context_mask
from castle_flow.flow_decoder.config import FlowDecoderDatasetConfig


class AudioLoader(Dataset):

    def __init__(
        self,
        metadata_paths: list[str],
        config: FlowDecoderDatasetConfig,
    ) -> None:
        
        self.config = config
        self.load_metadata(metadata_paths)
    
    def load_metadata(self, paths: list[str]):
        metadata = load_metadata(paths)
        self.audio_paths = [row[0] for row in metadata]
        self.lengths = [int(row[-1]) for row in metadata]

    def get_audio(
        self,
        audio_path: str,
        max_frames: int = -1
    ) -> tuple[torch.Tensor, int]:

        audio, sampling_rate = torchaudio.load(
            audio_path, 
            format=audio_path.split('.')[-1],
        )
        num_frames = audio.size(-1)
        if max_frames > 0 and num_frames > max_frames:
            start_frame = np.random.randint(0, num_frames - max_frames)
            audio = audio[..., start_frame:start_frame + max_frames]
        
        return audio.squeeze(0), sampling_rate

    def get_sample(self, index: int) -> dict:
        audio_path = os.path.join(self.config.dataset_root, self.audio_paths[index])
        
        max_duration_samples = self.config.max_duration * self.config.source_sampling_rate
        audio, sr = self.get_audio(audio_path, max_frames=int(max_duration_samples))

        assert sr == self.config.source_sampling_rate, \
            f"Expected {self.config.source_sampling_rate} sr, got {sr} sr"
        
        target_audio_size = math.ceil(
            audio.size(-1) * (self.config.sampling_rate / self.config.source_sampling_rate)
        )
        spec_len = target_audio_size // self.config.hop_length
        padding_len = (
            self.config.downsample_factor - spec_len % self.config.downsample_factor
        ) % self.config.downsample_factor
        spec_len = spec_len + padding_len
        
        context_mask = get_context_mask(
            sequence=torch.zeros((1, spec_len)),
            drop_prob=self.config.context_mask_drop_prob,
            min_seq_ratio=self.config.context_mask_min_ratio,
            max_seq_ratio=self.config.context_mask_max_ratio,
        )
        
        return dict(
            audio=audio,
            context_mask=context_mask,
            spec_len=spec_len,
        )

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.lengths)


class AudioCollate:

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        data = dict(
            audio=[sample['audio'] for sample in batch],
            spec_len=torch.LongTensor([sample['spec_len'] for sample in batch]),
            context_mask=torch.nn.utils.rnn.pad_sequence(
                [sample['context_mask'].transpose(0, 1) for sample in batch],
                batch_first=True,
                padding_value=0
            ).transpose(1, 2),
        )
        return data
