import os
import math

import torch
import random
import torchaudio

from torch.utils.data import Dataset
from audiokit.text import Tokenizer
from audiokit.utils import load_metadata, replace_file_suffix

from castle_flow.speech_token_predictor.config import STPDatasetConfig


class AudioLoader(Dataset):

    def __init__(
        self,
        metadata_paths: list[str],
        config: STPDatasetConfig,
        tokenizer: Tokenizer
    ) -> None:
        
        self.config = config
        self.tokenizer = tokenizer
        self.load_metadata(metadata_paths)
    
    def load_metadata(self, paths: list[str]) -> tuple[list[str], list[int]]:
        metadata = load_metadata(paths)
        metadata = [row for row in metadata if float(row[1]) <= self.config.max_frames]
        self.audio_paths = [row[0] for row in metadata]
        self.lengths = [int(row[1]) for row in metadata]
        self.speaker_ids = [row[2] for row in metadata]
        self.speaker_to_indices = {}  # Precompute mapping of speaker_id to list of indices

        # Create a mapping from speaker_id to indices
        for idx, speaker_id in enumerate(self.speaker_ids):
            if speaker_id not in self.speaker_to_indices:
                self.speaker_to_indices[speaker_id] = []
            self.speaker_to_indices[speaker_id].append(idx)

    def get_sample(self, index: int) -> dict:

        audio_path = self.audio_paths[index]
        audio_path = os.path.join(self.config.dataset_root, audio_path) # join here to not keep long strings in RAM
        
        audio, sr = self.get_audio(audio_path) # not resampled
        text = self.get_phonemes(audio_path)
        speaker_id = self.speaker_ids[index]
        audio, text = self.concat_augmentation(audio, text, speaker_id)

        target_audio_size = math.ceil(
            audio.size(-1) * (self.config.sampling_rate / self.config.source_sampling_rate)
        )
        spec_len = target_audio_size // self.config.hop_length
        padding_len = (
            self.config.downsample_factor - spec_len % self.config.downsample_factor
        ) % self.config.downsample_factor
        spec_len = spec_len + padding_len

        text_tokens = self.get_text_tokens(text)

        return dict(
            text_tokens=text_tokens,
            speech=audio,
            spec_len=spec_len,
        )
    
    def concat_augmentation(
        self, 
        audio: torch.Tensor, 
        phonemes: str, 
        speaker_id: str
    ) -> tuple[torch.Tensor, str]:
        """Perform concatenation-based augmentation on both audio and phonemes."""
        same_speaker_indices = self.speaker_to_indices[speaker_id]  # Fast lookup of indices
        
        if len(same_speaker_indices) > 1 and random.random() < self.config.concat_prob :
            other_index = random.choice(same_speaker_indices)
            if other_index != speaker_id:
                # Concatenate audio
                other_audio_path = os.path.join(
                    self.config.dataset_root, 
                    self.audio_paths[other_index])
                other_audio, _ = self.get_audio(other_audio_path)
                audio = torch.cat((audio, other_audio), dim=-1)  # Concatenate audio tensors

                # Concatenate phonemes
                other_phonemes = self.get_phonemes(other_audio_path)
                phonemes = phonemes + " " + other_phonemes  # Concatenate phoneme strings

        return audio, phonemes
    
    def get_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        audio, sampling_rate = torchaudio.load(audio_path, format=audio_path.split('.')[-1])
        assert sampling_rate == self.config.source_sampling_rate, \
            f'{audio_path} Expected {self.config.source_sampling_rate}Hz, got {sampling_rate}Hz'
        return audio.squeeze(0), sampling_rate
    
    def get_text_tokens(self, text: str) -> torch.LongTensor:
        text_tokens = self.tokenizer(text)
        return torch.LongTensor(text_tokens)
    
    def get_phonemes(self, audio_path: str) -> str:
        suffix = f'.{self.config.phoneme_postfix}.txt' if self.config.phoneme_postfix else '.txt'
        phonemes_path = replace_file_suffix(
            audio_path.replace(self.config.audio_dir, self.config.phoneme_dir),
            suffix=suffix
        )
        with open(phonemes_path, 'r') as f:
            phonemes = f.read().strip()
        return phonemes

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.lengths)


class AudioCollate:

    def __call__(self, batch: list[dict[str, torch.Tensor]]):
        max_token_len = max([x["text_tokens"].size(-1) for x in batch])
        max_speech_len = max([x["speech"].size(-1) for x in batch])
      
        text_tokens_padded = torch.LongTensor(len(batch), max_token_len)
        text_tokens_lengths = torch.LongTensor(len(batch))
        speech_padded = torch.FloatTensor(len(batch), max_speech_len)
        spec_lengths = torch.LongTensor(len(batch))
        text_tokens_padded.zero_()
        speech_padded.zero_()

        for idx, row in enumerate(batch):
            text_tokens = row["text_tokens"]
            text_tokens_padded[idx, :text_tokens.shape[-1]] = text_tokens
            text_tokens_lengths[idx] = text_tokens.shape[-1]
            
            speech = row['speech']
            speech_padded[idx, :speech.shape[-1]] = speech
            spec_lengths[idx] = row["spec_len"]

        return dict(
            text_tokens=text_tokens_padded,
            text_tokens_lengths=text_tokens_lengths,
            speech=speech_padded,
            spec_lengths=spec_lengths
        )
