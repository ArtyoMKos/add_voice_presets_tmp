from typing import Optional
from dataclasses import dataclass, field

from omegaconf import MISSING
from audiokit.trainer import config as trainer_config
from audiokit.nn.config import DecoderConfig


@dataclass
class STPModelConfig(trainer_config.ModelConfig):
    
    vocab_size_text: int = MISSING
    vocab_size_speech: int = MISSING
    speech_pad_token: int = MISSING
    
    max_pos: int = MISSING
    
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class STPDatasetConfig(trainer_config.DatasetConfig):
    
    dataset_root: str = MISSING
    
    train_metadata_paths: list[str] = MISSING
    val_metadata_paths: Optional[list[str]] = None
    
    vocab_path: str = MISSING
    token_separator: Optional[str] = None
    add_blank: bool = False
    
    source_sampling_rate: int = 44100
    sampling_rate: int = 22050
    hop_length: int = 256
    filter_length: int = 1024
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.
    mel_fmax: float | None = 8000.
    
    downsample_factor: int = MISSING
    
    mel_mean: float = MISSING
    mel_std: float = MISSING
    
    max_frames: int = 10_000
    
    phoneme_postfix: str = ''
    audio_dir: str = MISSING
    phoneme_dir: str = MISSING
    
    add_bos_token: bool = True
    add_eos_token: bool = True

    concat_prob: float = 0.4


@dataclass
class STPTrainerConfig(trainer_config.TrainerConfig):
    
    autoencoder_config_path: str = MISSING
    autoencoder_ckpt_path: str = MISSING
    
    top_k_accuracies: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 40])
    
    model: STPModelConfig = field(default_factory=STPModelConfig)
    data: STPDatasetConfig = field(default_factory=STPDatasetConfig)


if __name__ == "__main__":
    stp_config = STPTrainerConfig()
    stp_config.save_to_yaml("configs/speech_token_predictor1.yaml")
