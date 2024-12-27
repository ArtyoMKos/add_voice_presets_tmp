from dataclasses import dataclass, field
from typing import Optional

from audiokit.nn.config import EncoderConfig
from audiokit.trainer import config as trainer_config
from omegaconf import MISSING


@dataclass
class AutoEncoderModelConfig(trainer_config.ModelConfig):
    
    n_mels: int = 80
    downsample_factor: int = MISSING
    latent_dim: int = MISSING
    
    codebook_size: int = MISSING
    codebook_dim: int = MISSING
    low_dim_project: bool = False
    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: EncoderConfig = field(default_factory=EncoderConfig)
    speaker_encoder: EncoderConfig = field(default_factory=EncoderConfig)

@dataclass
class AutoEncoderDatasetConfig(trainer_config.DatasetConfig):
    
    dataset_root: str = MISSING
    
    train_metadata_paths: list[str] = MISSING
    val_metadata_paths: Optional[list[str]] = None
    
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
    
    max_duration: float = 3.0 # seconds
    vsr_transform: bool = False


@dataclass
class AutoEncoderTrainerConfig(trainer_config.TrainerConfig):
    
    model: AutoEncoderModelConfig = field(default_factory=AutoEncoderModelConfig)
    data: AutoEncoderDatasetConfig = field(default_factory=AutoEncoderDatasetConfig)
    loss_weights: dict[str, float] = field(default_factory=lambda: {})
    
    def __post_init__(self):
        assert self.data.n_mels == self.model.n_mels, \
            "n_mels in model and data config must match"
        
        assert self.data.downsample_factor == self.model.downsample_factor, \
            "downsample_factor in model and data config must match"


if __name__ == "__main__":
    autoencoder_config = AutoEncoderTrainerConfig()
    autoencoder_config.save_to_yaml("configs/autoencoder1.yaml")
