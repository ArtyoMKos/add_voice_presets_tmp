from dataclasses import dataclass, field
from typing import Optional

from audiokit.nn.config import EncoderConfig
from audiokit.trainer import config as trainer_config
from omegaconf import MISSING


@dataclass
class FlowDecoderModelConfig(trainer_config.ModelConfig):
    
    codebook_size: int = MISSING
    downsample_factor: int = MISSING
    
    n_mels: int = MISSING
    
    use_classifier_free_guidance: bool = MISSING
    conditional_dropout: float = MISSING
    
    decoder: EncoderConfig = field(default_factory=EncoderConfig)

@dataclass
class FlowDecoderDatasetConfig(trainer_config.DatasetConfig):
    
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
    
    context_mask_drop_prob: float = 0.3
    context_mask_min_ratio: float = 0.7
    context_mask_max_ratio: float = 1.0
    
    max_duration: float = 3.0 # seconds
    vsr_transform: bool = False


@dataclass
class FlowDecoderTrainerConfig(trainer_config.TrainerConfig):
    
    flow_matcher_sigma: float = 0.0
    flow_matcher_independent: bool = False
    use_loss_masking: bool = True
    
    autoencoder_config_path: str = MISSING
    autoencoder_ckpt_path: str = MISSING
    
    model: FlowDecoderModelConfig = field(default_factory=FlowDecoderModelConfig)
    data: FlowDecoderDatasetConfig = field(default_factory=FlowDecoderDatasetConfig)
    loss_weights: dict[str, float] = field(default_factory=lambda: {})
    
    def __post_init__(self):
        assert self.data.n_mels == self.model.n_mels, \
            "n_mels in model and data config must match"
        
        assert self.data.downsample_factor == self.model.downsample_factor, \
            "downsample_factor in model and data config must match"


if __name__ == "__main__":
    flow_decoder_config = FlowDecoderTrainerConfig()
    flow_decoder_config.save_to_yaml("configs/flow_decoder.yaml")
