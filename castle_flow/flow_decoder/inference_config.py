from dataclasses import dataclass

from omegaconf import OmegaConf
from audiokit.utils import load_config_from_yaml


@dataclass
class InferenceHParams:
    n_steps: int
    alpha: float
    sigma: float
    chunk_size: float
    cut_random_chunk: bool
    cut_left_chunk: bool
    cut_right_chunk: bool


@dataclass
class SaveFeatures:
    spectrogram: bool
    audio: bool
    prompt_audio_processed: bool
    prompt_audio_vocoded: bool


@dataclass
class InferenceConfig:
    flow_decoder_config_path: str
    flow_decoder_checkpoint_path: str
    enable_fp16: bool
    device: str
    vocoder_path: str
    autoencoder_config_path: str
    autoencoder_checkpoint_path: str
    inference_relpath: str
    inference_hparams: InferenceHParams
    dataset_name: str
    save_features: SaveFeatures
    seed: int
    
    inference_type: str
    
    @classmethod
    def load_from_yaml(cls, path: str) -> 'InferenceConfig':
        return load_config_from_yaml(path, cls)
    
    @classmethod
    def load_from_dict(cls, config_dict: dict) -> 'InferenceConfig':
        base_conf = OmegaConf.structured(cls)
        file_conf = OmegaConf.create(config_dict)
        merged_conf = OmegaConf.merge(base_conf, file_conf)
        return OmegaConf.to_object(merged_conf)
