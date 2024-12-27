from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf
from audiokit.utils import load_config_from_yaml



@dataclass
class InferenceHParams:
    # flow_decoder params
    n_steps: int
    alpha: float
    sigma: float
    
    # prompt audio params
    chunk_size: float
    cut_random_chunk: bool
    cut_left_chunk: bool
    cut_right_chunk: bool
    
    # speech_token_predictor params
    penalty: float
    max_ar_steps: Optional[int]
    min_prob_eos: Optional[float]
    top_k: Optional[int]
    top_p: float
    temperature: float
    use_cache: bool
    
    sample_version: str

    use_stp_prompt: bool
    use_flow_decoder_prompt: bool

@dataclass
class SaveFeatures:
    spectrogram: bool
    audio: bool
    prompt_audio_processed: bool
    prompt_audio_vocoded: bool


@dataclass
class InferenceConfig:
    stp_config_path: str
    stp_checkpoint_path: str
    flow_decoder_config_path: str
    flow_decoder_checkpoint_path: str
    autoencoder_config_path: str
    autoencoder_checkpoint_path: str
    
    enable_fp16: bool
    device: str
    vocoder_path: str
    inference_relpath: str
    inference_hparams: InferenceHParams
    dataset_name: str
    seed: Optional[int]
    save_features: SaveFeatures
    language: Optional[str]
    
    asr_model_id: str
    phonemizer_service_url: str
    
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
