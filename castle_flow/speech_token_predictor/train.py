import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR

from audiokit.text import Tokenizer
from audiokit.trainer import Trainer, main
from audiokit.utils.audio import Resampler

from castle_flow.utils import get_mel, lr_lambda_wrapper
from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.speech_token_predictor.model import SpeechTokenizer
from castle_flow.speech_token_predictor.config import STPTrainerConfig
from castle_flow.speech_token_predictor.data import AudioLoader, AudioCollate
from castle_flow.autoencoder.model import MelSpecAutoEncoder, MelSpecAutoEncoderForQRec
from castle_flow.speech_token_predictor.model import SpeechTokenPredictorForClassification


class STPTrainer(Trainer):
    def __init__(
        self,
        config: STPTrainerConfig,
        rank: int = 0,
        n_gpus: int = 1
    ) -> None:
        super().__init__(config, rank, n_gpus)
        self.config = config
        self.resampler = Resampler()
    
    def _load_autoencoder(
        self,
        autoencoder_config_path: str,
        autoencoder_ckpt_path: str
    ) -> MelSpecAutoEncoder:
        autoencoder_config: AutoEncoderTrainerConfig = AutoEncoderTrainerConfig.load_from_yaml(
            autoencoder_config_path
        )
        autoencoder_for_qrec = MelSpecAutoEncoderForQRec(autoencoder_config)
        autoencoder_for_qrec.load_state_dict(
            torch.load(
                autoencoder_ckpt_path, map_location='cpu', weights_only=True
            )["model"], strict=True
        )
        # self.logger.info(f"Autoencoder successfully loaded from {autoencoder_ckpt_path}")
        
        autoencoder = autoencoder_for_qrec.autoencoder
        autoencoder.eval()
        
        # check that downsample factor is the same
        assert autoencoder.config.downsample_factor == self.config.data.downsample_factor, \
            "downsample_factor in autoencoder and flow decoder config must match"
        
        return autoencoder
    
    def prepare_model(self) -> None:
        autoencoder = self._load_autoencoder(
            autoencoder_config_path=self.config.autoencoder_config_path,
            autoencoder_ckpt_path=self.config.autoencoder_ckpt_path
        )
        self.speech_tokenizer = SpeechTokenizer(
            autoencoder=autoencoder,
            add_bos_token=self.config.data.add_bos_token,
            add_eos_token=self.config.data.add_eos_token
        ).to(self.device)
        
        self.model = SpeechTokenPredictorForClassification(
            self.config
        ).to(self.device)
        self.model.accuracy_metrics = self.model.accuracy_metrics.to(self.device)
        assert self.config.model.vocab_size_speech >= self.speech_tokenizer.n_vocab_speech, \
            "vocab_size_speech in model config must be greater than or equal to the number of tokens in the tokenizer"
    
    def create_train_dataset(self) -> Dataset:
        self.text_tokenizer = Tokenizer(
            vocab_path=self.config.data.vocab_path,
            token_separator=self.config.data.token_separator,
            add_blank=self.config.data.add_blank
        )
        
        assert self.config.model.vocab_size_text >= self.text_tokenizer.n_symbols, \
            "vocab_size_text in model config must be greater than or equal to the number of tokens in the tokenizer"
        
        dataset = AudioLoader(
            metadata_paths=self.config.data.train_metadata_paths,
            config=self.config.data,
            tokenizer=self.text_tokenizer
        )
        return dataset
    
    # def prepare_lr_scheduler(self) -> None:
    #     self.lr_scheduler = LambdaLR(
    #         optimizer=self.optimizer,
    #         lr_lambda=lr_lambda_wrapper(self.config.total_step, self.config.optimizer.warmup_step)
    #     )
    
    def create_val_dataset(self) -> Dataset:
        pass
    
    def create_collator(self) -> callable:
        return AudioCollate()
    
    def prepare_model_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        speech = self.resampler.resample(
            batch["speech"],
            self.config.data.source_sampling_rate,
            self.config.data.sampling_rate,
            self.device
        )
        mel_spec = get_mel(speech, self.config.data)
        mel_spec_len = mel_spec.shape[-1]
        padding = (
            self.config.data.downsample_factor - mel_spec_len % self.config.data.downsample_factor
        ) % self.config.data.downsample_factor
        mel_spec = F.pad(mel_spec, (0, padding), value=np.log(1e-5))
        mel_spec = (mel_spec - self.config.data.mel_mean) / self.config.data.mel_std
        
        batch["speech_tokens"], batch["speech_tokens_lengths"] = self.speech_tokenizer.tokenize(
            mel_spec=mel_spec,
            mel_spec_lengths=batch["spec_lengths"]
        )
        
        batch.pop("speech")
        batch.pop("spec_lengths")
        
        return batch
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/speech_token_predictor.yaml',
    )
    
    args = parser.parse_args()
    config = STPTrainerConfig.load_from_yaml(args.config)
    main(config, STPTrainer)
