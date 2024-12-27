import argparse

import numpy as np
import torch
import torch.nn.functional as F
from audiokit.trainer import Trainer, main
from audiokit.utils.audio import Resampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset

from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.autoencoder.data import AudioCollate, AudioLoader
from castle_flow.autoencoder.model import MelSpecAutoEncoderForQRec
from castle_flow.utils import get_mel, lr_lambda_wrapper, vsr_transform_batch


class AutoEncoderTrainer(Trainer):
    def __init__(
        self,
        config: AutoEncoderTrainerConfig,
        rank: int = 0,
        n_gpus: int = 1
    ) -> None:
        super().__init__(config, rank, n_gpus)
        self.config = config
        self.resampler = Resampler()
    
    def prepare_model(self) -> None:
        self.model = MelSpecAutoEncoderForQRec(self.config).to(self.device)
    
    def create_train_dataset(self) -> Dataset:
        dataset = AudioLoader(
            metadata_paths=self.config.data.train_metadata_paths,
            config=self.config.data,
        )
        return dataset
    
    def create_val_dataset(self) -> Dataset:
        pass

    def create_collator(self) -> callable:
        return AudioCollate()

    def prepare_lr_scheduler(self) -> None:
        self.lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lr_lambda_wrapper(
                self.config.total_step, self.config.optimizer.warmup_step
            )
        )

    def prepare_model_inputs(self, batch: dict) -> dict:
        audio_padded = torch.nn.utils.rnn.pad_sequence(
            batch["audio"], batch_first=True, padding_value=0
        )
        audio = self.resampler.resample(
            audio_padded,
            self.config.data.source_sampling_rate,
            self.config.data.sampling_rate,
            self.device
        )

        mel_spec = get_mel(audio, self.config.data)
        mel_spec_len = mel_spec.shape[-1]
        padding = (
            self.config.data.downsample_factor - mel_spec_len % self.config.data.downsample_factor
        ) % self.config.data.downsample_factor
        mel_spec = F.pad(mel_spec, (0, padding), value=np.log(1e-5))
        batch["mel_target"] = mel_spec

        if self.config.data.vsr_transform:
            mel_spec = vsr_transform_batch(mel_spec, batch["spec_len"])
        
        batch["mel_input"] = mel_spec
        batch["spec_len"] = batch["spec_len"].to(self.device)
        batch["mel_target"] = (batch["mel_target"] - self.config.data.mel_mean) / self.config.data.mel_std
        batch["mel_input"] = (batch["mel_input"] - self.config.data.mel_mean) / self.config.data.mel_std
        batch.pop("audio")
        
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/autoencoder.yaml',
    )
    
    args = parser.parse_args()
    config = AutoEncoderTrainerConfig.load_from_yaml(args.config)
    main(config, AutoEncoderTrainer)
