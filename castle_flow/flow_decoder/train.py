import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from audiokit.trainer import Trainer, main
from audiokit.utils.audio import Resampler
from torch.optim.lr_scheduler import LambdaLR

from castle_flow.flow_decoder.model import FlowDecoderModelForCFM
from castle_flow.flow_decoder.data import AudioCollate, AudioLoader
from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.flow_decoder.config import FlowDecoderTrainerConfig
from castle_flow.utils import get_mel, lr_lambda_wrapper, vsr_transform_batch
from castle_flow.autoencoder.model import MelSpecAutoEncoderForQRec, MelSpecAutoEncoder


class FlowDecoderTrainer(Trainer):
    def __init__(
        self,
        config: FlowDecoderTrainerConfig,
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
        self.logger.info(f"Autoencoder successfully loaded from {autoencoder_ckpt_path}")
        
        autoencoder = autoencoder_for_qrec.autoencoder
        autoencoder.eval()
        
        # check that downsample factor is the same
        assert autoencoder.config.downsample_factor == self.config.model.downsample_factor, \
            "downsample_factor in autoencoder and flow decoder config must match"
        
        return autoencoder
    
    def prepare_model(self) -> None:
        self.autoencoder = self._load_autoencoder(
            autoencoder_config_path=self.config.autoencoder_config_path,
            autoencoder_ckpt_path=self.config.autoencoder_ckpt_path
        ).to(self.device)
        
        self.model = FlowDecoderModelForCFM(self.config).to(self.device)
    
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

    def prepare_model_inputs(self, batch: dict[str, torch.Tensor]) -> dict:
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
        batch["mel"] = mel_spec

        if self.config.data.vsr_transform:
            mel_spec = vsr_transform_batch(mel_spec, batch["spec_len"])
        
        batch["mel_vsr"] = mel_spec
        batch["spec_len"] = batch["spec_len"].to(self.device)
        batch["mel"] = (batch["mel"] - self.config.data.mel_mean) / self.config.data.mel_std
        batch["mel_vsr"] = (batch["mel_vsr"] - self.config.data.mel_mean) / self.config.data.mel_std
        
        with torch.no_grad():
            batch["tokens"] = self.autoencoder.encode(
                mel_input=batch["mel_vsr"],
                spec_len=batch["spec_len"]
            )["tokens"]
        
        batch.pop("mel_vsr")
        batch.pop("audio")
        
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/flow_decoder.yaml',
    )
    
    args = parser.parse_args()
    config = FlowDecoderTrainerConfig.load_from_yaml(args.config)
    main(config, FlowDecoderTrainer)
