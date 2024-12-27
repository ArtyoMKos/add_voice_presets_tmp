import os
import random
from typing import Optional
import torch
import torchaudio
import numpy as np

from audiokit.utils.audio import Resampler
from audiokit.utils.vocoder import VocoderModel

from castle_flow import utils
from castle_flow.autoencoder.model import (
    MelSpecAutoEncoder,
    MelSpecAutoEncoderForQRec
)
from castle_flow.flow_decoder.model import FlowDecoderModelForCFM
from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.flow_decoder.config import FlowDecoderTrainerConfig
from castle_flow.flow_decoder.inference_config import InferenceConfig, InferenceHParams


class Inference:
    def __init__(
        self,
        inference_config_path: str,
    ):
        if isinstance(inference_config_path, dict):
            self.inference_config = InferenceConfig.load_from_dict(inference_config_path)
        elif isinstance(inference_config_path, str):
            if not os.path.exists(inference_config_path):
                raise ValueError(f"{inference_config_path} does not exist")
            self.inference_config = InferenceConfig.load_from_yaml(inference_config_path)
        else:
            raise ValueError(
                f"inference_config_path must be either a path to a yaml file or a dictionary, got {type(inference_config_path)}"
            )
        
        self._fix_seed()
        
        self.flow_decoder_config: FlowDecoderTrainerConfig = FlowDecoderTrainerConfig.load_from_yaml(
            self.inference_config.flow_decoder_config_path
        )
        
        self.autoencoder = self._load_autoencoder()
        self.flow_decoder = self._load_flow_decoder_torchscript()
        self.vocoder = self._load_vocoder()
        
        self.resampler = Resampler()
    
    def _fix_seed(self):
        torch.manual_seed(self.inference_config.seed)
        torch.cuda.manual_seed(self.inference_config.seed)
        np.random.seed(self.inference_config.seed)
        random.seed(self.inference_config.seed)
    
    def _load_autoencoder(
        self,
    ) -> MelSpecAutoEncoder:
        assert self.flow_decoder_config.autoencoder_config_path == self.inference_config.autoencoder_config_path, \
            "autoencoder config path in flow decoder and inference config must match"
        assert self.flow_decoder_config.autoencoder_ckpt_path == self.inference_config.autoencoder_checkpoint_path, \
            "autoencoder checkpoint path in flow decoder and inference config must match"
        autoencoder_config: AutoEncoderTrainerConfig = AutoEncoderTrainerConfig.load_from_yaml(
            self.inference_config.autoencoder_config_path
        )
        autoencoder_for_qrec = MelSpecAutoEncoderForQRec(autoencoder_config)
        autoencoder_for_qrec.load_state_dict(
            torch.load(
                self.inference_config.autoencoder_checkpoint_path, map_location='cpu', weights_only=True
            )["model"], strict=True
        )
        autoencoder = autoencoder_for_qrec.autoencoder
        autoencoder.eval()
        
        return autoencoder.to(self.inference_config.device)
    
    def _load_flow_decoder_torchscript(
        self,
    ) -> FlowDecoderModelForCFM:
        flow_decoder = FlowDecoderModelForCFM(self.flow_decoder_config)
        
        # flow_decoder.load_state_dict(
        #     torch.load(
        #         self.inference_config.flow_decoder_checkpoint_path,
        #         map_location='cpu',
        #         weights_only=True
        #     )["model"], strict=True
        # )
        
        flow_decoder.model = torch.jit.load(self.inference_config.flow_decoder_torchscript_path)
        flow_decoder.eval()
        
        return flow_decoder.to(self.inference_config.device)
    
    def _load_flow_decoder(
        self,
    ) -> FlowDecoderModelForCFM:
        flow_decoder = FlowDecoderModelForCFM(self.flow_decoder_config)
        flow_decoder.load_state_dict(
            torch.load(
                self.inference_config.flow_decoder_checkpoint_path,
                map_location='cpu',
                weights_only=True
            )["model"], strict=True
        )
        
        # flow_decoder.model = torch.jit.load("/mnt/shared_data/users/artyom/projects/castle-ai-zero-shot-tts-service/models/mel_spec_flow_decoder_v2.pt")
        flow_decoder.eval()
        
        return flow_decoder.to(self.inference_config.device)

    def _load_vocoder(self) -> Optional[VocoderModel]:
        vocoder = VocoderModel(
            model_path=self.inference_config.vocoder_path,
            hop_length=self.flow_decoder_config.data.hop_length,
            device=self.inference_config.device
        )
        print(f'Loaded Vocoder {self.inference_config.vocoder_path}')
        
        return vocoder

    def prepare_prompt_target_audios(
        self,
        prompt_audio_path: Optional[str],
        target_audio_path: str,
        inference_hparams: InferenceHParams,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        prompt_audio, prompt_sr = utils.load_audio(
            audio_path=prompt_audio_path,
            chunk_size=inference_hparams.chunk_size,
            cut_left_chunk=inference_hparams.cut_left_chunk,
            cut_right_chunk=inference_hparams.cut_right_chunk,
            cut_random_chunk=inference_hparams.cut_random_chunk,
            device=self.inference_config.device,
        ) if prompt_audio_path else (None, None)
        target_audio, target_sr = utils.load_audio(
            audio_path=target_audio_path,
            device=self.inference_config.device,
        )
        prompt_audio = self.resampler.resample(
            audio=prompt_audio,
            sr=prompt_sr,
            target_sr=self.flow_decoder_config.data.sampling_rate,
            device=self.inference_config.device,
        ) if prompt_audio is not None else None
        target_audio = self.resampler.resample(
            audio=target_audio,
            sr=target_sr,
            target_sr=self.flow_decoder_config.data.sampling_rate,
            device=self.inference_config.device,
        )
        
        return prompt_audio, target_audio
    
    def get_mel(
        self,
        audio: torch.Tensor,
        downsample_factor: int,
    ) -> torch.Tensor:
        mel = utils.get_mel(
            audio=audio.unsqueeze(0),
            config=self.flow_decoder_config.data
        )
        padding = (
            downsample_factor - mel.shape[-1] % downsample_factor
        ) % downsample_factor
        mel = torch.nn.functional.pad(mel, (0, padding), value=np.log(1e-5))
        return mel
    
    @torch.no_grad()
    def get_autoencoder_tokens(
        self,
        prompt_mel: Optional[torch.Tensor],
        target_mel: torch.Tensor
    ) -> torch.Tensor:
        prompt_token = self.autoencoder.encode(
            mel_input=prompt_mel,
            spec_len=torch.tensor([prompt_mel.shape[-1]]).to(self.inference_config.device)
        )["tokens"] if prompt_mel is not None else None
        target_token = self.autoencoder.encode(
            mel_input=target_mel,
            spec_len=torch.tensor([target_mel.shape[-1]]).to(self.inference_config.device)
        )["tokens"]
        
        if prompt_token is None:
            return target_token
        return torch.cat([prompt_token, target_token], dim=-1)
    
    def prepare_model_inputs(
        self,
        prompt_audio: Optional[torch.Tensor],
        target_audio: torch.Tensor
    ):
        prompt_mel = self.get_mel(
            audio=prompt_audio,
            downsample_factor=self.flow_decoder_config.model.downsample_factor
        ) if prompt_audio is not None else None
        target_mel = self.get_mel(
            audio=target_audio,
            downsample_factor=self.flow_decoder_config.model.downsample_factor
        )
        if prompt_mel is not None:
            prompt_mel = (prompt_mel - self.flow_decoder_config.data.mel_mean) / self.flow_decoder_config.data.mel_std
        target_mel = (target_mel - self.flow_decoder_config.data.mel_mean) / self.flow_decoder.config.data.mel_std
        
        token = self.get_autoencoder_tokens(
            prompt_mel=prompt_mel,
            target_mel=target_mel
        )
        
        if prompt_mel is not None:
            speech = torch.cat([prompt_mel, torch.zeros_like(target_mel)], dim=-1)
            prompt_len = prompt_mel.shape[-1]
            prompt_audio_vocoded = self.vocoder(
                prompt_mel * self.flow_decoder_config.data.mel_std + self.flow_decoder_config.data.mel_mean
            )
        else:
            speech = torch.zeros_like(target_mel)
            prompt_len = 0
            prompt_audio_vocoded = None
        
        speech_lengths = torch.tensor([speech.shape[-1]]).to(self.inference_config.device)
        context_mask = torch.zeros_like(speech)
        
        context_mask[..., :prompt_len] = 1.0
        
        return dict(
            token=token,
            speech=speech,
            speech_lengths=speech_lengths,
            context_mask=context_mask,
            prompt_len=prompt_len,
            prompt_audio_vocoded=prompt_audio_vocoded
        )
    
    @torch.inference_mode()
    def run(
        self,
        target_audio_path: str,
        prompt_audio_path: Optional[str] = None,
        inference_hparams: Optional[InferenceHParams] = None,
    ) -> dict:
        
        if inference_hparams is None:
            inference_hparams = self.inference_config.inference_hparams
        
        dtype = torch.float16 if self.inference_config.enable_fp16 else torch.float32
        with torch.autocast(device_type=self.inference_config.device, dtype=dtype):
            prompt_audio, target_audio = self.prepare_prompt_target_audios(
                prompt_audio_path=prompt_audio_path,
                target_audio_path=target_audio_path,
                inference_hparams=inference_hparams
            )
            model_inputs = self.prepare_model_inputs(
                prompt_audio=prompt_audio,
                target_audio=target_audio
            )
            
            output_mel = self.flow_decoder.inference(
                tokens=model_inputs["token"],
                speech=model_inputs["speech"],
                speech_lengths=model_inputs["speech_lengths"],
                context_mask=model_inputs["context_mask"],
                alpha=inference_hparams.alpha,
                n_steps=inference_hparams.n_steps,
                sigma=inference_hparams.sigma
            )
        output_mel = output_mel * self.flow_decoder_config.data.mel_std + self.flow_decoder_config.data.mel_mean
        output_mel = output_mel[..., model_inputs["prompt_len"]: ]
        audio = self.vocoder(output_mel)
        
        output = dict(
            spectrogram=output_mel,
            audio=audio.float().squeeze().cpu().numpy(),
            sample_rate=self.flow_decoder_config.data.sampling_rate,
        )
        if prompt_audio is not None:
            output["prompt_audio_vocoded"] = prompt_audio.float().squeeze().cpu().numpy()
            output["prompt_audio_processed"] = model_inputs[
                "prompt_audio_vocoded"
            ].float().squeeze().cpu().numpy()
        
        return output


if __name__ == "__main__":
    infer = Inference(
        inference_config_path='configs/inference_flow_decoder.yaml'
    )
    prompt_audio_path = "/mnt/shared_data/users/sveta/ckpts/test-clean/61/70968/61-70968-0000.wav"
    target_audio_path = "/mnt/shared_data/users/sveta/ckpts/test-clean/61/70970/61-70970-0013.flac"
    
    output = infer.run(
        prompt_audio_path=prompt_audio_path,
        # prompt_audio_path=None,
        target_audio_path=target_audio_path
    )

    torchaudio.save(
        f"output4.wav",
        torch.tensor(output["audio"]).unsqueeze(0),
        output["sample_rate"]
    )
    
