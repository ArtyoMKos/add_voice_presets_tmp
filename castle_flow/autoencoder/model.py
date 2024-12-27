from dataclasses import dataclass

import torch
import torch.nn as nn
from audiokit.nn.utils import get_sequence_mask
from audiokit.nn.encoder import EncoderWithSkipConnectionLayers

from castle_flow import modules
from castle_flow.autoencoder.config import (
    AutoEncoderModelConfig,
    AutoEncoderTrainerConfig  
)


@dataclass
class MelSpecAutoEncoderOutput:
    mel_pred: torch.FloatTensor
    mel_pred_mask: torch.FloatTensor
    encoded_mel: torch.FloatTensor
    quantized_mel: torch.FloatTensor
    encoded_mel_mask: torch.FloatTensor


@dataclass
class MelSpecAutoEncoderQRecOutput:
    output: MelSpecAutoEncoderOutput
    loss_dict: dict[str, torch.FloatTensor]


class MelSpecEncoder(nn.Module):
    def __init__(self, config: AutoEncoderModelConfig) -> None:
        super().__init__()
        
        fusion_input_sizes = [config.n_mels]
        self.fusion = modules.FusionLayer(
            hidden_size=config.encoder.hidden_size,
            input_sizes=fusion_input_sizes
        )
        
        self.encoder = EncoderWithSkipConnectionLayers(config.encoder)
        self.proj = nn.Conv1d(config.encoder.hidden_size, config.encoder.hidden_size, 1)

        self.down = nn.AvgPool1d(
            kernel_size=config.downsample_factor,
            stride=config.downsample_factor
        )
        self.out = nn.Conv1d(config.encoder.hidden_size, 2 * config.latent_dim, 1)
        self.downsample_factor = config.downsample_factor
        
    def forward(
        self,
        speech: torch.Tensor, # [b, f, n]
        speech_lengths: torch.Tensor, # [b]
    ) -> torch.Tensor: # [b, h, n]
        
        mask = get_sequence_mask(
            lengths=speech_lengths,
            max_length=speech.shape[-1],
            dtype=speech.dtype
        ).unsqueeze(1) # [b, 1, n]
        
        embedding = self.fusion([speech], mask) # [b, h, n]
        hidden_states = self.encoder(embedding, mask)
        hidden_states = self.proj(hidden_states) * mask
        
        hidden_state_down = self.down(hidden_states)
        mask = mask[:, :, ::self.downsample_factor]
        enc_out = self.out(hidden_state_down) * mask
        enc_out, _ = enc_out.chunk(2, dim=1)
        
        return enc_out, mask


class MelSpecDecoder(nn.Module):
    def __init__(self, config: AutoEncoderModelConfig) -> None:
        super().__init__()
        
        fusion_input_sizes = [config.latent_dim]
        self.fusion = modules.FusionLayer(
            hidden_size=config.decoder.hidden_size,
            input_sizes=fusion_input_sizes
        )
        
        self.decoder = EncoderWithSkipConnectionLayers(config.decoder)
        self.proj = nn.Conv1d(config.decoder.hidden_size, config.decoder.hidden_size, 1)
        
        self.out = nn.Conv1d(config.decoder.hidden_size, config.n_mels, 1)
        self.upsample_factor = config.downsample_factor

    def forward(
        self,
        latents: torch.Tensor,
        latents_lengths: torch.Tensor,
        sp_emb: torch.Tensor,
    ) -> torch.Tensor:
        latents = latents.repeat_interleave(
            self.upsample_factor, 
            dim=-1
        )
        mask = get_sequence_mask(
            lengths=latents_lengths * self.upsample_factor,
            max_length=latents.shape[-1],
            dtype=latents_lengths.dtype
        ).unsqueeze(1) # [b, 1, n]
        
        embedding = self.fusion([latents], mask)
        embedding = torch.cat([sp_emb.unsqueeze(-1), embedding], dim=-1)
        mask_sp_emb = torch.ones((mask.shape[0], 1, 1)).to(mask.device)
        mask_extended = torch.cat([mask_sp_emb, mask], dim=-1)
        
        hidden_states = self.decoder(embedding, mask_extended)
        hidden_states = hidden_states[..., 1:]
        hidden_states = self.proj(hidden_states) * mask
        
        mel_hat = self.out(hidden_states) * mask
        
        return mel_hat, mask


class SpeakerEncoder(nn.Module):
    def __init__(self, config: AutoEncoderModelConfig) -> None:
        super().__init__()
        
        fusion_input_sizes = [config.n_mels]
        self.fusion = modules.FusionLayer(
            hidden_size=config.speaker_encoder.hidden_size,
            input_sizes=fusion_input_sizes
        )
        
        self.speaker_encoder = EncoderWithSkipConnectionLayers(config.speaker_encoder)
        self.proj = nn.Conv1d(config.speaker_encoder.hidden_size, config.speaker_encoder.hidden_size, 1)
        
        self.out = nn.Conv1d(config.speaker_encoder.hidden_size, config.speaker_encoder.hidden_size, 1)
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        mask = get_sequence_mask(
            lengths=speech_lengths,
            max_length=speech.shape[-1],
            dtype=speech.dtype
        ).unsqueeze(1)

        embedding = self.fusion([speech], mask)
        hidden_states = self.speaker_encoder(embedding, mask)
        hidden_states: torch.FloatTensor = self.proj(hidden_states) * mask
        hidden_states = hidden_states.sum(dim=-1) / mask.sum(dim=-1)
        hidden_states = hidden_states.unsqueeze(-1)

        sp_emb = self.out(hidden_states).squeeze(-1)

        return sp_emb


class MelSpecAutoEncoder(nn.Module):
    def __init__(self, config: AutoEncoderModelConfig) -> None:
        super().__init__()
        
        self.config = config
        self.encoder = MelSpecEncoder(config)
        self.decoder = MelSpecDecoder(config)
        self.speaker_encoder = SpeakerEncoder(config)
        self.quantizer = modules.VectorQuantize(
            config.latent_dim,
            config.codebook_size,
            config.codebook_dim,
            low_dim_project=config.low_dim_project
        )
    
    def forward(
        self,
        mel_input: torch.Tensor,
        mel_target: torch.Tensor,
        spec_len: torch.LongTensor,
    ) -> MelSpecAutoEncoderOutput:
        encoded_mel, encoded_mel_mask = self.encoder(mel_input, spec_len)
        speaker_emb = self.speaker_encoder(mel_target, spec_len)
        encoded_mel_quantized = self.quantizer(encoded_mel, encoded_mel_mask)
        mel_pred, mel_pred_mask = self.decoder(
            latents=encoded_mel_quantized['z_q_out'],
            latents_lengths=spec_len // self.encoder.downsample_factor,
            sp_emb=speaker_emb
        )
        
        return MelSpecAutoEncoderOutput(
            mel_pred=mel_pred,
            mel_pred_mask=mel_pred_mask,
            encoded_mel=encoded_mel_quantized["z_e"],
            quantized_mel=encoded_mel_quantized["z_q"],
            encoded_mel_mask=encoded_mel_mask
        )
    
    def encode(
        self,
        mel_input,
        spec_len
    ) -> dict[str, torch.Tensor]:
        encoded_mel, encoded_mel_mask = self.encoder(mel_input, spec_len)
        encoded_mel_quantized = self.quantizer(encoded_mel, encoded_mel_mask)
        
        return dict(
            encoded_mel=encoded_mel,
            quantized_mel=encoded_mel_quantized["z_q_out"],
            tokens=encoded_mel_quantized["indices"],
            encoded_mel_mask=encoded_mel_mask
        )
    
    def decode(
        self,
        quantized_mel,
        mel_input,
        spec_len
    ) -> torch.Tensor:
        speaker_emb = self.speaker_encoder(mel_input, spec_len)
        mel_pred, mel_pred_mask = self.decoder(
            latents=quantized_mel,
            latents_lengths=spec_len,
            sp_emb=speaker_emb
        )
        return mel_pred


class MelSpecAutoEncoderForQRec(nn.Module):
    def __init__(self, config: AutoEncoderTrainerConfig) -> None:
        super().__init__()
        
        self.config = config
        self.autoencoder = MelSpecAutoEncoder(config.model)
    
    def forward(
        self,
        mel_input: torch.Tensor,
        mel_target: torch.Tensor,
        spec_len: torch.LongTensor,
    ):
        loss_dict = {}
        
        output: MelSpecAutoEncoderOutput = self.autoencoder(
            mel_input=mel_input,
            mel_target=mel_target,
            spec_len=spec_len
        )
        
        if output.mel_pred_mask.shape != output.mel_pred.shape:
            output.mel_pred_mask = output.mel_pred_mask.expand(
                output.mel_pred.shape
            ) # [b, 1, n] -> [b, f, n]
        
        if output.encoded_mel_mask.shape != output.encoded_mel.shape:
            output.encoded_mel_mask = output.encoded_mel_mask.expand(
                output.encoded_mel.shape
            ) # [b, 1, n] -> [b, f, n]
        
        loss_dict["reconstruction"] = nn.functional.mse_loss(
            output.mel_pred * output.mel_pred_mask,
            mel_target * output.mel_pred_mask,
            reduction="sum"
        ) / output.mel_pred_mask.sum()
        loss_dict["codebook"] = nn.functional.mse_loss(
            output.quantized_mel * output.encoded_mel_mask,
            output.encoded_mel.detach() * output.encoded_mel_mask,
            reduction="sum"
        ) / output.encoded_mel_mask.sum()
        loss_dict["commitment"] = nn.functional.mse_loss(
            output.encoded_mel * output.encoded_mel_mask,
            output.quantized_mel.detach() * output.encoded_mel_mask,
            reduction="sum"
        ) / output.encoded_mel_mask.sum()
        
        return MelSpecAutoEncoderQRecOutput(
            output=output,
            loss_dict=loss_dict
        )
