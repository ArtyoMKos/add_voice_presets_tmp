from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
import torchdiffeq
from audiokit.cfm.conditional_flow_matching import ConditionalFlowMatcher
from audiokit.nn.encoder import EncoderWithSkipConnectionLayers
from audiokit.nn.output import ModelOutputForTraining
from audiokit.nn.utils import get_sequence_mask
from torch import nn

from castle_flow import modules
from castle_flow.autoencoder.config import AutoEncoderTrainerConfig
from castle_flow.autoencoder.model import MelSpecAutoEncoderForQRec
from castle_flow.flow_decoder.config import (FlowDecoderModelConfig,
                                             FlowDecoderTrainerConfig)


@dataclass
class FlowDecoderModelOutput:
    vector_field: torch.Tensor
    mask: torch.Tensor
    

@dataclass
class FlowDecoderForCFMOutput(ModelOutputForTraining):
    vector_field: torch.Tensor
    mask: torch.Tensor
    target_vector_field: torch.Tensor
    loss_dict: dict[str, torch.FloatTensor]


class FlowDecoderTransformer(nn.Module):
    def __init__(self, config: FlowDecoderModelConfig) -> None:
        super().__init__()
        
        fusion_input_sizes = [config.n_mels, config.n_mels, config.decoder.hidden_size]
        self.fusion = modules.FusionLayer(
            hidden_size=config.decoder.hidden_size,
            input_sizes=fusion_input_sizes
        )
        self.decoder = EncoderWithSkipConnectionLayers(config.decoder)
        self.proj = nn.Conv1d(config.decoder.hidden_size, config.decoder.hidden_size, 1)
        
    def forward(
        self,
        tokens: torch.Tensor, # [b, n]
        speech: torch.Tensor, # [b, f, n]
        noisy_speech: torch.Tensor, # [b, f, n] sample at flow step t
        flow_step: torch.Tensor, # [b]
        mask: torch.Tensor # [b, f, n]
    ) -> torch.Tensor:
        embedding = self.fusion([noisy_speech, speech, tokens], mask) # [b, h, n]
        
        embedding = torch.cat([flow_step, embedding], dim=-1) # [b, h, n + 1]
        mask_flow_step = torch.ones((mask.shape[0], 1, 1)).to(mask.device) # [b, 1, 1]
        mask_extended = torch.cat([mask_flow_step, mask], dim=-1) # [b, 1, n + 1]
        
        hidden_states = self.decoder(embedding, mask_extended)
        
        # remove the flow step hidden state
        hidden_states = hidden_states[..., 1:] # [b, h, n]
        hidden_states = self.proj(hidden_states) * mask
        
        return hidden_states


class FlowDecoderModel(nn.Module):
    def __init__(self, config: FlowDecoderModelConfig) -> None:
        super().__init__()
        
        self.config = config
        self.use_classifier_free_guidance = config.use_classifier_free_guidance
        self.keep_probability = 1 - config.conditional_dropout
        
        self.tok_embedding = nn.Embedding(config.codebook_size, config.decoder.hidden_size)
        self.flow_step_embedding = modules.FlowStepEmbedding(config.decoder.hidden_size)
        
        self.transformer = FlowDecoderTransformer(config)
        
        self.vector_field = nn.Conv1d(config.decoder.hidden_size, config.n_mels, 1)
        self.upsample_factor = config.downsample_factor
        
    def forward(
        self,
        tokens: torch.Tensor, # [b, n]
        speech: torch.Tensor, # [b, f, n]
        speech_lengths: torch.Tensor, # [b]
        noisy_speech: torch.Tensor, # [b, f, n] sample at flow step t
        flow_step: torch.Tensor, # [b]
        context_mask: torch.Tensor # [b, f, n]
    ) -> FlowDecoderModelOutput:
        
        tokens_embedding: torch.Tensor = self.tok_embedding(tokens)
        tokens_embedding = tokens_embedding.permute(0, 2, 1) # [b, h, n]
        tokens_embedding = tokens_embedding.repeat_interleave(
            self.upsample_factor,
            dim=-1
        )
        
        mask = get_sequence_mask(
            lengths=speech_lengths,
            max_length=speech.shape[-1],
            dtype=speech.dtype
        ).unsqueeze(1) # [b, 1, n]
        
        # leave only the prompt part of the speech
        speech = speech * context_mask
        flow_step_embedding: torch.Tensor = self.flow_step_embedding(flow_step)
        flow_step_embedding = flow_step_embedding.unsqueeze(-1) # [b, h, 1]
        
        if self.use_classifier_free_guidance:
            keep_mask = torch.bernoulli(
                torch.zeros(speech.shape[0]),
                self.keep_probability
            ).to(speech.device)
            speech = keep_mask[:, None, None] * speech
            tokens_embedding = keep_mask[:, None, None] * tokens_embedding
        
        hidden_states = self.transformer(
            tokens=tokens_embedding,
            speech=speech,
            noisy_speech=noisy_speech,
            flow_step=flow_step_embedding,
            mask=mask
        )
        vector_field = self.vector_field(hidden_states) * mask # [b, f, n]
        
        return FlowDecoderModelOutput(
            vector_field=vector_field,
            mask=mask
        )
    
    def inference(
        self,
        tokens: torch.Tensor, # [b, n]
        speech: torch.Tensor, # [b, f, n]
        speech_lengths: torch.Tensor, # [b]
        noisy_speech: torch.Tensor, # [b, f, n] sample at flow step t
        flow_step: torch.Tensor, # [b]
        context_mask: torch.Tensor, # [b, f, n]
        alpha: float = 0.0
    ) -> FlowDecoderModelOutput:
        
        tokens_embedding: torch.Tensor = self.tok_embedding(tokens)
        tokens_embedding = tokens_embedding.permute(0, 2, 1) # [b, h, n]
        tokens_embedding = tokens_embedding.repeat_interleave(
            self.upsample_factor,
            dim=-1
        )
        
        mask = get_sequence_mask(
            lengths=speech_lengths,
            max_length=speech.shape[-1],
            dtype=speech.dtype
        ).unsqueeze(1) # [b, 1, n]
        
        # leave only the prompt part of the speech
        speech = speech * context_mask
        
        flow_step_embedding: torch.Tensor = self.flow_step_embedding(flow_step)
        flow_step_embedding = flow_step_embedding.unsqueeze(-1) # [b, h, 1]
        
        hidden_states = self.transformer(
            tokens=tokens_embedding,
            speech=speech,
            noisy_speech=noisy_speech,
            flow_step=flow_step_embedding,
            mask=mask
        )
        vector_field = self.vector_field(hidden_states) * mask # [b, f, n]
        
        if self.use_classifier_free_guidance and alpha:
            hidden_states_zero_cond = self.transformer(
                tokens=torch.zeros_like(tokens_embedding),
                speech=torch.zeros_like(speech),
                noisy_speech=noisy_speech,
                flow_step=flow_step_embedding,
                mask=mask
            )
            hidden_states_zero_cond = self.vector_field(hidden_states_zero_cond) * mask
            vector_field = (1 + alpha) * vector_field - alpha * hidden_states_zero_cond
        
        return FlowDecoderModelOutput(
            vector_field=vector_field,
            mask=mask
        )


class FlowDecoderModelForCFM(nn.Module):
    def __init__(self, config: FlowDecoderTrainerConfig) -> None:
        super().__init__()
        
        self.config = config
        self.model = FlowDecoderModel(config.model)
        self.flow_matcher = ConditionalFlowMatcher(
            sigma=config.flow_matcher_sigma,
            independent=config.flow_matcher_independent
        )

    def forward(
        self,
        tokens: torch.Tensor, # [b, n]
        mel: torch.Tensor, # [b, f, n]
        spec_len: torch.Tensor, # [b]
        context_mask: torch.Tensor # [b, f, n]
    ) -> FlowDecoderForCFMOutput:
        
        loss_dict = {}
        flow_step, noisy_speech, target_vector_field = self.flow_matcher.sample_location_and_conditional_flow(
            x0=torch.randn_like(mel),
            x1=mel
        )
        output: FlowDecoderModelOutput = self.model(
            tokens=tokens,
            speech=mel,
            speech_lengths=spec_len,
            noisy_speech=noisy_speech,
            flow_step=flow_step,
            context_mask=context_mask
        )
        vector_field, mask = output.vector_field, output.mask
        
        if self.config.use_loss_masking:
            mask = mask * (1 - context_mask)
            
        if mask.shape != vector_field.shape:
            mask = mask.expand(vector_field.shape) # [b, 1, n] -> [b, f, n]
        
        loss_dict['flow'] = F.mse_loss(
            input=vector_field * mask,
            target=target_vector_field * mask,
            reduction='sum'
        ) / mask.sum()
        
        loss_dict['total'] = loss_dict['flow']
        
        return FlowDecoderForCFMOutput(
            loss_dict=loss_dict,
            vector_field=output.vector_field,
            mask=output.mask,
            target_vector_field=target_vector_field
        )

    def inference(
        self,
        tokens: torch.Tensor, # [b, n]
        speech: torch.Tensor, # [b, f, n]
        speech_lengths: torch.Tensor, # [b]
        context_mask: torch.Tensor, # [b, f, n]
        alpha: float = 0.0,
        n_steps: int = 16,
        sigma: float = 1.0
    ) -> torch.Tensor: # [b, f, n]
        
        with torch.no_grad():
            speech_predicted = torchdiffeq.odeint(
                lambda t, w: self.model.inference(
                    tokens=tokens,
                    speech=speech,
                    speech_lengths=speech_lengths,
                    noisy_speech=w,
                    flow_step=t.repeat(speech.shape[0]) if t.dim() == 0 else t,
                    context_mask=context_mask,
                    alpha=alpha
                # ).vector_field,
                )["vector_field"],
                sigma * torch.randn(speech.shape).to(speech.device),
                torch.linspace(0, 1, 2).to(speech.device),
                method="midpoint",
                options=dict(step_size=1 / n_steps)
            )
        speech_predicted = speech_predicted[-1]
        
        return speech_predicted
