from dataclasses import dataclass
from typing import Optional

import torch

from torch import nn
from audiokit.nn.decoder import Decoder
from audiokit.nn.utils import get_sequence_mask
from torchmetrics.classification import MulticlassAccuracy

from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
)


from castle_flow.autoencoder.model import MelSpecAutoEncoder
from castle_flow.speech_token_predictor.config import STPModelConfig, STPTrainerConfig
from castle_flow.speech_token_predictor.sampling_utils import EOSTokenMinProbLogitsProcessor


@dataclass
class SpeechTokenPredictorForClassificationOutput:
    loss_dict: dict[str, torch.Tensor]
    logits: torch.Tensor
    logits_mask: torch.Tensor


@dataclass
class SpeechTokenPredictorSampleInputs:
    text_tokens: torch.Tensor
    speech_tokens: torch.Tensor
    text_lengths: torch.Tensor
    speech_token_lengths: torch.Tensor
    logits_processor: Optional[LogitsProcessorList] = None
    logits_warper: Optional[LogitsProcessorList] = None
    max_ar_steps: int = 2000
    eos_token: int = 0
    top_k: Optional[int] = 5
    temperature: float = 1.0
    penalty: float = 1.0
    top_p: float = 0.9
    use_cache: bool = False
    min_prob_eos: float = 0.9


class SpeechTokenizer(nn.Module):

    def __init__(
        self, 
        autoencoder: MelSpecAutoEncoder,
        add_bos_token=False,
        add_eos_token=False,
    ):
        super().__init__()
       
        self.autoencoder = autoencoder
        self.n_vocab_speech = self.autoencoder.config.codebook_size

        if add_bos_token:
            self.bos_token = self.n_vocab_speech
            self.n_vocab_speech += 1
        if add_eos_token:
            self.eos_token = self.n_vocab_speech
            self.n_vocab_speech += 1

        self.pad_token = self.n_vocab_speech
        self.n_vocab_speech += 1
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    @torch.no_grad()
    def tokenize(
        self,
        mel_spec: torch.Tensor,
        mel_spec_lengths: torch.LongTensor,
    ):
        speech_tokens = self.autoencoder.encode(
            mel_input=mel_spec,
            spec_len=mel_spec_lengths
        )["tokens"]
        speech_tokens_lengths = mel_spec_lengths // self.autoencoder.config.downsample_factor
        if self.add_eos_token:
            speech_tokens = torch.cat(
                [speech_tokens, torch.ones_like(speech_tokens[:, :1]) * self.pad_token],
                dim=-1
            )
            for i in range(speech_tokens.size(0)):
                speech_tokens[i, speech_tokens_lengths[i].item()] = self.eos_token
                speech_tokens[i, speech_tokens_lengths[i].item() + 1:] = self.pad_token
        if self.add_bos_token:
            speech_tokens = torch.cat(
                [torch.ones_like(speech_tokens[:, :1]) * self.bos_token, speech_tokens], dim=-1
            )

        if self.add_eos_token:
            speech_tokens_lengths += 1
        if self.add_bos_token:
            speech_tokens_lengths += 1

        return speech_tokens.contiguous(), speech_tokens_lengths
    
class SpeechTokenPredictorWrapper(nn.Module):
    def __init__(self, config: STPModelConfig, model=None):
        super().__init__()
        
        self.config = config
        
        self.emb_speech, self.emb_text_tokens = self._prepare_embeddings()
        self.dec = Decoder(self.config.decoder)
        self.proj = nn.Linear(
            self.config.decoder.hidden_size,
            self.config.decoder.hidden_size,
        )
        if not self.config.decoder.use_rope:
            self.pos_emb_speech, self.pos_emb_text_tokens = self._prepare_rel_pos()
        
        self.classifier = nn.Linear(
            self.config.decoder.hidden_size,
            self.config.vocab_size_speech
        )
        self._model = model
    
    def _prepare_embeddings(self) -> tuple[nn.Embedding, nn.Embedding]:
        emb_speech = nn.Embedding(
            self.config.vocab_size_speech, 
            self.config.decoder.hidden_size, 
            padding_idx=self.config.speech_pad_token
        )
        emb_text_tokens = nn.Embedding(
            self.config.vocab_size_text,
            self.config.decoder.hidden_size,
        )
        
        return emb_speech, emb_text_tokens
    
    def _prepare_rel_pos(self) -> tuple[nn.Embedding, nn.Embedding]:
        pos_emb_speech = nn.Embedding(
            self.config.max_pos + 1,
            self.config.decoder.hidden_size,
            padding_idx=0
        )
        pos_emb_text_tokens = nn.Embedding(
            self.config.max_pos + 1,
            self.config.decoder.hidden_size,
            padding_idx=0
        )
        
        return pos_emb_speech, pos_emb_text_tokens
    
    def forward(
        self,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        text_tokens_lengths: torch.LongTensor,
        speech_tokens_lengths: torch.LongTensor,
        use_cache = False,
        past_key_values = None
    ) -> list[torch.Tensor]:
        results = self._model(
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            text_tokens_lengths=text_tokens_lengths,
            speech_tokens_lengths=speech_tokens_lengths,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        
        return results
    
    def prepare_processor_and_warper(
        self,
        inputs: SpeechTokenPredictorSampleInputs,
    ):
        if inputs.logits_processor is None:
            inputs.logits_processor = LogitsProcessorList()
            if inputs.penalty != 1.0:
                inputs.logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(penalty=inputs.penalty)
                )
            if inputs.max_ar_steps is not None:
                inputs.logits_processor.append(
                    ForcedEOSTokenLogitsProcessor(
                        max_length=inputs.max_ar_steps,
                        eos_token_id=inputs.eos_token
                    )
                )
            if inputs.min_prob_eos is not None:
                inputs.logits_processor.append(
                    EOSTokenMinProbLogitsProcessor(
                        eos_token_id=inputs.eos_token,
                        min_eos_p=inputs.min_prob_eos
                    )
                )

        if inputs.logits_warper is None:
            inputs.logits_warper = LogitsProcessorList()
            if inputs.temperature != 1.0:
                inputs.logits_warper.append(TemperatureLogitsWarper(inputs.temperature))
            if inputs.top_k is not None and inputs.top_k > 0:
                inputs.logits_warper.append(TopKLogitsWarper(top_k=inputs.top_k))
            if inputs.top_p is not None and inputs.top_p < 1.0:
                inputs.logits_warper.append(TopPLogitsWarper(top_p=inputs.top_p))
        
        return inputs
    
    def sample(
        self, inputs: SpeechTokenPredictorSampleInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        inputs = self.prepare_processor_and_warper(inputs)
        speech_token_ids = inputs.speech_tokens.long().clone()
        speech_token_probs = []
        past_key_values = None

        while True:
            model_inputs = {
                'text_tokens': inputs.text_tokens,
                'speech_tokens': speech_token_ids,
                'text_tokens_lengths': inputs.text_lengths,
                'speech_tokens_lengths': inputs.speech_token_lengths,
            }
            if inputs.use_cache:
                model_inputs['past_key_values'] = past_key_values
                model_inputs['use_cache'] = inputs.use_cache

            if inputs.use_cache:
                output, _, past_key_values = self(**model_inputs)
            else:
                output, _ = self(**model_inputs)

            next_token_logits = output[:, -1, :]

            next_token_logits = inputs.logits_processor(speech_token_ids, next_token_logits)
            next_token_logits = inputs.logits_warper(speech_token_ids, next_token_logits)

            next_token_probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            sampled_prob = next_token_probs.gather(-1, next_tokens).squeeze()
            
            if next_tokens[0,0].item() == inputs.eos_token:
                print(sampled_prob)
                break
            
            speech_token_ids = torch.cat([speech_token_ids, next_tokens], dim=-1)
            inputs.speech_token_lengths += 1
            speech_token_probs.append(sampled_prob.item())

        predicted_speech_token = speech_token_ids[:, inputs.speech_tokens.shape[-1]:]
        speech_token_probs = torch.tensor(speech_token_probs)

        return predicted_speech_token, speech_token_probs


class SpeechTokenPredictor(nn.Module):
    def __init__(self, config: STPModelConfig):
        super().__init__()
        
        self.config = config
        
        self.emb_speech, self.emb_text_tokens = self._prepare_embeddings()
        self.dec = Decoder(self.config.decoder)
        self.proj = nn.Linear(
            self.config.decoder.hidden_size,
            self.config.decoder.hidden_size,
        )
        if not self.config.decoder.use_rope:
            self.pos_emb_speech, self.pos_emb_text_tokens = self._prepare_rel_pos()
        
        self.classifier = nn.Linear(
            self.config.decoder.hidden_size,
            self.config.vocab_size_speech
        )
    
    def _prepare_embeddings(self) -> tuple[nn.Embedding, nn.Embedding]:
        emb_speech = nn.Embedding(
            self.config.vocab_size_speech, 
            self.config.decoder.hidden_size, 
            padding_idx=self.config.speech_pad_token
        )
        emb_text_tokens = nn.Embedding(
            self.config.vocab_size_text,
            self.config.decoder.hidden_size,
        )
        
        return emb_speech, emb_text_tokens
    
    def _prepare_rel_pos(self) -> tuple[nn.Embedding, nn.Embedding]:
        pos_emb_speech = nn.Embedding(
            self.config.max_pos + 1,
            self.config.decoder.hidden_size,
            padding_idx=0
        )
        pos_emb_text_tokens = nn.Embedding(
            self.config.max_pos + 1,
            self.config.decoder.hidden_size,
            padding_idx=0
        )
        
        return pos_emb_speech, pos_emb_text_tokens
    
    def forward(
        self,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        text_tokens_lengths: torch.LongTensor,
        speech_tokens_lengths: torch.LongTensor,
        use_cache = False,
        past_key_values = None
    ) -> list[torch.Tensor]:
        text_mask = get_sequence_mask(
            lengths=text_tokens_lengths,
            max_length=text_tokens.size(-1),
            dtype=text_tokens_lengths.dtype
        )
        speech_mask = get_sequence_mask(
            lengths=speech_tokens_lengths,
            max_length=speech_tokens.size(-1),
            dtype=speech_tokens_lengths.dtype
        )
        
        text_emb = self.emb_text_tokens(text_tokens) * text_mask.unsqueeze(-1)
        speech_emb = self.emb_speech(speech_tokens) * speech_mask.unsqueeze(-1)
        
        if not self.config.decoder.use_rope:
            text_pos_ids = text_mask.cumsum(1) * text_mask
            text_pos_emb = self.pos_emb_text_tokens(text_pos_ids) * text_mask.unsqueeze(-1)
            text_emb = text_emb + text_pos_emb
        
            speech_pos_ids = speech_mask.cumsum(1) * speech_mask
            speech_pos_emb = self.pos_emb_speech(speech_pos_ids) * speech_mask.unsqueeze(-1)
            speech_emb = speech_emb + speech_pos_emb
        
        if not use_cache or past_key_values is None:
            inputs_embeds = torch.cat([text_emb, speech_emb], dim=1)
        else:
            inputs_embeds = speech_emb[:, -1:, :]
        
        text_speech_masks = torch.cat([text_mask, speech_mask], dim=1)
        
        first_modality_length = text_tokens.shape[-1] if self.config.decoder.separate_pos_emb else None
        hidden_states, presents = self.dec(
            x=inputs_embeds,
            mask=text_speech_masks,
            past_key_values=past_key_values,
            use_cache=use_cache,
            first_modality_length=first_modality_length
        )
        
        text_speech_masks = text_speech_masks.unsqueeze(-1)
        hidden_states = hidden_states * text_speech_masks
        hidden_states = self.proj(hidden_states) * text_speech_masks
        
        speech_hidden_states = hidden_states[:, text_tokens.shape[-1]:, :]
        logits = self.classifier(speech_hidden_states) # [b, n_speech_tokens, n_vocab_speech]
        
        results = [logits, speech_mask]
        if use_cache:
            results.append(presents)
        
        return results
    
    def prepare_processor_and_warper(
        self,
        inputs: SpeechTokenPredictorSampleInputs,
    ):
        if inputs.logits_processor is None:
            inputs.logits_processor = LogitsProcessorList()
            if inputs.penalty != 1.0:
                inputs.logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(penalty=inputs.penalty)
                )
            if inputs.max_ar_steps is not None:
                inputs.logits_processor.append(
                    ForcedEOSTokenLogitsProcessor(
                        max_length=inputs.max_ar_steps,
                        eos_token_id=inputs.eos_token
                    )
                )
            if inputs.min_prob_eos is not None:
                inputs.logits_processor.append(
                    EOSTokenMinProbLogitsProcessor(
                        eos_token_id=inputs.eos_token,
                        min_eos_p=inputs.min_prob_eos
                    )
                )

        if inputs.logits_warper is None:
            inputs.logits_warper = LogitsProcessorList()
            if inputs.temperature != 1.0:
                inputs.logits_warper.append(TemperatureLogitsWarper(inputs.temperature))
            if inputs.top_k is not None and inputs.top_k > 0:
                inputs.logits_warper.append(TopKLogitsWarper(top_k=inputs.top_k))
            if inputs.top_p is not None and inputs.top_p < 1.0:
                inputs.logits_warper.append(TopPLogitsWarper(top_p=inputs.top_p))
        
        return inputs
    
    def sample(
        self, inputs: SpeechTokenPredictorSampleInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        inputs = self.prepare_processor_and_warper(inputs)
        speech_token_ids = inputs.speech_tokens.long().clone()
        speech_token_probs = []
        past_key_values = None

        while True:
            model_inputs = {
                'text_tokens': inputs.text_tokens,
                'speech_tokens': speech_token_ids,
                'text_tokens_lengths': inputs.text_lengths,
                'speech_tokens_lengths': inputs.speech_token_lengths,
            }
            if inputs.use_cache:
                model_inputs['past_key_values'] = past_key_values
                model_inputs['use_cache'] = inputs.use_cache

            if inputs.use_cache:
                output, _, past_key_values = self(**model_inputs)
            else:
                output, _ = self(**model_inputs)

            next_token_logits = output[:, -1, :]

            next_token_logits = inputs.logits_processor(speech_token_ids, next_token_logits)
            next_token_logits = inputs.logits_warper(speech_token_ids, next_token_logits)

            next_token_probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            sampled_prob = next_token_probs.gather(-1, next_tokens).squeeze()
            
            if next_tokens[0,0].item() == inputs.eos_token:
                print(sampled_prob)
                break
            
            speech_token_ids = torch.cat([speech_token_ids, next_tokens], dim=-1)
            inputs.speech_token_lengths += 1
            speech_token_probs.append(sampled_prob.item())

        predicted_speech_token = speech_token_ids[:, inputs.speech_tokens.shape[-1]:]
        speech_token_probs = torch.tensor(speech_token_probs)

        return predicted_speech_token, speech_token_probs

    def sample_old(
        self, inputs: SpeechTokenPredictorSampleInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize logits_processor if not provided
        if inputs.logits_processor is None:
            inputs.logits_processor = LogitsProcessorList()
            # if inputs.penalty != 1.0:
            #     inputs.logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=inputs.penalty))

        # Initialize logits_warper if not provided
        if inputs.logits_warper is None:
            inputs.logits_warper = LogitsProcessorList()
            if inputs.temperature != 1.0:
                inputs.logits_warper.append(TemperatureLogitsWarper(inputs.temperature))
            if inputs.top_k is not None and inputs.top_k > 0:
                inputs.logits_warper.append(TopKLogitsWarper(top_k=inputs.top_k))
            if inputs.top_p is not None and inputs.top_p < 1.0:
                inputs.logits_warper.append(TopPLogitsWarper(top_p=inputs.top_p))
            
            # TODO move this to processor
            if inputs.penalty != 1.0:
                inputs.logits_warper.append(RepetitionPenaltyLogitsProcessor(penalty=inputs.penalty))

        # Initialize generation variables
        speech_token_ids = inputs.speech_tokens.long().clone()
        speech_token_probs = []

        past_key_values = None

        for _ in range(inputs.max_ar_steps):
            # Prepare inputs for the model
            model_inputs = {
                'text_tokens': inputs.text_tokens,
                'speech_tokens': speech_token_ids,
                'text_tokens_lengths': inputs.text_lengths,
                'speech_tokens_lengths': inputs.speech_token_lengths,
            }
            if inputs.use_cache:
                model_inputs['past_key_values'] = past_key_values
                model_inputs['use_cache'] = inputs.use_cache

            # Forward pass through the model
            if inputs.use_cache:
                output, _, past_key_values = self(**model_inputs)
            else:
                output, _ = self(**model_inputs)

            # Get logits for the last token
            next_token_logits = output[:, -1, :]

            # Apply logits processors and warpers
            next_token_logits = inputs.logits_processor(speech_token_ids, next_token_logits)
            next_token_logits = inputs.logits_warper(speech_token_ids, next_token_logits)

            # Compute probabilities
            next_token_probs = nn.functional.softmax(next_token_logits, dim=-1)
            
            end = False
            while True:
                next_tokens = torch.multinomial(next_token_probs, num_samples=1)
                if next_tokens[0,0].item() != inputs.eos_token:
                    break
                if next_tokens[0, 0].item() == inputs.eos_token and next_token_probs[0, next_tokens[0, 0].item()].item() > inputs.min_prob_eos:
                    print('eos prob:', next_token_probs[0, next_tokens[0, 0].item()].item())
                    print('eos token found')
                    end = True
                    break
            if end:
                break

            # Sample next token
            # next_tokens = torch.multinomial(next_token_probs, num_samples=1)

            # Get the probability of the sampled token
            sampled_prob = next_token_probs.gather(-1, next_tokens).squeeze()

            # Check for EOS token and probability
            # if next_tokens.item() == inputs.eos_token and sampled_prob.item() >= inputs.min_prob_eos:
            #     print(f'eos prob: {sampled_prob.item()}')
            #     print('eos token found')
            #     break  # End generation
            
            # while True:
            #     next_tokens = torch.multinomial(sampled_prob, num_samples=1)
            
            # Append the sampled token and its probability
            speech_token_ids = torch.cat([speech_token_ids, next_tokens], dim=-1)
            inputs.speech_token_lengths += 1
            speech_token_probs.append(sampled_prob.item())

        # Get the predicted speech tokens (excluding the prompt)
        predicted_speech_token = speech_token_ids[:, inputs.speech_tokens.shape[-1]:]

        # Convert speech_token_probs to tensor
        speech_token_probs = torch.tensor(speech_token_probs)

        return predicted_speech_token, speech_token_probs


class SpeechTokenPredictorForClassification(nn.Module):
    def __init__(self, config: STPTrainerConfig):
        super().__init__()
        
        self.config = config
        self.model = SpeechTokenPredictor(config.model)
        
        self.accuracy_metrics = nn.ModuleDict({
            f"top_{top_k}_accuracy": MulticlassAccuracy(
                num_classes=self.config.model.vocab_size_speech,
                top_k=top_k,
                ignore_index=-100
            ) for top_k in self.config.top_k_accuracies
        })
        
    def forward(
        self,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        text_tokens_lengths: torch.LongTensor,
        speech_tokens_lengths: torch.LongTensor,
    ) -> SpeechTokenPredictorForClassificationOutput:
        
        loss_dict = {}
        
        logits, speech_mask = self.model(
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            text_tokens_lengths=text_tokens_lengths,
            speech_tokens_lengths=speech_tokens_lengths
        )

        loss_dict['lm_loss'] = self.compute_shifted_cls_loss(
            logits=logits,
            labels=speech_tokens,
            mask=speech_mask
        )
        
        loss_dict.update(
            self.compute_accuracy(
                logits=logits,
                labels=speech_tokens,
                mask=speech_mask
            )
        )
        loss_dict["total"] = loss_dict["lm_loss"]
        
        return SpeechTokenPredictorForClassificationOutput(
            loss_dict=loss_dict,
            logits=logits,
            logits_mask=speech_mask
        )
    
    def compute_shifted_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        
        shift_labels = labels[..., 1:].contiguous()
        mask = mask[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = shift_labels.where(mask.bool(), -100)
        batch_size, seq_length, vocab_size = shift_logits.shape
        
        return torch.nn.functional.cross_entropy(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length)
        )

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ):
        shift_labels = labels[..., 1:].contiguous()
        mask = mask[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()

        shift_labels = shift_labels.where(mask.bool(), -100)

        batch_size, seq_length, vocab_size = shift_logits.shape
        shift_logits_flat = shift_logits.view(batch_size * seq_length, vocab_size)
        shift_labels_flat = shift_labels.view(batch_size * seq_length)

        results = {}
        for name, metric in self.accuracy_metrics.items():
            accuracy_value = metric(shift_logits_flat, shift_labels_flat)
            results[name] = accuracy_value
        
        return results
