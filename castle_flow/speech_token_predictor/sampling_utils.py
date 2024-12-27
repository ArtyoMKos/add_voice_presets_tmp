import torch

from typing import Union

from transformers import LogitsProcessor


class EOSTokenMinProbLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        eos_token_id: Union[int, list[int], torch.Tensor],
        min_eos_p: float,
        device: str = "cpu"
    ):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores.float(), dim=-1)

        for eos_id in self.eos_token_id:
            eos_prob = probs[:, eos_id]
            
            scores[eos_prob < self.min_eos_p, eos_id] = -float('inf')
        
        return scores
