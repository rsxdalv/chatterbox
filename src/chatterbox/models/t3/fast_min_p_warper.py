import torch
from transformers.generation.logits_process import MinPLogitsWarper


class FastMinPLogitsWarper(MinPLogitsWarper):
    def __init__(self, *args, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.false_tensor = torch.zeros(1, dtype=torch.bool, device=device)
        self.min_p = torch.tensor(self.min_p, device=device)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(scores, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(
            tokens_to_remove, dim=-1, index=sorted_indices
        )
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = self.false_tensor

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
