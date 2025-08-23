import torch
from typing import Optional, Tuple, Callable, Any


class T3StepCUDAGraphWrapper:
    """
    A wrapper class that automatically captures and replays CUDA graphs for optimized inference.

    The graph is captured on the first call and subsequent calls replay the captured graph
    for improved performance.
    """

    def __init__(self, generate_t3_token: Callable, patched_model: Any, kv_cache: Any):
        """
        Initialize the CUDA graph wrapper.

        Args:
            generate_t3_token: The function to wrap with CUDA graph
            patched_model: The model instance
            kv_cache: The key-value cache
        """
        self.generate_t3_token = generate_t3_token
        self.patched_model = patched_model
        self.kv_cache = kv_cache

        # Graph state
        self._graph = None
        self._is_captured = False

        # Static tensors (will be set during capture)
        self._static_speech_embedding_cache = None
        self._static_output_logits = None
        self._static_i_tensor = None
        self._static_batch_idx = None
        self._static_position_embeds = None
        self._static_generated_ids = None
        self._static_cfg_weight = None
        self._static_temperature = None

        # Output tensors
        self._static_out_1 = None
        self._static_out_2 = None

    def _capture_graph(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        position_embeds: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
    ) -> None:
        """Capture the CUDA graph with the provided tensors."""
        self._graph = torch.cuda.CUDAGraph()

        # Clone static tensors
        self._static_speech_embedding_cache = speech_embedding_cache.clone()
        # self._static_output_logits = output_logits.clone()
        self._static_output_logits = output_logits
        self._static_i_tensor = i_tensor.clone()
        self._static_batch_idx = batch_idx.clone()
        self._static_position_embeds = position_embeds.clone()
        # self._static_generated_ids = generated_ids.clone()
        self._static_generated_ids = generated_ids
        self._static_cfg_weight = cfg_weight
        self._static_temperature = temperature

        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                self._static_out_1, self._static_out_2 = self.generate_t3_token(
                    self._static_speech_embedding_cache,
                    self._static_output_logits,
                    self._static_i_tensor,
                    self._static_batch_idx,
                    self._static_position_embeds,
                    self._static_generated_ids,
                    self._static_cfg_weight,
                    self._static_temperature,
                    self.patched_model,
                    self.kv_cache,
                )

        self._is_captured = True

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        position_embeds: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        patched_model: Any = None,
        kv_cache: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the wrapped function with CUDA graph optimization.

        On first call, captures the graph. Subsequent calls replay the captured graph.

        Args:
            speech_embedding_cache: Speech embedding cache tensor
            output_logits: Output logits tensor
            i_tensor: Index tensor
            batch_idx: Batch index tensor
            position_embeds: Position embeddings tensor
            generated_ids: Generated IDs tensor
            cfg_weight: Configuration weight
            temperature: Temperature value

        Returns:
            Tuple of output tensors from the wrapped function
        """
        if not self._is_captured:
            # First run: capture the graph
            self._capture_graph(
                speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                position_embeds,
                generated_ids,
                cfg_weight,
                temperature,
            )
        else:
            # Subsequent runs: update static tensors and replay
            self._static_speech_embedding_cache.copy_(speech_embedding_cache)
            self._static_output_logits.copy_(output_logits)
            self._static_i_tensor.copy_(i_tensor)
            self._static_batch_idx.copy_(batch_idx)
            self._static_position_embeds.copy_(position_embeds)
            self._static_generated_ids.copy_(generated_ids)
            self._static_cfg_weight = cfg_weight
            self._static_temperature = temperature

            # Replay the graph
            self._graph.replay()

        return self._static_out_1, self._static_out_2

    def is_captured(self) -> bool:
        """Check if the CUDA graph has been captured."""
        return self._is_captured

    def reset(self) -> None:
        """Reset the wrapper, forcing graph recapture on next call."""
        self._graph = None
        self._is_captured = False
        self._static_speech_embedding_cache = None
        self._static_output_logits = None
        self._static_i_tensor = None
        self._static_batch_idx = None
        self._static_position_embeds = None
        self._static_generated_ids = None
        self._static_cfg_weight = None
        self._static_temperature = None
        self._static_out_1 = None
        self._static_out_2 = None
