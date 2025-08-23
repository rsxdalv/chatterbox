import torch
from typing import Optional, Tuple, Callable, Any


class T3StepCUDAGraphWrapper:
    """
    A wrapper class that automatically captures and replays CUDA graphs for optimized inference.

    The graph is captured on the first call and subsequent calls replay the captured graph
    for improved performance.
    """

    def __init__(
        self,
        generate_token: Callable,
        patched_model: Any,
        kv_cache: Any,
        repetition_penalty_processor: Any,
        min_p_warper: Any,
        top_p_warper: Any,
    ):
        """
        Initialize the CUDA graph wrapper.

        Args:
            generate_token: The function to wrap with CUDA graph
            patched_model: The model instance
            kv_cache: The key-value cache
            repetition_penalty_processor: Repetition penalty processor
            min_p_warper: Min-p warper
            top_p_warper: Top-p warper
        """
        self.generate_token = generate_token
        self.patched_model = patched_model
        self.kv_cache = kv_cache
        self.repetition_penalty_processor = repetition_penalty_processor
        self.min_p_warper = min_p_warper
        self.top_p_warper = top_p_warper

        # Graph state
        self._graph = None
        self._is_captured = False

        # Static tensors (will be set during capture)
        self._static_speech_embedding_cache = None
        self._static_output_logits = None
        self._static_i_tensor = None
        self._static_batch_idx = None
        self._static_speech_pos_embedding_cache = None
        self._static_generated_ids = None
        self._static_cfg_weight = None
        self._static_temperature = None
        self._static_stride_length = None

        # Output tensors
        self._static_out_1 = None
        self._static_out_2 = None

    def _capture_graph(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        stride_length: int,
        max_position: Optional[int] = None,
    ) -> None:
        """Capture the CUDA graph with the provided tensors."""
        self._graph = torch.cuda.CUDAGraph()

        # Clone static tensors
        self._static_speech_embedding_cache = speech_embedding_cache.clone()
        self._static_output_logits = output_logits
        self._static_i_tensor = i_tensor.clone()
        self._static_batch_idx = batch_idx.clone()
        self._static_speech_pos_embedding_cache = speech_pos_embedding_cache.clone()
        self._static_generated_ids = generated_ids
        self._static_cfg_weight = cfg_weight
        self._static_temperature = temperature
        self._static_stride_length = stride_length
        self._static_max_position = max_position

        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                self._static_out_1, self._static_out_2 = self.generate_token(
                    self._static_speech_embedding_cache,
                    self._static_output_logits,
                    self._static_i_tensor,
                    self._static_batch_idx,
                    self._static_speech_pos_embedding_cache,
                    self._static_generated_ids,
                    self._static_cfg_weight,
                    self._static_temperature,
                    self.repetition_penalty_processor,
                    self.min_p_warper,
                    self.top_p_warper,
                    self.patched_model,
                    self.kv_cache,
                    self._static_stride_length,
                    self._static_max_position,
                )

        self._is_captured = True

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        repetition_penalty_processor: Any = None,
        min_p_warper: Any = None,
        top_p_warper: Any = None,
        patched_model: Any = None,
        kv_cache: Any = None,
        stride_length: int = 1,
        max_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the wrapped function with CUDA graph optimization.

        On first call, captures the graph. Subsequent calls replay the captured graph.

        Args:
            speech_embedding_cache: Speech embedding cache tensor
            output_logits: Output logits tensor
            i_tensor: Index tensor
            batch_idx: Batch index tensor
            speech_pos_embedding_cache: Speech position embedding cache tensor
            generated_ids: Generated IDs tensor
            cfg_weight: Configuration weight
            temperature: Temperature value
            repetition_penalty_processor: Repetition penalty processor (ignored after capture)
            min_p_warper: Min-p warper (ignored after capture)
            top_p_warper: Top-p warper (ignored after capture)
            patched_model: Model instance (ignored after capture)
            kv_cache: Key-value cache (ignored after capture)
            stride_length: Stride length

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
                speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                stride_length,
                max_position,
            )
        else:
            # Subsequent runs: update static tensors and replay
            self._static_speech_embedding_cache.copy_(speech_embedding_cache)
            self._static_output_logits.copy_(output_logits)
            self._static_i_tensor.copy_(i_tensor)
            self._static_batch_idx.copy_(batch_idx)
            self._static_speech_pos_embedding_cache.copy_(speech_pos_embedding_cache)
            self._static_generated_ids.copy_(generated_ids)
            self._static_cfg_weight = cfg_weight
            self._static_temperature = temperature
            self._static_stride_length = stride_length
            self._static_max_position = max_position

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
        self._static_speech_pos_embedding_cache = None
        self._static_generated_ids = None
        self._static_cfg_weight = None
        self._static_temperature = None
        self._static_stride_length = None
        self._static_out_1 = None
        self._static_out_2 = None
        self._static_max_position = None
