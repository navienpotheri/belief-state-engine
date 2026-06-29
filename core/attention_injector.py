import torch
import torch.nn as nn
import numpy as np

class LMMAttentionInjector:
    def __init__(self, model, target_layer_idx: int = -1):
        """
        PyTorch Forward Hook manager to inject belief-state tracking vectors 
        directly into the LLM's Key-Value attention stream.
        """
        self.model = model
        
        # FIX: Dynamically adapt path depending on base GPT2Model vs GPT2LMHeadModel wrapper
        if hasattr(self.model, "transformer"):
            self.layer = self.model.transformer.h[target_layer_idx].attn
        else:
            self.layer = self.model.h[target_layer_idx].attn
            
        self.hook_handle = None
        self.current_drift_vector = None

    def _attention_hook(self, module, input_tensors, output_tensors):
        """
        The runtime hook that intercepts and warps the internal attention tensors.
        """
        if self.current_drift_vector is None:
            return output_tensors

        hidden_states = output_tensors[0]
        
        # Single-line tensor allocation matching type and execution device memory
        drift_tensor = torch.tensor(self.current_drift_vector, dtype=hidden_states.dtype, device=hidden_states.device).squeeze(-1)
        
        # Pad tracking spaces out to 768 dimensions smoothly
        padded_drift = torch.zeros(hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
        padded_drift[:drift_tensor.shape[0]] = drift_tensor
        
        # Inject state warp straight into the final causal generation frame token
        hidden_states[:, -1, :] += padded_drift
        
        return (hidden_states,) + output_tensors[1:]

    def register_injector(self):
        """Activates the forward hook pipeline."""
        if self.hook_handle is None:
            self.hook_handle = self.layer.register_forward_hook(self._attention_hook)
            print("Attention injection hook successfully registered on target transformer layer.")

    def remove_injector(self):
        """Deactivates the hook to restore normal inference."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("Attention injection hook detached.")

    def set_active_drift(self, drift_vector: np.ndarray):
        """Updates the tracking coordinates applied during the next model pass."""
        self.current_drift_vector = drift_vector
