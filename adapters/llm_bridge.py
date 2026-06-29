import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.engine import ContinuousBeliefStateEngine

class LLMTelemetryBridge:
    def __init__(self, model_name: str = "gpt2", target_layer: int = -1):
        print(f"Loading tokenizer and causal model weight structures for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load as a Causal Language Model so it contains the .generate() interface loop
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.target_layer = target_layer
        self.embedding_dim = self.model.config.hidden_size
        print(f"Bridge initialized. Detected latent space dimension: {self.embedding_dim}")

    def extract_telemetry(self, text_stream: str) -> np.ndarray:
        inputs = self.tokenizer(text_stream, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[self.target_layer]
        
        # Extract exclusively the final causal token state payload vector
        final_token_tensor = hidden_states[0, -1, :]
        
        return final_token_tensor.cpu().numpy().reshape(-1, 1)

if __name__ == "__main__":
    print("Testing LLM Bridge Architecture...")
    bridge = LLMTelemetryBridge("gpt2")
    engine = ContinuousBeliefStateEngine(state_dim=64, observation_dim=bridge.embedding_dim, learning_rate=0.2)
    
    mock_stream = [
        "The system initial conditions look completely stable.",
        "An unexpected external distribution shift has been detected in the environment.",
        "The system is rapidly adapting to minimize prediction errors across all channels."
    ]
    
    for iteration, text in enumerate(mock_stream):
        latent_vector = bridge.extract_telemetry(text)
        metrics = engine.update(latent_vector)
        print(f"\n[Step {iteration+1}] Input Signal: '{text}'")
        print(f" -> Real Telemetry Vector Shape: {latent_vector.shape}")
        print(f" -> Computed Trajectory Drift: {metrics['prediction_error_norm']:.4f}")
