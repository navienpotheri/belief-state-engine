import torch
import torch.nn.functional as F
import numpy as np
from adapters.llm_bridge import LLMTelemetryBridge
from core.engine import ContinuousBeliefStateEngine
from core.attention_injector import LMMAttentionInjector

print("====== Launching Automated Evaluation Entropy Sweep ======")

bridge = LLMTelemetryBridge("gpt2")
model = bridge.model
tokenizer = bridge.tokenizer

learning_rates = [0.0, 0.05, 0.2, 0.5, 0.8]
context_prompt = "The deployment system environment is experiencing anomalous distribution variance."
inputs = tokenizer(context_prompt, return_tensors="pt")

print(f"\nTarget Evaluation Context: '{context_prompt}'\n")

for lr in learning_rates:
    # 1. Instantiate engine with current step learning rate
    engine = ContinuousBeliefStateEngine(state_dim=64, observation_dim=bridge.embedding_dim, learning_rate=lr)
    injector = LMMAttentionInjector(model, target_layer_idx=-1)
    
    # 2. Extract initial telemetry step vector
    initial_telemetry = bridge.extract_telemetry(context_prompt)
    metrics = engine.update(initial_telemetry)
    
    # 3. Register our hook and apply our belief state transformation vector
    injector.register_injector()
    injector.set_active_drift(engine.x)
    
    # 4. Run a single-token forward evaluation pass to extract raw generation logits
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Calculate Shannon Entropy of the output distribution profile
        entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
        
    injector.remove_injector()
    print(f"Learning Rate: {lr:<5} | Initial Drift: {metrics['prediction_error_norm']:8.4f} | Logit Prediction Entropy: {entropy:.4f}")

print("\n====== Evaluation Sweep Completed Successfully ======")
