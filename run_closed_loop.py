import time
import torch
import numpy as np
from adapters.llm_bridge import LLMTelemetryBridge
from core.engine import ContinuousBeliefStateEngine
from core.attention_injector import LMMAttentionInjector

print("====== Booting Closed-Loop Agency Pipeline ======")

# 1. Initialize our telemetry bridge components
bridge = LLMTelemetryBridge("gpt2")
model = bridge.model
tokenizer = bridge.tokenizer

# 2. Spin up our non-linear belief tracking engine core
engine = ContinuousBeliefStateEngine(
    state_dim=64, 
    observation_dim=bridge.embedding_dim, 
    learning_rate=0.1, 
    init_var=1.0
)

# 3. Connect our PyTorch attention forward injector hook
injector = LMMAttentionInjector(model, target_layer_idx=-1)
injector.register_injector()

# Define an input context sequence to monitor
context_prompt = "The deployment system environment is experiencing anomalous distribution variance."
print(f"\nInitial Context Window Input: '{context_prompt}'")

# Extract initial telemetry coordinates and seed the tracking state
initial_telemetry = bridge.extract_telemetry(context_prompt)
metrics = engine.update(initial_telemetry)
print(f"Initial State Engine Drift Metric: {metrics['prediction_error_norm']:.4f}")

# Feed the tracking error directly into the injector cache setup
injector.set_active_drift(engine.x)

print("\nExecuting forward generation step under active belief vector injection...")
inputs = tokenizer(context_prompt, return_tensors="pt")

# Generate the next token while the hook warps the internal attention layers
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=5, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResulting Closed-Loop Generated Text Output:\n -> \"{generated_text}\"")

# Clean up hooks to restore clean environment state
injector.remove_injector()
print("\n====== Pipeline Cycle Completed Successfully ======")
