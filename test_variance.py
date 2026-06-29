from adapters.llm_bridge import LLMTelemetryBridge
from core.engine import ContinuousBeliefStateEngine
import numpy as np

print("Initializing Telemetry Bridge...")
bridge = LLMTelemetryBridge("gpt2")
text_signal = "An unexpected external distribution shift has been detected in the environment."
latent_vector = bridge.extract_telemetry(text_signal)

# Extract pristine target segment (first 64 principal coordinates)
y_target = latent_vector[:64, :]
y_target_norm = np.linalg.norm(y_target)
print(f"Target Manifold Signal Norm (||y_t||): {y_target_norm:.4f}")

variance_experiments = [0.0, 0.1, 1.0, 5.0, 25.0, 100.0]

print(f"\n--- Pristine Zero-Step Variance Spectrum ---")
for var in variance_experiments:
    # 1. Instantiate engine (clean state creation)
    engine = ContinuousBeliefStateEngine(
        state_dim=64, 
        observation_dim=bridge.embedding_dim, 
        learning_rate=0.2, 
        init_var=var
    )
    
    # 2. Capture the pristine raw initial state norm before any update runs
    raw_state_norm = np.linalg.norm(engine.x)
    
    # 3. Process direct forward pass through the pristine initialization state
    x_hat = engine._forward_mlp(engine.x)
    
    # 4. Measure pure structural initialization drift
    pure_init_drift = np.linalg.norm(y_target - x_hat)
    
    print(f"Variance: {var:<5} | Pure Init State Norm: {raw_state_norm:8.4f} | Zero-Step Drift: {pure_init_drift:.4f}")
