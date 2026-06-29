import numpy as np

class ContinuousBeliefStateEngine:
    def __init__(self, state_dim: int, observation_dim: int, learning_rate: float = 0.05):
        """
        Initializes the continuous tracking engine using dynamic state-space mechanics.
        
        Args:
            state_dim (int): Dimensionality of the internal hidden belief manifold (s).
            observation_dim (int): Dimensionality of incoming streaming token embeddings (x).
            learning_rate (float): Step size parameter (eta) governing error correction adjustments.
        """
        self.d_s = state_dim
        self.d_x = observation_dim
        self.eta = learning_rate
        
        # Initialize internal belief state vector s_t at the origin
        self.s = np.zeros((self.d_s, 1))
        
        # State Transition Matrix A (Models intrinsic dynamic propagation of the manifold)
        self.A = np.eye(self.d_s) * 0.99
        
        # Observation Projection Matrix C (Maps internal states to expected observation space)
        # Using a normalized random orthogonal projection to simulate initial sensory mapping
        q, _ = np.linalg.qr(np.random.randn(self.d_x, self.d_s))
        self.C = q

    def update(self, x_t: np.ndarray) -> dict:
        """
        Updates the internal state trajectory vector based on incoming multi-modal data streams
        by computing and minimizing the local prediction error field.
        
        Args:
            x_t (np.ndarray): Streaming input matrix/vector of shape (observation_dim, 1)
            
        Returns:
            dict: Diagnostic metrics outlining internal manifold trajectory states
        """
        # Ensure correct vector orientation
        x_t = x_t.reshape((self.d_x, 1))
        
        # 1. Internal Dynamic Prediction Step (Prior Belief Propagation)
        s_prior = np.dot(self.A, self.s)
        
        # 2. Sensory Projection Mapping (Expected Observation Generation)
        x_expected = np.dot(self.C, s_prior)
        
        # 3. Compute Prediction Error Matrix Field (e_t)
        # Measures the absolute directional divergence between reality and internal models
        prediction_error = x_t - x_expected
        
        # 4. State Manifold Correction Step via Active Error Integration
        # Warps the persistent hidden trajectory coordinates directly based on gradient surprise
        correction_term = self.eta * np.dot(self.C.T, prediction_error)
        self.s = s_prior + correction_term
        
        return {
            "internal_state_trajectory": self.s.copy(),
            "prediction_error_norm": float(np.linalg.norm(prediction_error)),
            "correction_magnitude": float(np.linalg.norm(correction_term))
        }

    def reset_state(self):
        """Resets the continuous internal state tracking vector back to baseline coordinates."""
        self.s = np.zeros((self.d_s, 1))

# Simple integration test verification block
if __name__ == "__main__":
    print("Initializing Core Continuous Tracking Engine Test Loop...")
    # Setup standard manifold space configurations
    engine = ContinuousBeliefStateEngine(state_dim=128, observation_dim=512, learning_rate=0.1)
    
    # Simulate a small sequence of streaming text/image embedding frames
    for step in range(3):
        mock_embedding = np.random.randn(512, 1)
        metrics = engine.update(mock_embedding)
        print(f"Frame {step+1} -> Prediction Error: {metrics['prediction_error_norm']:.4f} | Correction Delta: {metrics['correction_magnitude']:.4f}")
