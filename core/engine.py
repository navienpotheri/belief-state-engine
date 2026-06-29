import numpy as np

class ContinuousBeliefStateEngine:
    def __init__(self, state_dim: int = 64, observation_dim: int = 768, learning_rate: float = 0.1, init_var: float = 1.0):
        """
        Publication-Grade Recurrent Belief State Engine using Direct Subspace Manifold Anchoring.
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.alpha = learning_rate
        
        # 1. Initialize latent state coordinates from a clean variance envelope
        self.x = np.random.randn(state_dim, 1) * np.sqrt(init_var)
        
        # Continuous Recurrent Hidden State Memory Matrix
        self.h_prev = np.zeros((state_dim, 1))
        
        # Recurrent state transition weights
        self.W_rec = np.eye(state_dim) * 0.85
        self.W_state = np.eye(state_dim) * 0.10
        
        # 2. Non-linear activation gate parameters operating inside the 64-D space
        self.hidden_dim = state_dim * 2
        self.W1 = np.random.randn(self.hidden_dim, state_dim) * 0.05
        self.b1 = np.zeros((self.hidden_dim, 1))
        
        self.W2 = np.random.randn(state_dim, self.hidden_dim) * 0.05
        self.b2 = np.zeros((state_dim, 1))

    def _gelu(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * np.power(z, 3))))

    def _forward_mlp(self, x: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = self._gelu(self.z1)
        x_residual = np.dot(self.W2, self.a1) + self.b2
        return x + x_residual

    def update(self, y_t: np.ndarray) -> dict:
        # Multi-Step Recurrent Transition Logic (Continuous Hidden Memory State Propagation)
        self.h_current = np.tanh(np.dot(self.W_rec, self.h_prev) + np.dot(self.W_state, self.x))
        self.x = self.h_current
        self.h_prev = self.h_current
        
        # Extract the principal 64 dimensions from the causal embedding payload
        y_projected = y_t[:self.state_dim, :]
        
        # Generate the non-linear belief state synthesis
        x_hat = self._forward_mlp(self.x)
        
        # Compute pure Euclidean error distance inside the shared subspace manifold
        prediction_error = y_projected - x_hat
        prediction_error_norm = float(np.linalg.norm(prediction_error))
        
        # State optimization correction step
        delta_a1 = np.dot(self.W2.T, prediction_error)
        delta_z1 = delta_a1 * (self.z1 > 0) 
        delta_x = np.dot(self.W1.T, delta_z1) + prediction_error
        
        self.x += self.alpha * delta_x
        
        return {
            "prediction_error_norm": prediction_error_norm,
            "latent_state_norm": float(np.linalg.norm(self.x))
        }
