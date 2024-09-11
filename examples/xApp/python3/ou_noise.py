import numpy as np
import torch
from networks.actor import Actor

class OUNoise:
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2, decay_rate=0.99):
        """
        Initialize OU Noise process parameters.
        :param action_size: Size of the action space.
        :param mu: The mean around which the noise reverts (default: 0).
        :param theta: The speed of mean reversion (default: 0.15).
        :param sigma: Initial volatility or randomness in the noise (default: 0.2).
        :param decay_rate: Decay rate for sigma, reduces noise over time (default: 0.99).
        """
        self.action_size = action_size
        self.mu = mu * np.ones(self.action_size)
        self.theta = theta
        self.sigma = sigma
        self.decay_rate = decay_rate  # Rate at which noise decays
        self.state = np.zeros(self.action_size)
        self.reset()

    def reset(self):
        """Reset the internal state (noise) to the mean."""
        self.state = np.ones(self.action_size) * self.mu

    def sample(self):
        """
        Generate noise based on the OU process.
        :return: A noise vector to add to the action.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state

    def decay_noise(self):
        """
        Apply decay to the noise over time by reducing sigma.
        """
        self.sigma *= self.decay_rate  # Reduce sigma with the decay rate
        self.sigma = max(self.sigma, 0.01)  # Ensure sigma doesn't go below a small threshold

'''
# Example usage with BLER and Energy states (mean, max, min, skewness)
bler_state = [0.1, 0.2, 0.05, 0.1]  # Example BLER state (mean, max, min, skewness)
energy_state = [1.0, 1.5, 0.8, 0.05]  # Example Energy state (mean, max, min, skewness)

# Concatenate BLER and energy states
state = bler_state + energy_state  # Total state size = 8
state_size = len(state)  # 8
action_size = 2  # Example number of actions (weights)

# Example: Generate state and pass it through the actor
actor = Actor(state_size, action_size)
state = torch.FloatTensor([[0.1, 0.2, 0.05, 0.1, 1.0, 1.5, 0.8, 0.05]])  # Example state
action = actor(state).detach().numpy()  # Get action from the actor (as a NumPy array)

noise = OUNoise(action_size)

# Add OU noise to the action
noisy_action = action + noise.sample()

# Clip the action to the valid range (e.g., between -1 and 1 for many environments)
clipped_action = np.clip(noisy_action, -1, 1)

# Print the noisy, clipped action
print("Noisy, clipped action:", clipped_action)

'''