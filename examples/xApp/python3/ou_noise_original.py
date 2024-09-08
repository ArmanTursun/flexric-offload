import numpy as np

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
