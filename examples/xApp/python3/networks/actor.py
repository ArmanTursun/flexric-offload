import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=(128, 128), learning_rate=0.0001, tau=0.001):
        """
        Constructor for the Actor network in PyTorch.

        Args:
        - state_size (int): Dimension of the input state.
        - action_size (int): Dimension of the output action space (number of weights).
        - hidden_units (tuple): Number of hidden units in each layer. Default: (128, 128).
        - learning_rate (float): Learning rate for training the model. Default: 0.0001.
        - tau (float): Soft update rate for the target network. Default: 0.001.
        """
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.tau = tau

        # Define the fully connected layers with reduced hidden units for more stable training
        self.fc1 = nn.Linear(state_size, hidden_units[0])  # First hidden layer
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])  # Second hidden layer
        self.fc3 = nn.Linear(hidden_units[1], action_size)  # Output layer

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Xavier initialization of weights
        self.init_weights()

    def init_weights(self):
        """Initializes the network weights with Xavier uniform initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass for the Actor network.

        Args:
        - state (torch.Tensor): Input state tensor of shape (batch_size, state_size).

        Returns:
        - action (torch.Tensor): Output tensor representing weights of shape (batch_size, action_size),
                                 where the values sum to 1.
        """
        x = torch.relu(self.fc1(state))  # Pass through the first hidden layer
        x = torch.relu(self.fc2(x))  # Pass through the second hidden layer
        action_weights = torch.softmax(self.fc3(x), dim=-1)  # Softmax to ensure the sum of weights is 1
        return action_weights

    def train(self, states, action_gradients):
        """
        Updates the weights of the main network based on the provided gradients.

        Args:
        - states (torch.Tensor): Input states of shape (batch_size, state_size).
        - action_gradients (torch.Tensor): Action gradients for updating the network, shape (batch_size, action_size).
        """
        self.optimizer.zero_grad()  # Zero out the gradients before backward pass
        actions = self.forward(states)
        loss = -torch.mean(actions * action_gradients)  # The negative sign is for gradient ascent
        loss.backward()
        self.optimizer.step()

    def _soft_update(self, target_model, tau=None):
        """
        Soft update of the target model parameters using the main model's parameters.

        Args:
        - target_model (Actor): The target actor model to be updated.
        - tau (float): Soft update parameter. If None, the class's tau value is used.
        """
        if tau is None:
            tau = self.tau
        for target_param, param in zip(target_model.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_target_model(self, target_model):
        """
        Update the target network using the soft update method.
        """
        self._soft_update(target_model)
'''
# Example usage
if __name__ == "__main__":
    # Example with BLER and Energy states (mean, max, min, skewness)
    bler_state = [0.1, 0.2, 0.05, 0.1]  # Example BLER state (mean, max, min, skewness)
    energy_state = [1.0, 1.5, 0.8, 0.05]  # Example Energy state (mean, max, min, skewness)

    # Concatenate BLER and energy states
    state = np.array(bler_state + energy_state)  # Shape: (8,)
    state_size = len(state)  # 8
    action_size = 2  # Example number of actions (weights)

    # Create the actor network
    actor = Actor(state_size, action_size)

    # Convert state to a PyTorch tensor and perform a forward pass
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension if needed
    actions = actor(state_tensor)  # Forward pass to get actions (weights)
    print("Actions:", actions)  # The actions (weights) will sum to 1

    # Soft update target actor network from the actor network
    actor.update_target_model(actor)
'''