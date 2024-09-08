import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=(300, 600), learning_rate=0.0001, tau=0.001):
        """
        Constructor for the Actor network in PyTorch.

        Args:
        - state_size (int): Dimension of the input state.
        - action_size (int): Dimension of the output action space (number of weights).
        - hidden_units (tuple): Number of hidden units in each layer. Default: (300, 600).
        - learning_rate (float): Learning rate for training the model. Default: 0.0001.
        - tau (float): Soft update rate for the target network. Default: 0.001.
        """
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.tau = tau
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(state_size, hidden_units[0])  # First hidden layer
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])  # Second hidden layer
        self.fc3 = nn.Linear(hidden_units[1], action_size)  # Output layer
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
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
        self.optimizer.zero_grad()
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
# Example usage with BLER and Energy states (mean, max, min, skewness)
bler_state = [0.1, 0.2, 0.05, 0.1]  # Example BLER state (mean, max, min, skewness)
energy_state = [1.0, 1.5, 0.8, 0.05]  # Example Energy state (mean, max, min, skewness)

# Concatenate BLER and energy states
#state = bler_state + energy_state  # Total state size = 8
state = np.array(bler_state + energy_state)  # Shape: (6,)
state_size = len(state)  # 8
action_size = 2  # Example number of actions (weights)

# Create the actor network
actor = Actor(state_size, action_size)

# Example: Convert state to a PyTorch tensor and perform a forward pass
#state_tensor = torch.FloatTensor([state])  # Shape (1, 8)
state_tensor = torch.FloatTensor(state).unsqueeze(0).to("cpu")  # Add batch dimension if needed
actions = actor(state_tensor)  # Forward pass to get actions (weights)
print(actions)  # The actions (weights) will sum to 1

# Soft update target actor network from actor network
actor.update_target_model(actor)
'''