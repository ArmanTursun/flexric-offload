import torch
import torch.nn as nn
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=(300, 600), learning_rate=0.0001, tau=0.001):
        """
        Constructor for the Critic network in PyTorch

        Args:
        - state_size (int): Dimension of the input state (e.g., 8 for BLER and energy).
        - action_size (int): Dimension of the input action (e.g., 3 for weights).
        - hidden_units (tuple): Number of hidden units in each layer. Default: (300, 600).
        - learning_rate (float): Learning rate for training the model. Default: 0.0001.
        - tau (float): Soft update rate for the target network. Default: 0.001.
        """
        super(Critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.tau = tau
        
        # State pathway
        self.fc1_state = nn.Linear(state_size, hidden_units[0])
        
        # Action pathway
        self.fc1_action = nn.Linear(action_size, hidden_units[0])

        # Combined pathway
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)  # Output a single Q-value
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        """
        Forward pass for the Critic network.

        Args:
        - state (torch.Tensor): Input state tensor of shape (batch_size, state_size).
        - action (torch.Tensor): Input action tensor of shape (batch_size, action_size).

        Returns:
        - q_value (torch.Tensor): Output Q-value tensor of shape (batch_size, 1).
        """
        # Pass state through the state pathway
        state_out = torch.relu(self.fc1_state(state))
        
        # Pass action through the action pathway
        action_out = torch.relu(self.fc1_action(action))

        # Combine state and action outputs by adding them
        combined = state_out + action_out
        
        # Pass through the combined layers
        x = torch.relu(self.fc2(combined))
        q_value = self.fc3(x)  # Output a single Q-value

        return q_value

    def train(self, states, actions, q_targets):
        """
        Trains the Critic network.

        Args:
        - states (torch.Tensor): Input states of shape (batch_size, state_size).
        - actions (torch.Tensor): Input actions of shape (batch_size, action_size).
        - q_targets (torch.Tensor): Target Q-values of shape (batch_size, 1).

        Returns:
        - loss_value (float): The loss value after training.
        """
        self.optimizer.zero_grad()
        q_values = self.forward(states, actions)
        loss = nn.MSELoss()(q_values, q_targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _soft_update(self, target_model, tau=None):
        """
        Soft update of the target model parameters using the main model's parameters.
        
        Args:
        - target_model (Critic): The target critic model to be updated.
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
state = bler_state + energy_state  # Total state size = 8
state_size = len(state)  # 8
action_size = 2  # Example number of actions (weights)

# Create the critic network
critic = Critic(state_size, action_size)

# Create a separate target critic network
target_critic = Critic(state_size, action_size)

# Example: Convert state and action to PyTorch tensors and perform a forward pass
state_tensor = torch.FloatTensor([state])  # Shape (1, 8)
action_tensor = torch.FloatTensor([[0.45, 0.55]])  # Example action (weights), shape (1, 3)

# Get Q-value from the critic network
q_value = critic(state_tensor, action_tensor)
print(q_value)

# Soft update target critic network from the critic network
critic.update_target_model(target_critic)
'''