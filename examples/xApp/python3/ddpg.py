import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from networks.actor import Actor
from networks.critic import Critic
from ou_noise import OUNoise

class DDPG:
    def __init__(self, state_size, action_size, actor_hidden_units=(128, 128),
                 actor_learning_rate=0.0001, critic_hidden_units=(256, 256),
                 critic_learning_rate=0.001, batch_size=64, discount=0.99,
                 memory_size=10000, tau=0.001, gradient_clip_value=0.5):
        
        # Set the device to GPU if available, otherwise default to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Actor and Critic networks and move them to the device
        self.actor = Actor(state_size=state_size, action_size=action_size,
                           hidden_units=actor_hidden_units,
                           learning_rate=actor_learning_rate, tau=tau).to(self.device)

        self.critic = Critic(state_size=state_size, action_size=action_size,
                             hidden_units=critic_hidden_units,
                             learning_rate=critic_learning_rate, tau=tau).to(self.device)

        # Target networks
        self.target_actor = Actor(state_size=state_size, action_size=action_size,
                                  hidden_units=actor_hidden_units,
                                  learning_rate=actor_learning_rate, tau=tau).to(self.device)
        
        self.target_critic = Critic(state_size=state_size, action_size=action_size,
                                    hidden_units=critic_hidden_units,
                                    learning_rate=critic_learning_rate, tau=tau).to(self.device)

        self.actor.update_target_model(self.target_actor)  # Initialize target actor
        self.critic.update_target_model(self.target_critic)  # Initialize target critic

        # Memory buffer for experience replay
        self._memory = deque(maxlen=memory_size)

        # Batch size
        self._batch_size = batch_size
        self._discount = discount

        # Initialize OU noise for exploration with decay
        self.noise = OUNoise(action_size)

        # Critic loss history (optional for monitoring)
        self.critic_loss = 0
        self.actor_loss = 0

        # Gradient clipping value
        self.gradient_clip_value = gradient_clip_value

    def get_action(self, state):
        """
        Return action predicted by the actor network based on the current state,
        with added noise for exploration.
        """
        # Ensure state is a NumPy array
        if isinstance(state, list) or isinstance(state, tuple):
            state = np.array(state)  # Convert to NumPy array if needed

        # Convert NumPy array to PyTorch tensor and move it to the correct device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get the action from the actor network
        action = self.actor(state_tensor).cpu().detach().numpy()[0]
        
        # Add exploration noise and clip the action
        noise = self.noise.sample()        
        action_with_noise = np.clip(action + noise, 0, 1)

        # Re-normalize the actions to ensure their sum equals 1
        action_sum = np.sum(action_with_noise)
        if action_sum > 0:
            action_with_noise = action_with_noise / action_sum  # Normalize to sum to 1
        
        # Decay the noise after each action
        self.noise.decay_noise()
        
        return action_with_noise

    def reset_noise(self):
        """Reset the OU noise at the start of each episode."""
        self.noise.reset()

    def remember(self, state, action, reward, done, next_state):
        # Perform validity checks on the state, action, reward, and next_state
        if not self._is_valid_sample(state, action, reward, done, next_state):
            print("Invalid sample detected, skipping...")
            return  # Skip storing this invalid sample
        """Store experience in memory."""
        self._memory.append((state, action, reward, done, next_state))

    def train(self):
        """Train the agent if memory contains enough samples."""
        if len(self._memory) > self._batch_size:
            self._train()

    def _train(self):
        """Perform a training step: update critic and actor networks, then update target networks."""
        states, actions, rewards, done, next_states = self._get_sample()
        # Check for NaN values in the inputs
        if self._check_for_nan([states, actions, rewards, done, next_states]):
            print("NaN detected in inputs! Skipping training step.")
            return  # Skip training if NaNs are found
        self.critic_loss = self._train_critic(states, actions, next_states, done, rewards)
        self.actor_loss = self._train_actor(states)
        self._update_target_models()

    def _get_sample(self):
        """Sample a batch of experiences from memory for training."""
        sample = random.sample(self._memory, self._batch_size)
        states, actions, rewards, done, next_states = zip(*sample)

        states = np.array(states)  # Convert list of NumPy arrays to a single NumPy array
        actions = np.array(actions)
        rewards = np.array(rewards)
        done = np.array(done)
        next_states = np.array(next_states)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # Ensure reward is (batch_size, 1)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)  # Ensure done is (batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)

        return states, actions, rewards, done, next_states

    def _train_critic(self, states, actions, next_states, done, rewards):
        """Train the critic network using sampled states, actions, and calculated Q-targets."""
        q_targets = self._get_q_targets(next_states, done, rewards)
        critic_loss = self.critic.train(states, actions, q_targets)  # Train critic and return loss
        # Clip gradients for critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_value)
        return critic_loss

    def _get_q_targets(self, next_states, done, rewards):
        """Calculate Q-targets for critic network."""
        # Predict next actions using the target actor
        next_actions = self.target_actor(next_states)

        # Predict next Q-values using the target critic
        next_q_values = self.target_critic(next_states, next_actions).detach()

        # Calculate Q-targets: reward + discount * next Q value (for non-terminal states)
        q_targets = rewards + self._discount * next_q_values * (1 - done)

        return q_targets

    def _train_actor(self, states):
        """Train the actor network using policy gradient."""
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()  # Maximize Q-value for predicted actions
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        # Clip gradients before applying them
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_value)

        self.actor.optimizer.step()

        return actor_loss.item()  # Return the loss as a scalar value

    def _update_target_models(self):
        """Soft update target networks for actor and critic."""
        self.critic.update_target_model(self.target_critic)  # Update target critic model
        self.actor.update_target_model(self.target_actor)  # Update target actor model

    def _check_for_nan(self, tensors):
        """Check for NaN values in the list of tensors."""
        for tensor in tensors:
            if torch.isnan(tensor).any().item():
                return True
        return False
    
    def _is_valid_sample(self, state, action, reward, done, next_state):
        """
        Check if the sample (state, action, reward, done, next_state) is valid.
        This includes checking for NaN, Inf, and valid dimensions/ranges.
        """
        # Check for NaN or Inf in the state, action, next_state, reward, and done flag
        if (np.isnan(state).any() or np.isinf(state).any() or
            np.isnan(action).any() or np.isinf(action).any() or
            np.isnan(next_state).any() or np.isinf(next_state).any() or
            np.isnan(reward) or np.isinf(reward) or
            np.isnan(done) or np.isinf(done)):
            return False  # Invalid sample due to NaN or Inf values
    
        # Check if the state and next_state have the expected dimensions
        if len(state) != self.actor.state_size or len(next_state) != self.actor.state_size:
            return False  # Invalid sample due to incorrect state dimensions
    
        # Check if action is in the valid range [0, 1] (or other ranges if needed)
        if np.any(action < 0) or np.any(action > 1):
            return False  # Invalid action due to out-of-bounds values

        # Optionally, check if the reward is within a reasonable range (e.g., avoid extreme values)
        #if reward < -1000 or reward > 1000:  # Modify range based on your problem
            #return False  # Invalid sample due to extreme reward values
    
        return True  # Sample is valid

"""
# Test DDPG with random BLER and energy data
def test_ddpg_random_data():
    state_size = 8  # Assuming the state consists of 3 BLER values and 3 energy values
    action_size = 2  # Assume the action size is 2 for simplicity

    # Initialize DDPG agent
    agent = DDPG(state_size=state_size, action_size=action_size)

    num_episodes = 1  # Run for 5 episodes to test
    max_steps = 100  # Max steps per episode

    for episode in range(num_episodes):
        # Generate random initial state for BLER and energy
        bler_state = np.random.rand(4)  # Random values for [bler_mean, bler_max, bler_min]
        energy_state = np.random.rand(4)  # Random values for [energy_mean, energy_max, energy_min]
        state = np.concatenate((bler_state, energy_state))  # Combine into a single state array

        agent.reset_noise()  # Reset the OU noise for each episode

        total_reward = 0
        for step in range(max_steps):
            # Get action from the agent
            action = agent.get_action(state)

            # Simulate next state (generate random values for the next step)
            next_bler_state = np.random.rand(4)  # Random next BLER values
            next_energy_state = np.random.rand(4)  # Random next energy values
            next_state = np.concatenate((next_bler_state, next_energy_state))

            # Simulate reward (randomly generated for this example)
            reward = np.random.rand() * 10 - 5  # Random reward between -5 and 5

            # Randomly decide if the episode is done
            done = np.random.choice([True, False])

            # Store the experience in the agent's memory
            agent.remember(state, action, reward, done, next_state)

            # Train the agent
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break
        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


if __name__ == "__main__":
    test_ddpg_random_data()
"""
