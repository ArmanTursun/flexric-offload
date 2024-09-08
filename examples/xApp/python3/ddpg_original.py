import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic
from collections import deque
import numpy as np
import random
from ou_noise import OUNoise  # Import OU Noise class

class DDPG(object):
    def __init__(self, state_size, action_size, actor_hidden_units=(300, 600),
                 actor_learning_rate=0.0001, critic_hidden_units=(300, 600),
                 critic_learning_rate=0.001, batch_size=64, discount=0.99,
                 memory_size=10000, tau=0.001):
        self._discount = discount
        self._batch_size = batch_size
        self._memory_size = memory_size

        # Initialize Actor and Critic networks
        self._actor = Actor(state_size=state_size, action_size=action_size,
                            hidden_units=actor_hidden_units,
                            learning_rate=actor_learning_rate,
                            batch_size=batch_size, tau=tau)

        self._critic = Critic(state_size=state_size, action_size=action_size,
                              hidden_units=critic_hidden_units,
                              learning_rate=critic_learning_rate,
                              batch_size=batch_size, tau=tau)

        self._memory = deque(maxlen=memory_size)
        self.critic_loss_history = []  # Track critic loss per training step

        # Initialize OU noise for exploration with decay
        self.noise = OUNoise(action_size)

    def get_action(self, state):
        """
        Return action predicted by the actor network based on current state,
        plus noise for exploration.
        """
        state = np.expand_dims(state, axis=0)  # Ensure the input state is batch-like
        action = self._actor._model.predict(state)[0]  # Get the action from the actor
        noise = self.noise.sample()  # Get OU noise
        action_with_noise = np.clip(action + noise, 0, 1)  # Apply noise and clip action within valid range
        self.noise.decay_noise()  # Apply decay to the noise after each action
        return action_with_noise

    def reset_noise(self):
        """
        Reset the OU noise at the start of each episode.
        """
        self.noise.reset()

    def train(self):
        """
        Train the agent when memory contains enough samples.
        """
        if len(self._memory) > self._batch_size:
            self._train()


    def _train(self):
        """
        Perform a training step: update critic and actor networks, then update target networks.
        """
        states, actions, rewards, done, next_states = self._get_sample()
        critic_loss = self._train_critic(states, actions, next_states, done, rewards)
        self.critic_loss_history.append(critic_loss)  # Track critic loss
        self._train_actor(states)
        self._update_target_models()


    def _get_sample(self):
        """
        Get a batch sample from memory for training.
        """
        sample = random.sample(self._memory, self._batch_size)
        states, actions, rewards, done, next_states = zip(*sample)
    
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).astype(np.float32).flatten()  # Ensure rewards are flat floats
        done = np.array(done).astype(bool).flatten()  # Ensure done is a flat boolean array
        next_states = np.array(next_states)

        return states, actions, rewards, done, next_states


    def _train_critic(self, states, actions, next_states, done, rewards):
        """
        Train the critic network using sampled states, actions, and calculated Q-targets.
        """
        q_targets = self._get_q_targets(next_states, done, rewards)
        critic_loss = self._critic.train(states, actions, q_targets)  # Train critic and return loss
        return critic_loss


    def _get_q_targets(self, next_states, done, rewards):
        """
        Calculate Q-targets for critic network.
        """
        next_actions = self._actor._model.predict(next_states)  # Predict next actions using actor
        next_q_values = self._critic._model.predict([next_states, next_actions]).flatten()  # Predict Q-values

        # Calculate Q targets: reward + discount * next Q value (for non-terminal states)
        q_targets = np.array([
            reward if this_done else reward + self._discount * next_q_value 
            for reward, next_q_value, this_done in zip(rewards, next_q_values, done)
        ]).reshape(-1, 1)  # Reshape to (batch_size, 1)

        return q_targets


    def _train_actor(self, states):
        """
        Train the actor network using policy gradient.
        """
        predicted_actions = self._actor._model.predict(states)  # Get predicted actions
        gradients = self._critic.get_gradients(states, predicted_actions)  # Get gradients from critic
        self._actor.train(states, gradients)  # Update actor policy using gradients


    def _update_target_models(self):
        """
        Soft update target networks for actor and critic to follow the main networks.
        """
        self._critic.train_target_model()  # Update target critic model
        self._actor.train_target_model()  # Update target actor model


    def remember(self, state, action, reward, done, next_state):
        """
        Store experience in memory.
        """
        state = np.asarray(state).astype(np.float32)
        action = np.asarray(action).astype(np.float32)
        next_state = np.asarray(next_state).astype(np.float32)
        reward = float(reward)  # Ensure reward is a float
        done = bool(done)  # Ensure done is a boolean

        self._memory.append((state, action, reward, done, next_state))
