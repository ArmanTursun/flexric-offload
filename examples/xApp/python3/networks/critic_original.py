from keras.layers import Dense, Input, Add
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

class Critic(object):
    def __init__(self, state_size, action_size,
                 hidden_units=(300, 600), learning_rate=0.0001, batch_size=64,
                 tau=0.001):
        """
        Constructor for the Critic network
        """
        # Store parameters
        self._batch_size = batch_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units
        self._state_size = state_size
        self._action_size = action_size

        # Generate the main model
        self._model, self._state_input, self._action_input = self._generate_model()
        # Generate a carbon copy of the model so that we avoid divergence
        self._target_model, _, _ = self._generate_model()

        # Define the optimization function
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

    def get_gradients(self, states, actions):
        """
        Returns the gradients of the actions with respect to the Q-values.
        """
        # Convert NumPy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        #print(f"States shape (in get_gradients): {states.shape}")
        #print(f"Actions shape (in get_gradients): {actions.shape}")

        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self._model([states, actions])
        action_gradients = tape.gradient(q_values, actions)
        #print(f"Action gradients shape: {action_gradients.shape}")
        return action_gradients


    def train(self, states, actions, q_targets):
        """
        Trains the Critic network.
        """
        #print(f"Training Critic - States shape: {states.shape}")
        #print(f"Training Critic - Actions shape: {actions.shape}")
        #print(f"Training Critic - Q targets shape: {q_targets.shape}")

        with tf.GradientTape() as tape:
            q_values = self._model([states, actions])
            #print(f"Q values shape: {q_values.shape}")  # Add debug for q_values

            # Ensure q_targets and q_values have the same shape
            if q_targets.shape != q_values.shape:
                #print(f"Reshaping Q targets from {q_targets.shape} to {q_values.shape}")
                q_targets = tf.reshape(q_targets, q_values.shape)

            loss = tf.keras.losses.MeanSquaredError()(q_targets, q_values)
    
        critic_gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(critic_gradients, self._model.trainable_variables))
        return loss.numpy() 
    
    def train_target_model(self):
        """
        Updates the weights of the target network to slowly track the main network.
        """
        main_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        target_weights = [self._tau * main_weight + (1 - self._tau) * target_weight 
                          for main_weight, target_weight in zip(main_weights, target_weights)]
        self._target_model.set_weights(target_weights)
        #print("Target model weights updated.")

    def _generate_model(self):
        """
        Generates the model based on the hyperparameters defined in the constructor.
        """
        state_input_layer = Input(shape=[self._state_size])
        action_input_layer = Input(shape=[self._action_size])

        # State pathway
        s_layer = Dense(self._hidden[0], activation='relu')(state_input_layer)

        # Action pathway
        a_layer = Dense(self._hidden[0], activation='linear')(action_input_layer)

        # Combine state and action pathways
        combined = Add()([s_layer, a_layer])
        hidden = Dense(self._hidden[1], activation='relu')(combined)

        # Output layer
        output_layer = Dense(1, activation='linear')(hidden)

        # Create the model
        model = Model(inputs=[state_input_layer, action_input_layer], outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))

        return model, state_input_layer, action_input_layer
