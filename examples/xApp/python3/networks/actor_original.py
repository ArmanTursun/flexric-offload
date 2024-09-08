import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, Identity
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class Actor(object):
    def __init__(self, state_size, action_size,
                 hidden_units=(300, 600), learning_rate=0.0001, batch_size=64,
                 tau=0.001):
        """
        Constructor for the Actor network

        :param state_size: An integer denoting the dimensionality of the states
            in the current problem
        :param action_size: An integer denoting the dimensionality of the
            actions in the current problem
        :param hidden_units: An iterable defining the number of hidden units in
            each layer. default: (300, 600)
        :param learning_rate: A float denoting the speed at which the network
            will learn. default: 0.0001
        :param batch_size: An integer denoting the batch size. default: 64
        :param tau: A float denoting the rate at which the target model will
            track the main model. default: 0.001
        """
        # Store parameters
        self._batch_size = batch_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units
        self._state_size = state_size
        self._action_size = action_size

        # Generate the main model
        self._model, self._model_weights, self._model_input = self._generate_model()
        # Generate carbon copy of the model so that we avoid divergence
        self._target_model, self._target_weights, self._target_state = self._generate_model()

        # Gradient calculation
        self._action_gradients = tf.keras.backend.function(
            inputs=[self._model.input],
            outputs=[self._model.output]
        )

        # Define the optimization function
        self._optimize = tf.keras.optimizers.Adam(learning_rate)

    def train(self, states, action_gradients):
        """
        Updates the weights of the main network
        :param states: The states of the input to the network
        :param action_gradients: The gradients of the actions to update the
            network
        :return: None
        """
        with tf.GradientTape() as tape:
            actions = self._model(states)
            loss = -tf.reduce_mean(actions * action_gradients)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimize.apply_gradients(zip(gradients, self._model.trainable_variables))

    def train_target_model(self):
        """
        Updates the weights of the target network to slowly track the main
        network.
        :return: None
        """
        main_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        target_weights = [self._tau * main_weight + (1 - self._tau) * target_weight 
                          for main_weight, target_weight in zip(main_weights, target_weights)]
        self._target_model.set_weights(target_weights)
        #print("Target model weights updated.")

    def _generate_model(self):
        """
        Generates the model based on the hyperparameters defined in the
        constructor.
        :return: A tuple containing references to the model, weights,
            and input layer
        """
        input_layer = Input(shape=[self._state_size])
        layer = Dense(self._hidden[0], activation='relu')(input_layer)
        layer = Dense(self._hidden[1], activation='relu')(layer)
        output_layer = Dense(self._action_size, activation='sigmoid')(layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model, model.trainable_weights, input_layer
