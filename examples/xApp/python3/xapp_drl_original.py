import xapp_sdk as ric
import time
import threading
from aggr_data import AggrData

#############################
#### Global Variables
#############################

# Global dictionary to store bler and energy data
# It keeps an window of 1000 of most recent data
global_ue_aggr_data = AggrData(1000)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()

##################################################
#### MACCallback Class
##################################################

class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)

    def handle(self, ind):
        # save bler and energy values of each tbs of each ue into aggr_data
        if len(ind.ue_stats) > 0:
            for i in range(ind.len_ue_stats):
                ue_stats = ind.ue_stats[i]

                # Calculate average energy from the TBS data
                total_energy = 0
                for tbs_stat in ue_stats.tbs[:ue_stats.num_tbs]:
                    total_energy += tbs_stat.tbs
                avg_energy = total_energy / ue_stats.num_tbs if ue_stats.num_tbs > 0 else 0

                # Add the new BLER and energy data to the AggrData object
                with global_lock:
                    global_ue_aggr_data.add_bler(ue_stats.context.ul_bler, ind.tstamp)
                    global_ue_aggr_data.add_energy(avg_energy, ind.tstamp)

# fill mac_ctrl_msg with data
def fill_mac_ctrl_msg():
    msg = ric.mac_ctrl_msg_t()
    msg.action = 42
    msg.tms = time.time_ns() / 1000.0
    msg.num_ues = 3
    ues = ric.mac_ue_ctrl_array(msg.num_ues) ## define array and it's lenth
    for i in range(msg.num_ues):
        ues[i].rnti = i
        ues[i].offload = 1
    msg.ues = ues
    return msg
#ctrl = fill_mac_ctrl_msg(self.ldpc_offload)
#ric.control_mac_sm(conn[i].id, ctrl)


#################################
#### DRL main method
#################################

def run_drl(stop_event):   
    while not stop_event.is_set():
        time.sleep(1)  
        with global_lock:
            print("BLER Stats:", global_ue_aggr_data.aggr_bler['mean'])
            print("Energy Stats:", global_ue_aggr_data.aggr_enrg['mean'])


####################
####  init RIC
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

##############################
#### MAC IND&CTRL with DRL
##############################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)

try:
    # Create a stop event for the drl thread
    stop_event = threading.Event()

    # Start the drl thread using the global dictionary
    drl_thread = threading.Thread(target=run_drl, args=(stop_event,))
    drl_thread.daemon = True  # Ensures the thread exits when the main program exits
    drl_thread.start()

    # Simulate main program running for a long time or until Ctrl+C is pressed
    time.sleep(1000)

except KeyboardInterrupt:
    print("Stopping drl and cleaning up...")

    # Set the stop event to stop the drl thread
    stop_event.set()

    # Wait for the drl thread to finish
    drl_thread.join()

    for i in range(0, len(mac_hndlr)):
        ric.rm_report_mac_sm(mac_hndlr[i])

    # Avoid deadlock. ToDo revise architecture 
    while ric.try_stop == 0:
        time.sleep(1)

    print("Test finished")

'''
import os

import tensorflow as tf
import numpy as np
from collections import deque
label = 'DDPG_model'
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
import var
winit = tf.contrib.layers.xavier_initializer()
binit = tf.constant_initializer(0.01)
class CriticNetwork(object):
    def __init__(self, sess, s_dim, a_dim, learning_rate=1e-3, tau=1e-3, gamma=0.995, hidden_unit_size=64):

        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden_unit_size = hidden_unit_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.seed = 0

        # Create the critic network
        self.inputs, self.action, self.out = self.buil_critic_nn(scope='critic')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.buil_critic_nn(scope='target_critic')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                                             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.losses.huber_loss(self.out,self.predicted_q_value, delta=0.5)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)
        if var.opt == 2:
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=var.learning_rate, momentum=0.95,
                                                      epsilon=0.01).minimize(self.loss)
        elif var.opt == 0:
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=var.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def buil_critic_nn(self, scope='network'):
        hid1_size = self.hidden_unit_size
        hid2_size = self.hidden_unit_size
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            state = tf.placeholder(name='c_states', dtype=tf.float32, shape=[None, self.s_dim])
            action = tf.placeholder(name='c_action', dtype=tf.float32, shape=[None, self.a_dim])

            net = tf.concat([state, action], 1)

            net1 = tf.layers.dense(inputs=net, units=1000, activation="linear",
                                   kernel_initializer=tf.zeros_initializer(),
                                   name='anet1')

            net2 = tf.layers.dense(inputs=net1, units=520, activation="relu", kernel_initializer=tf.zeros_initializer(),
                                   name='anet2')
            net3 = tf.layers.dense(inputs=net2, units=220, activation="linear", kernel_initializer=tf.zeros_initializer(),
                                   name='anet3')


            out = tf.layers.dense(inputs=net3, units=1,
                                  kernel_initializer=tf.zeros_initializer(),
                                  name='anet_out')
            out = (tf.nn.softsign(out))


        return state, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    def return_loss(self, predict, inputs, action):
        return self.sess.run(self.loss ,feed_dict={
        self.predicted_q_value: predict, self.inputs: inputs,
            self.action: action})




class ActorNetwork(object):
    def __init__(self, sess, s_dim, a_dim,lr=1e-4,  tau=1e-3, batch_size=64,action_bound=1):

        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.act_min = 0
        self.act_max = 51

        self.hdim = 64
        self.lr = lr

        self.tau = tau  # Parameter for soft update
        self.batch_size = batch_size

        self.seed = 0
        self.action_bound = action_bound
        # Actor Network
        self.inputs, self.out , self.scaled_out =  self.create_actor_network(scope='actor')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(scope='target_actor')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')

        # Parameter Updating Operator
        self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
            for i in range(len(self.target_network_params))]
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        # Gradient will be provided by the critic network
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
        #         self.actor_gradients = list(map(lambda x: x/self.batch_size, self.unnormalized_actor_gradients))
        self.actor_gradients = [unnz_actor_grad / self.batch_size for unnz_actor_grad in
                                self.unnormalized_actor_gradients]

        # Optimizer
        self.optimize = tf.train.AdamOptimizer(-self.lr).apply_gradients(zip(self.actor_gradients, self.network_params))


        if var.opt == 2:
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=var.learning_rate, momentum=0.95, epsilon=0.01). \
                apply_gradients(zip(self.unnormalized_actor_gradients, self.network_params))
        elif var.opt == 0:
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=var.learning_rate). \
                apply_gradients(zip(self.unnormalized_actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, scope='network'):
        hid1_size = self.hdim
        hid2_size = self.hdim

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            state = tf.placeholder(name='a_states', dtype=tf.float32, shape=[None, self.s_dim])

            net1 = tf.layers.dense(inputs=state, units=1500, activation="linear", kernel_initializer=tf.zeros_initializer(),
                                  name='anet1')


            net2 = tf.layers.dense(inputs=net1, units=1250, activation="relu",kernel_initializer=tf.zeros_initializer(), name='anet2')



            out = tf.layers.dense(inputs=net2, units=self.a_dim, kernel_initializer=tf.zeros_initializer(),
                                   name='anet_out')
            out=(tf.nn.sigmoid(out))
            scaled_out = tf.multiply(out, self.action_bound)
            # out = tf.nn.tanh(out)
        return state, out,scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def save_models(self, sess, model_path):
        """ Save models to the current directory with the name filename """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "DDPGmodel" + str(var.n_vehicle) + "/" + model_path)
        saver = tf.train.Saver(max_to_keep=var.n_vehicle * var.n_neighbor)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        saver.save(sess, model_path, write_meta_graph=True)

    def save(self):
        print('Training Done. Saving models...')
        model_path = label + '/agent_'
        print(model_path)
        self.save_models(self.sess, model_path)

    def load_models(self, sess, model_path):
        """ Restore models from the current directory with the name filename """
        dir_ = os.path.dirname(os.path.realpath(__file__))
        saver = tf.train.Saver(max_to_keep=var.n_vehicle * var.n_neighbor)
        model_path = os.path.join(dir_, "DDPGmodel" + str(var.n_vehicle) + "/" + model_path)
        saver.restore(self.sess, model_path)

    def load(self,sess):
        print("\nRestoring the model...")
        model_path = label + '/agent_'
        self.load_models(sess, model_path)
'''
'''
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.input_size = 10 + K * 2
        self.output_size = 1 + 1
        self.fc1 = nn.Linear(self.input_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
        self.fc4 = nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4)
        self.fc5 = nn.Linear(HIDDEN_SIZE_4, self.output_size)
        # init weight
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.constant_(self.fc5.bias, 0)


    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x[0][0] = torch.sigmoid(x[0][0])
        x[0][1] = torch.tanh(x[0][1])
        return x
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.input_size = 10 + K * 2 + 2
        self.output_size = 1
        self.fc1 = nn.Linear(self.input_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
        self.fc4 = nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4)
        self.fc5 = nn.Linear(HIDDEN_SIZE_4, self.output_size)
        # init weight
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.constant_(self.fc5.bias, 0)

    def forward(self, state,action):
        x = torch.cat([state, action], 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        return x
policy_Q = self.critic(state_batch, self.actor(state_batch))
actor_loss = -policy_Q.mean()

next_action_batch = self.target_actor(next_state_batch)
target_Q = self.target_critic(next_state_batch,next_action_batch.detach())
label_Q = reward_batch + GAMMA * target_Q
policy_Q_ = self.critic(state_batch, action_batch)
#critic_loss = ((label_Q - policy_Q_) ** 2).mean()
critic_loss = self.value_criterion(label_Q, policy_Q_.detach())
'''
