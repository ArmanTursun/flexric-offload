import time
import numpy as np
import csv
import os
import sys
from datetime import datetime
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torch.autograd.functional import hessian
import torch.nn as nn
import pandas as pd
import seaborn as sns
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy
import threading
from aggr_data import AggrData

cur_dir = os.path.dirname(os.path.abspath(__file__))
# print("Current Directory:", cur_dir)
sdk_path = cur_dir + "/../xapp_sdk/"
sys.path.append(sdk_path)
import xapp_sdk as ric


class ThompsonSampling(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, is_linear, eta):
        self.eta = eta
        self.device = device
        self.dimension = dimension
        self.V_inv =  self.eta * cov_prior
        self.bt = torch.diag(1/(self.eta * torch.diag(cov_prior))) @  mean_prior

    def get_config(self):
        return {'eta': self.eta, 'dimension': self.dimension, 'algorithm': 'TS'}
        
    def sample_posterior(self):
        mean = self.V_inv @ self.bt
        cov = self.V_inv / self.eta
        return MultivariateNormal(mean, covariance_matrix=cov).sample()
        
    def reward(self, context):
        theta = self.sample_posterior()
        return context.dot(theta).item()
        
    def update(self, context, action, reward):
        self.bt += reward * context
        omega = self.V_inv @ context
        self.V_inv -= omega[:, None] @ omega[None, :] / (1 + omega.dot(context))

class VITS(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, is_linear, eta, h, nb_updates, approx, hessian_free, mc_samples):
        self.eta = eta
        self.device = device
        self.dimension = dimension
        self.approx = approx
        self.hessian_free = hessian_free
        self.mc_samples = mc_samples
        self.context = torch.empty((0, dimension)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.linear = is_linear
        self.h = h
        self.nb_updates = nb_updates
        self.mean_prior = mean_prior
        #self.mean = self.mean_prior.to(self.device)
        self.mean = torch.tensor(mean_prior, dtype=torch.float32).to(self.device)
        self.cov_prior_inv = torch.diag(1/(self.eta * torch.diag(cov_prior))).to(self.device)
        self.cov_semi = torch.diag(torch.sqrt(torch.diag(cov_prior))).to(self.device)
        self.cov_semi_inv = torch.diag(1/torch.sqrt(torch.diag(cov_prior))).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
    def update_lr(self):
        if (self.h > 0.0001):
            self.h /= 10.0
    
    def update_eta(self):
        if (self.eta < 1000):
            self.eta *= 10
    
    def get_config(self):
        return {'eta': self.eta, 'dimension': self.dimension, 'h': self.h,
                'nb_updates': self.nb_updates, 'algorithm': 'VITS'}
   
    def sample_posterior(self):
        eps = torch.normal(0, 1, size=(self.dimension,)).to(self.device)
        theta = self.mean + self.cov_semi @ eps
        return theta
        
    def reward(self, context):
        theta = self.sample_posterior()
        return context.dot(theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.context @ theta - self.rewards))
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
            #return torch.sum(torch.square(self.context @ theta - self.rewards))
        else:
            data_term = self.criterion(self.context @ theta, self.rewards)
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
    
    def compute_gradient_hessian(self):
        gradients = torch.zeros((self.mc_samples, self.dimension)).to(self.device)
        hessian_matrices = torch.zeros((self.mc_samples, self.dimension, self.dimension)).to(self.device)
        for idx in range(self.mc_samples):
            theta = self.sample_posterior()
            theta.requires_grad = True
            #current_grad = (self.eta / 2) * (grad(self.potential(theta), theta)[0] + self.ldb * theta)
            current_grad = grad(self.potential(theta), theta)[0]
            gradients[idx, :] = current_grad
            if self.hessian_free:
                theta.requires_grad = False
                hessian_matrices[idx, :, :] = ((self.cov_semi_inv.T @ self.cov_semi_inv) @ (theta - self.mean)[:, None] @ current_grad[None, :])
            else:
                hessian_matrices[idx, :, :] = hessian(self.potential, theta)
            del theta
        return gradients.mean(0), hessian_matrices.mean(0)
  
    def update_cov(self, h, hessian_matrix):
        cov_semi = (torch.eye(self.dimension).to(self.device) - h * hessian_matrix) @ self.cov_semi + h * self.cov_semi_inv.T
        if self.approx:
            cov_semi_inv = self.cov_semi_inv @ (torch.eye(self.dimension).to(self.device) - h * (self.cov_semi_inv.T @ self.cov_semi_inv - hessian_matrix))
        else:
            cov_semi_inv = torch.linalg.pinv(cov_semi)
        return cov_semi, cov_semi_inv

    def update(self, context, action, reward):
        #print(self.context)
        #print(context)
        #self.context = torch.cat([self.context, torch.tensor(context, dtype=torch.float32)[None, :].to(self.device)])
        self.context = torch.cat([self.context, context[None, :].to(self.device)])
        #self.context = torch.cat([self.context, context.unsqueeze(0).to(self.device)])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / (len(self.rewards) * self.eta)
        for _ in range(self.nb_updates):
            gradient, hessian_matrix = self.compute_gradient_hessian()
            self.mean -= h * gradient
            self.cov_semi, self.cov_semi_inv = self.update_cov(h, hessian_matrix)
            assert not torch.isnan(self.cov_semi).any(), "cov_semi contains NaN"
            assert not torch.isnan(self.cov_semi_inv).any(), "cov_semi_inv contains NaN"
            del gradient, hessian_matrix


class Langevin(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, is_linear, eta, h, nb_updates):
        self.eta = eta
        self.device = device
        self.dimension = dimension
        self.mean_prior = mean_prior
        self.cov_prior = cov_prior
        self.cov_prior_inv = torch.diag(1/(self.eta * torch.diag(cov_prior))).to(self.device)
        self.context = torch.empty((0, dimension)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.linear = is_linear
        self.h = h
        self.nb_updates = nb_updates

    def get_config(self):
        return {'eta': self.eta, 'dimension': self.dimension, 'h': self.h,
                'nb_updates': self.nb_updates, 'algorithm': 'lmcts'}

    def reward(self, context):
        if len(self.rewards) == 0:
            self.theta = self.mean_prior + torch.diag(1/torch.sqrt(torch.diag(self.cov_prior))) @ torch.normal(0, 1, size=(self.dimension,)).to(self.device)
        else:
            h = self.h / (1 + len(self.rewards))
            for _ in range(10):
                gradient = self.compute_gradient()
                self.theta += - h * gradient + torch.normal(0, np.sqrt(2 * h), size=gradient.shape).to(self.device)
        return context.dot(self.theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.context @ theta - self.rewards))
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
        else:
            data_term = torch.nn.BCEWithLogitsLoss()(self.context @ theta, self.rewards)
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
    
    def compute_gradient(self):
        self.theta.requires_grad = True
        gradient = grad(self.potential(self.theta), self.theta)[0]
        self.theta.requires_grad = False
        return gradient
    
    def update(self, context, action, reward):
        self.context = torch.cat([self.context, torch.tensor(context, dtype=torch.float32).to(self.device)[None, :]])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / (len(self.rewards))
        for _ in range(self.nb_updates):
            gradient = self.compute_gradient()
            self.theta += - h * gradient + torch.normal(0, np.sqrt(2 * h), size=gradient.shape).to(self.device)
            del gradient

class Encoder:
    def __init__(self, action_space, embedding_dim=10):
        """
        Encoder for actions and contexts.

        :param action_space: List of all possible actions ([MCS, PRB]).
        :param embedding_dim: The desired embedding dimension (default: 10).
        """
        self.embedding_dim = embedding_dim
        
        # Action embeddings: Precompute embeddings for all actions
        self.action_space = torch.tensor(action_space, dtype=torch.long)
        self.num_actions = len(action_space)
        self.action_embedding = nn.Embedding(self.num_actions, embedding_dim)
        self.action_embeddings = self.action_embedding(
            torch.arange(self.num_actions)
        )  # Precompute embeddings for actions
        
        # Context embeddings: Dynamically encode contexts
        self.context_embedding = nn.EmbeddingBag(
            num_embeddings=1000,  # Initial size for unique contexts (expandable)
            embedding_dim=embedding_dim,
            mode="mean"
        )
        self.context_index = {}  # Store unique context indices
        self.context_counter = 0  # To assign new indices dynamically

    def encode_action(self, action):
        """
        Encode a given action ([MCS, PRB]).
        :param action: [MCS, PRB] pair as a list or tensor.
        :return: Encoded 10-dimensional action embedding.
        """
        action = torch.tensor(action, dtype=torch.long)
        action_idx = (self.action_space == action).all(dim=1).nonzero(as_tuple=True)[0]
        return self.action_embeddings[action_idx]

    def encode_context(self, context):
        """
        Dynamically encode a given context ([SNR, Demand]).
        :param context: [SNR, Demand] pair as a list or tensor.
        :return: Encoded 10-dimensional context embedding.
        """
        context = tuple(context)  # Convert to tuple for dictionary key
        if context not in self.context_index:
            # Add new context dynamically
            self.context_index[context] = self.context_counter
            self.context_counter += 1
            
            # Expand embedding bag if needed
            if self.context_counter > self.context_embedding.num_embeddings:
                self.expand_context_embedding()

        # Get the index of the context and return its embedding
        context_idx = self.context_index[context]
        context_idx_tensor = torch.tensor([context_idx], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        return self.context_embedding(context_idx_tensor, offsets).squeeze(0).detach()

    def expand_context_embedding(self):
        """
        Expand the context embedding layer dynamically when the number of contexts exceeds its current capacity.
        """
        new_size = self.context_counter + 100  # Add buffer for new contexts
        new_context_embedding = nn.EmbeddingBag(
            num_embeddings=new_size,
            embedding_dim=self.embedding_dim,
            mode="mean",
        )
        # Copy weights from the old embedding
        new_context_embedding.weight.data[: self.context_embedding.num_embeddings] = (
            self.context_embedding.weight.data
        )
        self.context_embedding = new_context_embedding


parser = argparse.ArgumentParser(description="MAC Control")
parser.add_argument('--do_train', default=True, action=argparse.BooleanOptionalAction, help="Does it train? (default: True)")
parser.add_argument('--aggr_size', type=int, default=5, help="Size of aggregated data (default: 5)")
parser.add_argument('--dimension', type=int, default=10, help="Dimension of embeded data (default: 10)")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument('--algo1', default='vits', help="Choose algorithm 1 (default: vits)")
parser.add_argument('--algo2', default='baseline', help="Choose algorithm 2 (default: baseline)")
parser.add_argument('--lr', type=float, default=0.01, help="Learning Rate (default: 0.01)")
parser.add_argument('--linear', default=False, action=argparse.BooleanOptionalAction, help="Is linear? (default: False)")
parser.add_argument('--step', type=int, default=1000, help="Total training steps (default: 1000)")
parser.add_argument('--eta', type=int, default=500, help="Eta (default: 500)")
parser.add_argument('--nb_updates', type=int, default=10, help="Number of gradient steps K_t (default: 10)")
parser.add_argument('--approx', default=False, action=argparse.BooleanOptionalAction, help="Whether to use approximation (default: False)")
parser.add_argument('--hessian_free', default=False, action=argparse.BooleanOptionalAction, help="Is hessian free? (default: False)")
parser.add_argument('--mcs_low', type=int, default=9, help="lower bound of mcs (default: 9)")
parser.add_argument('--mcs_high', type=int, default=29, help="upper bound of mcs (default: 29)")
parser.add_argument('--mcs_step', type=int, default=2, help="step of mcs values (default: 2)")
parser.add_argument('--prb_low', type=int, default=5, help="lower bound of prb (default: 5)")
parser.add_argument('--prb_high', type=int, default=107, help="upper bound of prb (default: 107)")
parser.add_argument('--prb_step', type=int, default=5, help="step of prb values (default: 5)")
parser.add_argument('--report_time', type=int, default=1, help="E2 reports time frequency (default: 1)")
parser.add_argument('--rounds', type=int, default=10, help="Total rounds that test runs (default: 10)")
parser.add_argument('--scale_cqi', type=int, default=15, help="upper bound of cqi (default: 15)")
parser.add_argument('--scale_demand', type=int, default=300000, help="upper bound of demand (default: 300000)")
parser.add_argument('--scale_tbs', type=int, default=300000, help="upper bound of tbs (default: 300000)")
parser.add_argument('--scale_power', type=int, default=50, help="upper bound of power (default: 50)")
parser.add_argument('--scale_snr', type=int, default=51, help="upper bound of snr (default: 51)")
parser.add_argument('--power_penalty', type=float, default=2.0, help="penalty factor of power (default: 2.0)")
parser.add_argument('--bler_penalty', type=float, default=2.0, help="penalty factor of bler (default: 2.0)")
parser.add_argument('--a', type=int, default=1, help="a in reward function a * np.exp(b * x) (default: 1)")
parser.add_argument('--b', type=int, default=2, help="b in reward function a * np.exp(b * x) (default: 2)")
args = parser.parse_args()

mac_hndlr = []

global_ue_aggr_data = AggrData(max_length=args.aggr_size, datasets=['cqi', 'ul_demand', 'ul_throughput', 'power', 'ul_snr', 'ul_bler', 'ul_prb', 'ul_mcs', 'prate'])

# Global lock to ensure thread-safe access to the global dictionary
global_lock_data = threading.Lock()
global_lock_offload = threading.Lock()

class vran(object):
    def __init__(self, do_train, algo_name, conn, dimension, device, is_linear, Agent, hyperparameters, T, seed, action_ranges, scales, power_penalty, bler_penalty, a, b):
        self.dimension = dimension
        self.device = device
        self.is_linear = is_linear
        self.Agent = Agent
        self.hyperparameters = hyperparameters
        self.T = T
        self.seed = seed
        self.conn = conn
        self.action_ranges = action_ranges
        self.ucqi, self.udemand, self.utbs, self.upower, self.usnr = scales
        self.do_train = do_train
        self.mcs_high = 28
        self.prb_high = 106
        self.power_penalty = power_penalty
        self.bler_penalty = bler_penalty
        self.reward_a = a
        self.reward_b = b

        print('Train with ', algo_name)
        # Generate all possible actions (MCS and PRB)
        actions, normalized_actions = self.generate_actions(*self.action_ranges)
        self.actions_original = torch.tensor(actions, dtype=torch.float32).to(self.device)
        self.encoder = Encoder(normalized_actions, self.dimension)
        self.actions = self.encoder.action_embeddings.detach().to(self.device)

        # initialize agents
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.mean_prior = torch.mean(self.actions, axis=0)
        self.cov_prior = torch.diag(torch.var(self.actions, axis=0))
        self.agents = [self.Agent(self.dimension, self.mean_prior, self.cov_prior, self.device, self.is_linear, *self.hyperparameters) for _ in range(len(self.actions))]
        print("Initilized agents")
        if not self.do_train:
            print('Run original scheduler')
    
    # Checking Action Sampling Bias
    def check_action_sampling_bias(self):
        for agent in self.agents:
            theta = agent.sample_posterior()
            rewards = self.actions @ theta.cpu().numpy()
            top_action_indices = np.argsort(rewards)[-10:]  # Top 10 actions
            top_actions = self.actions.cpu().numpy()[top_action_indices]
            print("Top Actions based on Sampled Posterior:")
            print(top_actions)

    def scalemcsprb(self, value, min_val, max_val):
        return round((value - min_val) / (max_val - min_val), 2)
    
    def generate_actions(self, mcs_low, mcs_high, mcs_step, prb_low, prb_high, prb_step):
        """
        Generate all possible actions (MCS and PRB combinations).
        :return: Array of actions.
        """
        MCS_range = list(range(mcs_low, mcs_high+1, mcs_step))  # MCS values [mcs_low, mcs_high]
        PRB_range = list(range(prb_low, prb_high+1, prb_step))  # PRB values [prb_low, prb_high]
        if PRB_range[-1] != prb_high:
            PRB_range[-1] = prb_high
        if MCS_range[-1] != mcs_high:
            MCS_range[-1] = mcs_high
        self.mcs_high = mcs_high
        self.prb_high = prb_high
        actions = [[mcs, prb] for mcs in MCS_range for prb in PRB_range]
        normalized_actions = [[self.scalemcsprb(mcs, mcs_low, mcs_high), self.scalemcsprb(prb, prb_low, prb_high)] for mcs in MCS_range for prb in PRB_range]
        return actions, normalized_actions
    
    def cal_reward(self, observe):
        observe_tbs, context_demand, observe_prate, observe_pwr, observe_mcs, observe_prb, action_mcs, action_prb = observe
        tbs_reward = ((self.utbs - observe_tbs) / abs(context_demand - observe_tbs)) if abs(context_demand - observe_tbs) > 0 else (self.utbs - observe_tbs)
        u = np.log(1 + tbs_reward)
        reward = u - self.power_penalty * np.log(1 + observe_pwr)

        return reward
    
    def choose_action(self, context, agents):
        rewards = [agent.reward(context) for agent in agents]
        selected_action = np.argmax(rewards)
        best_expected_reward = max(rewards)
        return selected_action , best_expected_reward
    
    def create_conf(self, rnti, mcs, prb, add):
        conf = ric.mac_conf_t()
        conf.isset_pusch_mcs = add
        conf.pusch_mcs = mcs
        conf.pusch_prb = prb
        conf.rnti = rnti
        return conf

    def send_action(self, action):
        msg = ric.mac_ctrl_msg_t()
        msg.ran_conf_len = 1
        confs = ric.mac_conf_array(1)
        mcs, prb = action if self.do_train else (self.mcs_high, self.prb_high)
        add = True #if self.do_train else False
        for i in range(0, msg.ran_conf_len):
            confs[i] = self.create_conf(i, int(mcs), int(prb), add)
        
        msg.ran_conf = confs
        ric.control_mac_sm(self.conn[0].id, msg)
    
    def min_max_scale(self, value, min_val, max_val):
        if value > min_val:
            return round((value - min_val) / (max_val - min_val), 5)
        else:
            return round((min_val - min_val) / (max_val - min_val), 5)

    def scale_features(self, cqi, demand, tbs, power, snr, mcs, prb):
        scaled_cqi = self.min_max_scale(cqi, 0, self.ucqi)  # Scale CQI to [0, 1]
        scaled_demand = round(self.min_max_scale(demand, 0, self.udemand), 5)  # Scale Demand to [0, 1]
        scaled_tbs = round(self.min_max_scale(tbs, 0, self.utbs), 5)  # Scale TBS to [0, 1]
        scaled_power = round(self.min_max_scale(power, 0, self.upower), 2)   # Scale Power to [0, 1]
        scaled_snr = round(self.min_max_scale(snr, 8, self.usnr), 1)
        scaled_mcs = self.min_max_scale(mcs, 0, self.mcs_high)
        scaled_prb = self.min_max_scale(prb, 0, self.prb_high)
        return scaled_cqi, scaled_demand, scaled_tbs, scaled_power, scaled_snr, scaled_mcs, scaled_prb

    def gen_noise(self, scaled_mcs, scaled_prb):
        # Generate custom non-linear noise
        raw_noise = np.sin(np.random.uniform(-1, 1)) + np.power(np.random.uniform(-1, 1), 2)
        scaled_noise = 0.1 * raw_noise / max(abs(raw_noise), 1e-8)
        noise = np.clip(scaled_noise, 0, scaled_mcs * scaled_prb)

        return noise
    
    def compute(self):
        cumulative_regret = torch.zeros((self.T,))
        average_regret = torch.zeros((self.T,))
        rwd = torch.zeros((self.T,))
        amcs = torch.zeros((self.T,))
        aprb = torch.zeros((self.T,))        
        mcs = torch.zeros((self.T,))
        prb = torch.zeros((self.T,))
        snr = torch.zeros((self.T,))
        dmnd = torch.zeros((self.T,))
        pwr = torch.zeros((self.T,))
        tbs = torch.zeros((self.T,))
        action = len(self.actions) - 1
        mcs_action, prb_action = self.mcs_high, self.prb_high
        scaled_mcs_action = self.min_max_scale(mcs_action, 9, self.mcs_high)
        scaled_prb_action = self.min_max_scale(prb_action, 6, self.prb_high)
        start_time = time.time()
        for t in range(self.T):
            with global_lock_data:
                # context
                context_cqi = global_ue_aggr_data.get_stats('cqi')['mean']               
                context_snr = global_ue_aggr_data.get_stats('ul_snr')['mean']
                # observation
                if t == 0:
                    observe_tbs = global_ue_aggr_data.get_stats('ul_throughput')['mean']
                    context_demand = global_ue_aggr_data.get_stats('ul_demand')['mean']
                    observe_pwr = global_ue_aggr_data.get_stats('power')['mean']
                    observe_prate = global_ue_aggr_data.get_stats('prate')['mean']
                    observe_prb = global_ue_aggr_data.get_stats('ul_prb')['mean']
                    observe_prb_min = global_ue_aggr_data.get_stats('ul_prb')['min']
                    observe_prb_max = global_ue_aggr_data.get_stats('ul_prb')['max']
                else:
                    observe_tbs = global_ue_aggr_data.get_stats('ul_throughput')['mean']
                    context_demand = global_ue_aggr_data.get_stats('ul_demand')['mean']
                    observe_pwr = global_ue_aggr_data.get_stats('power')['mean']
                    observe_prate = global_ue_aggr_data.get_stats('prate')['mean']
                    observe_prb = global_ue_aggr_data.get_stats('ul_prb')['mean']
                    observe_prb_min = global_ue_aggr_data.get_stats('ul_prb')['min']
                    observe_prb_max = global_ue_aggr_data.get_stats('ul_prb')['max']
                observe_bler = global_ue_aggr_data.get_stats('ul_bler')['mean'] 
                observe_mcs = global_ue_aggr_data.get_stats('ul_mcs')['mean']
                observe_mcs_min = global_ue_aggr_data.get_stats('ul_mcs')['min']
                observe_mcs_max = global_ue_aggr_data.get_stats('ul_mcs')['max']

                scaled_cqi, scaled_demand, scaled_tbs, scaled_pwr, scaled_snr, scaled_mcs, scaled_prb = self.scale_features(context_cqi, context_demand, observe_tbs, observe_pwr, context_snr, observe_mcs, observe_prb)
            
            reward = self.cal_reward((scaled_tbs * self.utbs, scaled_demand * self.udemand, observe_prate, scaled_pwr, int(observe_mcs), int(observe_prb), int(mcs_action), int(prb_action)))
            
            if t == 0:
                context = [scaled_snr] # , scaled_demand
                context = self.encoder.encode_context(context).to(self.device)
                _, best_expected_reward = self.choose_action(context, self.agents)
                                   
            start_time_update = int(time.time() * 1000)
            if self.do_train:       # t > 0 and                            
               self.agents[action].update(context, action, reward)      
            end_time_update = int(time.time() * 1000)
            update_time = end_time_update - start_time_update
                
            regret = abs(best_expected_reward - reward)
            average_regret[t] = (average_regret[t-1] + regret) / (t + 1)
            average_regret[t] = regret
            cumulative_regret[t] = cumulative_regret[t-1] + regret
            rwd[t] = reward
            mcs_action, prb_action = self.actions_original[action].cpu().numpy()
            amcs[t] = int(mcs_action)
            aprb[t] = int(prb_action)
            mcs[t] = observe_mcs
            prb[t] = observe_prb
            snr[t] = context_snr
            dmnd[t] = context_demand
            pwr[t] = observe_pwr
            tbs[t] = observe_tbs
            
            if (t % 1 == 0): #  mcs_min:{int(observe_mcs_min):2d} mcs_max:{int(observe_mcs_max):2d} prb_min:{int(observe_prb_min):3d} prb_max:{int(observe_prb_max):3d}
                #print(f"{t} dmnd:{scaled_demand:3.2f} tbs:{scaled_tbs:5.2f} prate:{observe_prate:2.1f} mcs:{int(observe_mcs):2d} prb:{int(observe_prb):3d} pwr:{observe_pwr:5.2f}") # T:{update_time:3.0f}
                #print(f"{t} dmnd:{scaled_demand:3.2f} tbs:{scaled_tbs:5.2f} prate:{observe_prate:2.1f} mcs:{int(observe_mcs):2d} amcs:{int(mcs_action):2d} prb:{int(observe_prb):3d} aprb:{int(prb_action):3d} rw:{reward:+5.2f} rg:{regret:5.2f} pwr:{observe_pwr:5.2f}") # T:{update_time:3.0f}
                print(f"{t} dmnd:{scaled_demand:3.2f} tbs:{scaled_tbs:5.2f} prate:{observe_prate:2.1f} snr:{context_snr:2.0f} lr:{self.agents[0].h:5.4f} eta:{self.agents[0].eta} mcs:{int(observe_mcs):2d} prb:{int(observe_prb):3d} amcs:{int(mcs_action):2d} aprb:{int(prb_action):3d} pwr:{observe_pwr:5.2f} exrw:{best_expected_reward:+5.2f} rw:{reward:+5.2f} rg:{regret:5.2f} T:{update_time:3.0f}") # T:{update_time:3.0f}
            context = [scaled_snr] # , scaled_demand
            context = self.encoder.encode_context(context).to(self.device)
            action, best_expected_reward = self.choose_action(context, self.agents)
            self.send_action(self.actions_original[action].cpu().numpy())
            if (t + 1 > 0 and (t + 1) % 1000 == 0):
               for agent in self.agents:
                   agent.update_eta()
            if (t + 1 > 0 and (t + 1) % 1000 == 0):
               for agent in self.agents:
                   agent.update_lr()
            time.sleep(0.05)
            
        end_time = time.time()
        print("Training time: {:.2f}".format(end_time - start_time))
        print("Stopping VITS")
        # reset mcs and prb to default highest value
        self.send_action((self.mcs_high, self.prb_high))
        return (cumulative_regret, average_regret, rwd, mcs, prb, amcs, aprb, snr, dmnd, pwr, tbs)


class MACCallback(ric.mac_cb):
    def __init__(self):
        ric.mac_cb.__init__(self)
        self.ind_num = 0

    def handle(self, ind):
        # save contexts and observation into aggr_data
        if len(ind.ue_stats) > 0:
            ue_stats = ind.ue_stats[0]
            with global_lock_data:
                global_ue_aggr_data.add_data('cqi', ue_stats.wb_cqi, ind.tstamp) # ue_stats.wb_cqi
                global_ue_aggr_data.add_data('ul_throughput', ue_stats.ul_curr_tbs, ind.tstamp)
                global_ue_aggr_data.add_data('ul_demand', ue_stats.bsr, ind.tstamp)
                global_ue_aggr_data.add_data('power', ue_stats.pwr, ind.tstamp)
                global_ue_aggr_data.add_data('ul_snr', ue_stats.pusch_snr, ind.tstamp)
                global_ue_aggr_data.add_data('ul_bler', ue_stats.ul_bler, ind.tstamp)
                global_ue_aggr_data.add_data('ul_prb', ue_stats.ul_sched_rb, ind.tstamp)
                global_ue_aggr_data.add_data('ul_mcs', ue_stats.ul_mcs1, ind.tstamp)
                global_ue_aggr_data.add_data('prate', ue_stats.poor_sched_rate, ind.tstamp)

def run_monitor(stop_event, conn, run_time_sec, frequency):
    for i in range(0, len(conn)):
        print(f"Global E2 Node [{i}]: PLMN MCC = {conn[i].id.plmn.mcc}")
        print(f"Global E2 Node [{i}]: PLMN MNC = {conn[i].id.plmn.mnc}")

    if frequency == 1:
        freq = ric.Interval_ms_1
    elif frequency == 2:
        freq = ric.Interval_ms_2
    elif frequency == 5:
        freq = ric.Interval_ms_5
    else:
        freq = ric.Interval_ms_10

    for i in range(0, len(conn)):
        mac_cb = MACCallback()
        hndlr = ric.report_mac_sm(conn[i].id, freq, mac_cb)
        mac_hndlr.append(hndlr)
    
    # Wait for the stop_event or timeout
    if stop_event.wait(timeout=run_time_sec):
        print("Stop event received. Exiting monitor.")
    else:
        print("Timeout elapsed. Exiting monitor.")

if __name__ == '__main__':

    # Parse the command-line arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[SYSTEM], device: {device}')
    dimension = args.dimension
    approx = args.approx
    hessian_free = args.hessian_free
    mc_sample = 20 if hessian_free else 1

    # Initialize RIC and connections
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No connected E2 nodes found."


    T = args.step
    if args.algo1 == 'ts':
        algo_name1 = 'TS'
        algo = ThompsonSampling
        hyperparameter = lambda eta : (eta,)
    elif args.algo1 == 'vits':
        algo_name1 = 'VITS'
        algo = VITS
        hyperparameter = lambda eta : (eta, float(args.lr), args.nb_updates, approx, hessian_free, mc_sample)
    elif args.algo1 == 'lmcts':
        algo_name1 = 'LMC-TS'
        algo = Langevin
        hyperparameter = lambda eta : (eta, float(args.lr), 100)
    elif args.algo1 == 'baseline':
        algo_name1 = 'Baseline'
        algo = VITS
        hyperparameter = lambda eta : (eta, float(args.lr), args.nb_updates, approx, hessian_free, mc_sample)
    else:
        raise ValueError(args.algo1)
    
    if args.algo2 == 'baseline':
        algo_name2 = 'Baseline'
    else:
        algo_name2 = 'no-Baseline'

    algos = [algo_name1, algo_name2] # , 'Baseline'
    eta_list = [50, 100, 200, 500]
    eta = args.eta
    action_ranges = (args.mcs_low, args.mcs_high, args.mcs_step, args.prb_low, args.prb_high, args.prb_step)
    scales = (args.scale_cqi, args.scale_demand, args.scale_tbs, args.scale_power, args.scale_snr)

    # start monitoring thread
    run_time_sec = 6000
    stop_event = threading.Event()
    drl_thread = threading.Thread(target=run_monitor, args=(stop_event, conn, run_time_sec, args.report_time))
    drl_thread.daemon = True  # Ensures the thread exits when the main program exits
    drl_thread.start()

    time.sleep(1)
    
    # start resource allocation
    df = pd.DataFrame()
    np.random.seed(args.seed)
    for cur_algo_name in algos:
        if cur_algo_name == 'no-Baseline':
            continue
        do_train = True if cur_algo_name != "Baseline" else False
        for i in range(args.rounds):
            seed = np.random.randint(args.seed)
            env = vran(do_train, cur_algo_name, conn, dimension, device, args.linear, algo, hyperparameter(eta), T, seed, action_ranges, scales, args.power_penalty, args.bler_penalty, args.a, args.b)
            cum_regret, avg_regret, rwd, mcs, prb, amcs, aprb, snr, dmnd, pwr, tbs = env.compute()
            row = pd.DataFrame({'seed': seed,
                            'legend': f'{cur_algo_name}',
                            'step': range(T),
                            'snr': snr,
                            'demand': dmnd,
                            'action_mcs': amcs,
                            'action_prb': aprb,
                            'mcs': mcs,
                            'prb': prb,
                            'pwr': pwr,
                            'tbs': tbs,
                            'reward': rwd,
                            'cum_regret': cum_regret,
                            'avg_regret': avg_regret})
            df = pd.concat([df, row], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    if args.do_train:
        output_dir = '/home/nakaolab/ra/train/new_stationary_baseline/' + str(args.seed)
    else:
        output_dir = '/home/nakaolab/ra/no_train/new_stationary_baseline/'  + str(args.seed)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    result_csv_path = os.path.join(output_dir, 'vits_result.csv')
    df.to_csv(result_csv_path, index=False)
    print(f"Results saved to {result_csv_path}")

    df = df[df['avg_regret'] < 30]

    # Generate and save plots
    plot_metrics = ['cum_regret', 'avg_regret', 'pwr'] #, 'action_mcs', 'action_prb' , 'mcs', 'prb'
    for metric in plot_metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='step', y=metric, hue='legend')
        plt.xlabel('Step', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=14)
        plt.title(f'{metric.replace("_", " ").capitalize()} over Steps', fontsize=16)
        plt.legend(fontsize=12) # title='Legend', 
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    
    # stopping 
    print("Stopping xApp and cleaning up...")
    stop_event.set()
    drl_thread.join()
    for i in range(0, len(mac_hndlr)):
        ric.rm_report_mac_sm(mac_hndlr[i])
    # Avoid deadlock. ToDo revise architecture
    while ric.try_stop == 0:
        time.sleep(1)
    print("Test finished")