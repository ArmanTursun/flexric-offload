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
        self.mean = torch.tensor(mean_prior, dtype=torch.float32).to(self.device)
        self.cov_prior_inv = torch.diag(1/(self.eta * torch.diag(cov_prior))).to(self.device)
        self.cov_semi = torch.diag(torch.sqrt(torch.diag(cov_prior))).to(self.device)
        self.cov_semi_inv = torch.diag(1/torch.sqrt(torch.diag(cov_prior))).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
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
        self.context = torch.cat([self.context, torch.tensor(context, dtype=torch.float32)[None, :].to(self.device)])
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

mac_hndlr = []

global_ue_aggr_data = AggrData(max_length=5, datasets=['cqi', 'ul_demand', 'ul_throughput', 'power', 'ul_snr', 'ul_bler', 'ul_prb', 'ul_mcs'])

# Global lock to ensure thread-safe access to the global dictionary
global_lock_data = threading.Lock()
global_lock_offload = threading.Lock()

class vran(object):
    def __init__(self, conn, dimension, regularisation, nb_iters, device, is_linear, Agent, hyperparameters, T, seed, mcs_low, mcs_high, mcs_step, prb_low, prb_high, prb_step):
        self.dimension = dimension
        self.device = device
        self.is_linear = is_linear
        self.regularisation = regularisation
        self.nb_iters = nb_iters
        self.Agent = Agent
        self.hyperparameters = hyperparameters
        self.T = T
        self.seed = seed
        self.conn = conn
        self.mcs_low = mcs_low
        self.mcs_high = mcs_high
        self.mcs_step = mcs_step
        self.prb_low = prb_low
        self.prb_high = prb_high
        self.prb_step = prb_step

        # Generate all possible actions (MCS and PRB)
        self.actions = torch.tensor(self.generate_actions(), dtype=torch.float32).to(self.device)

        # initialize agents
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.mean_prior = torch.mean(self.actions, axis=0)
        self.cov_prior = torch.diag(torch.var(self.actions, axis=0))
        #self.cov_prior += torch.eye(self.dimension) * 1e-6  # Add small value to diagonal
        self.agents = [self.Agent(self.dimension, self.mean_prior, self.cov_prior, self.device, self.is_linear, *self.hyperparameters) for _ in range(len(self.actions))]
        print("Initilized agents")
    
    def generate_actions(self):
        """
        Generate all possible actions (MCS and PRB combinations).
        :return: Array of actions.
        """
        MCS_range = range(self.mcs_low, self.mcs_high, self.mcs_step)  # MCS values [0, 28]
        PRB_range = range(self.prb_low, self.prb_high, self.prb_step)  # PRB values [1, 273]
        return np.array([[mcs, prb] for mcs in MCS_range for prb in PRB_range])
        
    def cal_reward(self, observe):
        observe_tbs, context_demand, observe_pwr = observe
        reward = np.log(1 + observe_tbs / context_demand) - 1.5 * np.log(1 + observe_pwr) #1.5 * energy_cost
        #if self.is_linear:
        #    reward= reward + torch.normal(0, 1, size=(1,)).to(self.device)[0]
        #else:
        #    reward = torch.bernoulli(1 / (1 + torch.exp(-reward)))

        return reward
    
    def choose_action(self, context, agents):
        context = torch.tensor(context, dtype=torch.float32).to(self.device)
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
        mcs, prb = action
        add = True
        for i in range(0, msg.ran_conf_len):
            confs[i] = self.create_conf(i, int(mcs), int(prb), add)
            #print(f"Sending to rnti: {i}, mcs value: {int(mcs)}, prb value: {int(prb)}, add value: {add}")
        
        msg.ran_conf = confs
        ric.control_mac_sm(self.conn[0].id, msg)
    
    def min_max_scale(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def scale_features(self, cqi, demand, tbs, power, snr):
        scaled_cqi = self.min_max_scale(cqi, 0, 15)  # Scale CQI to [0, 1]
        scaled_demand = self.min_max_scale(demand, 0, 300000)  # Scale Demand to [0, 1]
        scaled_tbs = self.min_max_scale(tbs, 0, 6000)  # Scale TBS to [0, 1]
        scaled_power = self.min_max_scale(power, 0, 50)  # Scale Power to [0, 1]
        scaled_snr = self.min_max_scale(snr, 0, 32)
        return scaled_cqi, scaled_demand, scaled_tbs, scaled_power, scaled_snr

    
    def compute(self):
        cumulative_regret = torch.zeros((self.T,))
        average_regret = torch.zeros((self.T,))
        mcs = torch.zeros((self.T,))
        prb = torch.zeros((self.T,))
        context_cqi, context_demand = 0, 0
        observe_tbs, observe_pwr = 0, 0
        best_expected_reward = 0
        action = torch.tensor(0, dtype=torch.float32).to(self.device)
        start_time = time.time()
        for t in range(self.T):
            with global_lock_data:
                # context
                context_cqi = global_ue_aggr_data.get_stats('cqi')['mean']
                context_demand = global_ue_aggr_data.get_stats('ul_demand')['mean']
                context_snr = global_ue_aggr_data.get_stats('ul_snr')['mean']
                # observation
                observe_tbs = global_ue_aggr_data.get_stats('ul_throughput')['mean']
                observe_pwr = global_ue_aggr_data.get_stats('power')['mean']
                observe_bler = global_ue_aggr_data.get_stats('ul_bler')['mean']
                observe_prb = global_ue_aggr_data.get_stats('ul_prb')['mean']
                observe_mcs = global_ue_aggr_data.get_stats('ul_mcs')['mean']
                context_cqi, context_demand, observe_tbs, observe_pwr, context_snr = self.scale_features(context_cqi, context_demand, observe_tbs, observe_pwr, context_snr)
            if t > 0:
                reward = self.cal_reward((observe_tbs, context_demand, observe_pwr))
                print(t, context_snr, observe_bler, context_demand, reward, best_expected_reward)
                self.agents[action].update(context, action, reward)
                average_regret[t] = best_expected_reward - reward
                cumulative_regret[t] = cumulative_regret[t-1] + average_regret[t]
                mcs[t] = observe_mcs
                prb[t] = observe_prb
                
            context = [context_snr, context_demand]
            action, best_expected_reward = self.choose_action(context, self.agents)
            self.send_action(self.actions[action].cpu().numpy())
            #time.sleep(0.04)
            
        end_time = time.time()
        print("Training time: {:.2f}".format(end_time - start_time))
        print("Stopping VITS")
        return cumulative_regret, average_regret, mcs, prb


class MACCallback(ric.mac_cb):
    def __init__(self):
        ric.mac_cb.__init__(self)
        self.ind_num = 0

    def handle(self, ind):
        # save cqi and demand values of each tbs of each ue into aggr_data
        if len(ind.ue_stats) > 0:
            ue_stats = ind.ue_stats[0]
            with global_lock_data:
                #if (self.ind_num > 100):
                global_ue_aggr_data.add_data('cqi', ue_stats.wb_cqi, ind.tstamp) # ue_stats.wb_cqi
                global_ue_aggr_data.add_data('ul_throughput', ue_stats.ul_curr_tbs, ind.tstamp)
                global_ue_aggr_data.add_data('ul_demand', ue_stats.bsr, ind.tstamp)
                global_ue_aggr_data.add_data('power', ue_stats.pwr, ind.tstamp)
                global_ue_aggr_data.add_data('ul_snr', ue_stats.pusch_snr, ind.tstamp)
                global_ue_aggr_data.add_data('ul_bler', ue_stats.ul_bler, ind.tstamp)
                global_ue_aggr_data.add_data('ul_prb', ue_stats.ul_sched_rb, ind.tstamp)
                global_ue_aggr_data.add_data('ul_mcs', ue_stats.ul_mcs1, ind.tstamp)
            #self.ind_num += 1

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
        #time.sleep(1)
    
    #time.sleep(run_time_sec)
    # Wait for the stop_event or timeout
    if stop_event.wait(timeout=run_time_sec):
        print("Stop event received. Exiting monitor.")
    else:
        print("Timeout elapsed. Exiting monitor.")

parser = argparse.ArgumentParser(description="MAC Control")
#parser.add_argument('-m', '--mcs', type=int, default=28, help="MCS value (default: 28)")
#parser.add_argument('-p', '--prb', type=int, default=106, help="PRB value (default: 106)")
#parser.add_argument('-a', '--add', type=int, default=True, help="Add Configurations? (default: True)")
parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument('-g', '--algo', default='vits', help="Choose algorithm (default: vits)")
parser.add_argument('-r', '--lr', type=float, default=0.01, help="Learning Rate (default: 0.01)")
parser.add_argument('-l', '--logistic', default=True, action=argparse.BooleanOptionalAction, help="Is logistic or linear? (default: True)")
parser.add_argument('-t', '--step', type=int, default=1000, help="Total training steps (default: 1000)")
parser.add_argument('-e', '--eta', type=int, default=500, help="Eta (default: 500)")
parser.add_argument('-k', '--nb_updates', type=int, default=10, help="Number of gradient steps K_t (default: 10)")
parser.add_argument('-x', '--approx', default=False, action=argparse.BooleanOptionalAction, help="Whether to use approximation (default: False)")
parser.add_argument('-f', '--hessian_free', default=False, action=argparse.BooleanOptionalAction, help="Is hessian free? (default: False)")
parser.add_argument('--mcs_low', type=int, default=9, help="lower bound of mcs (default: 9)")
parser.add_argument('--mcs_high', type=int, default=29, help="upper bound of mcs (default: 29)")
parser.add_argument('--mcs_step', type=int, default=2, help="step of mcs values (default: 2)")
parser.add_argument('--prb_low', type=int, default=5, help="lower bound of prb (default: 5)")
parser.add_argument('--prb_high', type=int, default=107, help="upper bound of prb (default: 107)")
parser.add_argument('--prb_step', type=int, default=5, help="step of prb values (default: 5)")
parser.add_argument('--report_time', type=int, default=1, help="E2 reports time frequency (default: 1)")
parser.add_argument('--rounds', type=int, default=10, help="Total rounds that test runs (default: 10)")

if __name__ == '__main__':

    # Parse the command-line arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[SYSTEM], device: {device}')
    dimension = 2
    approx = args.approx
    hessian_free = args.hessian_free
    mc_sample = 20 if hessian_free else 1

    # Initialize RIC and connections
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No connected E2 nodes found."

    T = args.step
    if args.algo == 'ts':
        algo_name = 'TS'
        algo = ThompsonSampling
        hyperparameter = lambda eta : (eta,)
    elif args.algo == 'vits':
        algo_name = 'VITS'
        algo = VITS
        hyperparameter = lambda eta : (eta, float(args.lr), args.nb_updates, approx, hessian_free, mc_sample)
    elif args.algo == 'lmcts':
        algo_name = 'LMC-TS'
        algo = Langevin
        hyperparameter = lambda eta : (eta, float(args.lr), 100)
    else:
        raise ValueError(args.algo)

    eta_list = [50, 100, 200, 500]
    eta = args.eta
    env = vran(conn, dimension, 1, 50, device, args.logistic, algo, hyperparameter(eta), T, args.seed, args.mcs_low, args.mcs_high, args.mcs_step, args.prb_low, args.prb_high, args.prb_step)

    # start monitoring thread
    run_time_sec = 1000
    stop_event = threading.Event()
    drl_thread = threading.Thread(target=run_monitor, args=(stop_event, conn, run_time_sec, args.report_time))
    drl_thread.daemon = True  # Ensures the thread exits when the main program exits
    drl_thread.start()

    time.sleep(1)
    
    # start resource allocation
    df = pd.DataFrame()
    for _ in range(args.rounds):
        cum_regret, avg_regret, mcs, prb = env.compute()
        row = pd.DataFrame({'seed': args.seed,
                            'legend': f'{algo_name} - eta: {eta}',
                            'step': range(T),
                            'cum_regret': cum_regret,
                            'avg_regret': avg_regret,
                            'mcs': mcs,
                            'prb': prb})
        df = pd.concat([df, row], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    output_dir = '/home/nakaolab/ra'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    result_csv_path = os.path.join(output_dir, 'vits_result.csv')
    df.to_csv(result_csv_path, index=False)
    print(f"Results saved to {result_csv_path}")

    # Generate and save plots
    plot_metrics = ['cum_regret', 'avg_regret', 'mcs', 'prb']
    for metric in plot_metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='step', y=metric, hue='legend')
        plt.xlabel('Step', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=14)
        plt.title(f'{metric.replace("_", " ").capitalize()} over Steps', fontsize=16)
        plt.legend(title='Legend', fontsize=12)
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    
    #plt.show()
    

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
        


