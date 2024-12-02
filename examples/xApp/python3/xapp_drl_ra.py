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
        
    def reward(self, user):
        theta = self.sample_posterior()
        return user.dot(theta).item()
        
    def update(self, user, action, reward):
        self.bt += reward * user
        omega = self.V_inv @ user
        self.V_inv -= omega[:, None] @ omega[None, :] / (1 + omega.dot(user))

class VITS(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, is_linear, eta, h, nb_updates, approx, hessian_free, mc_samples):
        self.eta = eta
        self.device = device
        self.dimension = dimension
        self.approx = approx
        self.hessian_free = hessian_free
        self.mc_samples = mc_samples
        self.users = torch.empty((0, dimension)).to(self.device)
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
        
    def reward(self, user):
        theta = self.sample_posterior()
        return user.dot(theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.users @ theta - self.rewards))
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
            #return torch.sum(torch.square(self.users @ theta - self.rewards))
        else:
            data_term = self.criterion(self.users @ theta, self.rewards)
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
        
    def update(self, user, action, reward):
        self.users = torch.cat([self.users, torch.tensor(user, dtype=torch.float32)[None, :].to(self.device)])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / (len(self.rewards) * self.eta)
        for _ in range(self.nb_updates):
            gradient, hessian_matrix = self.compute_gradient_hessian()
            self.mean -= h * gradient
            self.cov_semi, self.cov_semi_inv = self.update_cov(h, hessian_matrix)
            del gradient, hessian_matrix


class Langevin(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, is_linear, eta, h, nb_updates):
        self.eta = eta
        self.device = device
        self.dimension = dimension
        self.mean_prior = mean_prior
        self.cov_prior = cov_prior
        self.cov_prior_inv = torch.diag(1/(self.eta * torch.diag(cov_prior))).to(self.device)
        self.users = torch.empty((0, dimension)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.linear = is_linear
        self.h = h
        self.nb_updates = nb_updates

    def get_config(self):
        return {'eta': self.eta, 'dimension': self.dimension, 'h': self.h,
                'nb_updates': self.nb_updates, 'algorithm': 'lmcts'}

    def reward(self, user):
        if len(self.rewards) == 0:
            self.theta = self.mean_prior + torch.diag(1/torch.sqrt(torch.diag(self.cov_prior))) @ torch.normal(0, 1, size=(self.dimension,)).to(self.device)
        else:
            h = self.h / (1 + len(self.rewards))
            for _ in range(10):
                gradient = self.compute_gradient()
                self.theta += - h * gradient + torch.normal(0, np.sqrt(2 * h), size=gradient.shape).to(self.device)
        return user.dot(self.theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.users @ theta - self.rewards))
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
        else:
            data_term = torch.nn.BCEWithLogitsLoss()(self.users @ theta, self.rewards)
            regu = (theta - self.mean_prior).T @ self.cov_prior_inv @ (theta - self.mean_prior)
            return self.eta * (data_term + regu) / 2
    
    def compute_gradient(self):
        self.theta.requires_grad = True
        gradient = grad(self.potential(self.theta), self.theta)[0]
        self.theta.requires_grad = False
        return gradient
    
    def update(self, user, action, reward):
        self.users = torch.cat([self.users, torch.tensor(user, dtype=torch.float32).to(self.device)[None, :]])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / (len(self.rewards))
        for _ in range(self.nb_updates):
            gradient = self.compute_gradient()
            self.theta += - h * gradient + torch.normal(0, np.sqrt(2 * h), size=gradient.shape).to(self.device)
            del gradient


# MACCallback class is defined and derived from C++ class mac_cb
class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_mac_ind_msg_t
        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            t_diff = t_now - t_mac
            print(f"MAC Indication tstamp {t_now} diff {t_diff}")
            print('MAC rnti = ' + str(ind.ue_stats[0].rnti))


def create_conf(rnti, mcs, prb, add):
    conf = ric.mac_conf_t()
    conf.isset_pusch_mcs = add
    conf.pusch_mcs = mcs
    conf.pusch_prb = prb
    conf.rnti = rnti
    return conf

node_idx = 0
mac_hndlr = []
parser = argparse.ArgumentParser(description="MAC Control")
#parser.add_argument('-m', '--mcs', type=int, default=28, help="MCS value (default: 28)")
#parser.add_argument('-p', '--prb', type=int, default=106, help="PRB value (default: 106)")
#parser.add_argument('-a', '--add', type=int, default=True, help="Add Configurations? (default: True)")
parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument('-g', '--algo', default='vits', help="Choose algorithm (default: vits)")
parser.add_argument('-r', '--lr', type=float, default=0.01, help="Learning Rate (default: 0.01)")
parser.add_argument('-l', '--logistic', default=True, action=argparse.BooleanOptionalAction, help="Is logistic or linear? (default: True)")
parser.add_argument('-t', '--step', type=int, default=1000, help="Total training steps (default: 1000)")
parser.add_argument('-e', '--eta', type=int, default=500, help="Eta (default: 5000)")
parser.add_argument('-k', '--nb_pdates', type=int, default=10, help="Number of gradient steps K_t (default: 10)")
parser.add_argument('-x', '--approx', default=False, action=argparse.BooleanOptionalAction, help="Whether to use approximation (default: False)")
parser.add_argument('-h', '--hessian_free', default=False, action=argparse.BooleanOptionalAction, help="Is hessian free? (default: False)")

if __name__ == '__main__':

    # Parse the command-line arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[SYSTEM], device: {device}')
    dimension = 2

    # Initialize RIC and connections
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No connected E2 nodes found."
    
    for i in range(0, len(conn)):
        print(f"Global E2 Node [{i}]: PLMN MCC = {conn[i].id.plmn.mcc}")
        print(f"Global E2 Node [{i}]: PLMN MNC = {conn[i].id.plmn.mnc}")

    for i in range(0, len(conn)):
        mac_cb = MACCallback()
        hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_10, mac_cb)
        mac_hndlr.append(hndlr)
    time.sleep(1)
        
    msg = ric.mac_ctrl_msg_t()
    msg.ran_conf_len = 2
    confs = ric.mac_conf_array(2)
    for i in range(0, msg.ran_conf_len):
        confs[i] = create_conf(i, mcs, prb, add)
        print(f"Sending to rnti: {i}, mcs value: {mcs}, prb value: {prb}, add value: {add}")
        
    msg.ran_conf = confs
    ric.control_mac_sm(conn[node_idx].id, msg)
        
    for i in range(0, len(mac_hndlr)):
        ric.rm_report_mac_sm(mac_hndlr[i])
        
    while ric.try_stop == 0:
        time.sleep(1)
    print("Test finished")


