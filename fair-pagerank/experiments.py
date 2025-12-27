import os, sys
from collections.abc import Iterable
from itertools import product
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pandas import DataFrame, read_csv
from scipy.sparse import coo_array, csr_array
from tqdm import tqdm
from timeit import default_timer as timer

from_array_to_dict = lambda an_array: {i: an_array[i] for i in range(an_array.shape[0])}
from_dict_to_array = lambda a_dict: np.array([a_dict[i] for i in range(len(a_dict))])

digits_after_point = lambda number: str(number).split(".")[1]
phi_to_str = lambda number: "d" if number > 1 else digits_after_point(number)

class Network:
    def __init__(self, dataset):
        df = read_csv(sys.path[0] + "/Datasets/" + dataset + "/nodes.csv")
        labels = df["Label"].to_numpy()
        self.red_group = df[df["Label"] == 0]["Node"].to_numpy()
        self.blue_group = df[df["Label"] == 1]["Node"].to_numpy()
        red_group = set(self.red_group)
        blue_group = set(self.blue_group)
        self.n_nodes = self.red_group.shape[0] + self.blue_group.shape[0]
        self.nodes = np.arange(self.n_nodes)

        df = read_csv(sys.path[0] + "/Datasets/" + dataset + "/edges.csv")
        edge_source_array = df["Source"].to_numpy()
        edge_target_array = df["Target"].to_numpy()
        self.n_edges = edge_target_array.shape[0]
        self.adjacency_matrix = coo_array((np.ones(self.n_edges), (edge_source_array, edge_target_array)), shape = (self.n_nodes, self.n_nodes)).todense()
        
        self.red_out_neighborhood = {}
        self.blue_out_neighborhood = {}
        self.red_out_degree = np.empty(self.n_nodes)
        self.blue_out_degree = np.empty(self.n_nodes)
        for node in self.nodes:
            out_neighborhood = set(np.nonzero(self.adjacency_matrix[[node],:])[1])
            self.red_out_neighborhood[node] = np.array(list(out_neighborhood.intersection(red_group)))
            self.blue_out_neighborhood[node] = np.array(list(out_neighborhood.intersection(blue_group)))
            self.red_out_degree[node] = self.red_out_neighborhood[node].shape[0]
            self.blue_out_degree[node] = self.blue_out_neighborhood[node].shape[0]
        
        DataFrame({"Node": self.nodes, "Group": labels, "Red Group Out Degree": self.red_out_degree, "Blue Group Out Degree": self.blue_out_degree}).to_csv(sys.path[0] + "/Datasets/" + dataset + "/degrees.csv", index = False)

class Configuration:
    def __init__(self,
                 gamma: float = .5,
                 phi: float = .5,
                 selector: str | None = "safg",
                 intervention: str | None = "pgoi",
                 n_runs: int | None = None):
        self.phi = phi
        self.gamma = gamma
        self.selector = selector
        self.intervention = intervention
        self.n_runs = n_runs
        if self.n_runs is None: # dynamically set a default n_runs
            if self.selector == "rp":
                self.n_runs = 10
            else:
                self.n_runs = 1

class Settings:
    def __init__(self,
                 save: bool | None = False,
                 verbose: bool | None = False):
        self.save = save
        self.verbose = verbose

class Experiment:
    selectors = ["ref", "defg", "dafg", "sefg", "safg", "rp"]
    selector_descriptions = {
        "ref": "Reference (No Selection)",
        "defg": "Fairness Gain",
        "dafg": "Pagerank Mass",
        "sefg": "Original Fairness Gain",
        "safg": "Original Pagerank Mass",
        "dul": "Minus Utility Loss",
        "duli": "Minus Utility Loss Incremental",
        "duls": "Minus Utility Loss Square Root",
        "dulis": "Minus Utility Loss Incremental Square Root",
        "dgl": "Fairness Gain / Utility Loss",
        "dgli": "Fairness Gain / Utility Loss Incremental",
        "dgls": "Fairness Gain / Utility Loss Square Root",
        "dglis": "Fairness Gain / Utility Loss Incremental Square Root",
        "suli": "Original Minus Utility Loss Incremental",
        "sgli": "Original Fairness Gain / Utility Loss Incremental",
        "dpr": "Pagerank",
        "spr": "Original Pagerank",
        "f4": "g + (1-l) where g,l are rescaled to [0,1]",
        "f5": "g * (1-l) where g,l are rescaled to [0,1]",
        "f6": "- [(1-g)^2 + l^2] where g,l are rescaled to [0,1]",
        "f7": "Formula 7",
        "f8": "Log Fairness Gain / Log Utility Loss",
        "rp": "Uniformly Random (No Maximization)"
    }
    interventions = ["gsir", "mix", "pgoi", "nrb", "rrd"]
    intervention_descriptions = {
        "gsir": "Group Social Influence Rebalancing if able",
        "mix": "Group Social Influence Rebalancing if able, Protected Group Opinion Injection otherwise",
        "pgoi": "Protected Group Opinion Injection always",
        "nrb": "Neighborhood Rebalancing",
        "rrd": "Residual Redirection"
    }
    # _results_template = {"Iteration": [], "Intervention Target": [], "Expected Fairness Gain": [], "Red Group Influence": [], "Fairness Gain": [], "Utility Loss": [], "Performance Time": []}
    _results_template = {"Iteration": [], "Intervention Target": [], "Red Group Pagerank": [], "Fairness Gain": [], "Utility Loss": [], "Performance Time": []}

    def __init__(self,
                 dataset: str,
                 configurations: Configuration | Iterable[Configuration],
                 settings: Settings,
                 description: str | None = None):
        self.dataset = dataset
        self.nw = Network(self.dataset)
        self.configs = configurations
        if isinstance(self.configs, Configuration): # ensure self.configs is an Iterable
            self.configs = [self.configs]
        self.settings = settings
        self.description = description
        if self.description is None: # dynamically set a default description
            self.description = self.dataset + " dataset, " + str(len(self.configs)) + " configs"
        
        self.stats = {"Phi": [], "Gamma": [],
                      "Original Red Group Pagerank": [], "Achieved Red Group Pagerank": [],
                      "# Available Intervention Candidates": [], "# Selected Intervention Candidates": [],
                      "Utility Loss": []}
        self.local_values = {"Node": [], "Label": [], "Social Influence": [], "Fairness Gain": []}

    # def _compute_fairness_gain(self, node):
    #     Lambda_numerator = np.sum(self._Sigma[node, self.nw.red_out_neighborhood[node]] @ self._Q[self.nw.red_out_neighborhood[node], :][:, self.nw.red_group]) / np.sum(self._Sigma[node, self.nw.red_group])\
    #         - np.sum(self._Sigma[node, self.nw.blue_out_neighborhood[node]] @ self._Q[self.nw.blue_out_neighborhood[node], :][:, self.nw.red_group]) / np.sum(self._Sigma[node, self.nw.blue_group])
    #     Lambda_denominator = self._gamma[node] / ((1-self._gamma[node]) * (self._phi - np.sum(self._Sigma[node, self.nw.red_group]) / np.sum(self._Sigma[node, :])))\
    #         - np.sum(self._Sigma[node, self.nw.red_out_neighborhood[node]] @ self._Q[self.nw.red_out_neighborhood[node], node]) / np.sum(self._Sigma[node, self.nw.red_group])\
    #             + np.sum(self._Sigma[node, self.nw.blue_out_neighborhood[node]] @ self._Q[self.nw.blue_out_neighborhood[node], node]) / np.sum(self._Sigma[node, self.nw.blue_group])
    #     return (np.sum(self._Q[:, node]) / self.nw.n_nodes) * (Lambda_numerator / Lambda_denominator)
    
    # def _compute_intervention_vector_gsir(self, node):
    #     alpha = self._phi / np.sum(self._Sigma[node, self.nw.red_group])
    #     beta = (1- self._phi) / np.sum(self._Sigma[node, self.nw.blue_group])
    #     return self._Sigma[node, :].multiply(((alpha - 1) * np.array([1 if other_node in self.nw.red_out_neighborhood[node] else 0 for other_node in self.nw.nodes])
    #             + (beta - 1) * np.array([1 if other_node in self.nw.blue_out_neighborhood[node] else 0 for other_node in self.nw.nodes])).reshape((1, self.nw.n_nodes)))

    # def _compute_intervention_vector_mix(self, node):
    #     if self._intervention_candidancy_condition_gsir(node):
    #         return self._compute_intervention_vector_gsir(node)
    #     else:
    #         return self._compute_intervention_vector_pgoi(node)

    # def _compute_intervention_vector_pgoi(self, node):
    #     delta = (self._phi - np.sum(self._Sigma[node, self.nw.red_group])) / (1 - np.sum(self._Sigma[node, self.nw.red_group]))
    #     return - delta * self._Sigma[node, :].multiply(np.array([1 if other_node in self.nw.red_out_neighborhood[node] or other_node in self.nw.blue_out_neighborhood[node] else 0 for other_node in self.nw.nodes]).reshape((1, self.nw.n_nodes)))\
    #         + (delta / len(self.nw.red_group)) * np.array([1 if other_node in self.nw.red_group else 0 for other_node in self.nw.nodes]).reshape((1, self.nw.n_nodes))

    # _compute_intervention_vector = {
    #     "gsir": _compute_intervention_vector_gsir,
    #     "mix": _compute_intervention_vector_mix,
    #     "pgoi": _compute_intervention_vector_pgoi
    # }

    # def _compute_exact_fairness_gain(self, intervention, node):
    #     d = self._compute_intervention_vector[intervention](self, node)
    #     return (np.sum(self._Q[:, node]) / self.nw.n_nodes) * (d @ np.sum(self._Q[:, self.nw.red_group], axis = 1).reshape((self.nw.n_nodes, 1))) / (self._gamma[node] - d @ self._Q[:, node])
    
    # def _compute_approximate_fairness_gain(self, intervention, node):
    #     return np.sum(self._Q[:, node]) / self.nw.n_nodes

    def _compute_red_group_influence(self):
        return np.sum(self._Q[:,self.nw.red_group]) / self.nw.n_nodes

    def _intervention_candidancy_condition_gsir(self, node):
        return self.nw.red_out_degree[node] > 0 and np.sum(self._Sigma[node, self.nw.red_group]) / (1 - self._gamma[node]) < self._phi

    def _intervention_candidancy_condition_pgoi(self, node):
        return np.sum(self._Sigma[node, self.nw.red_group]) / (1 - self._gamma[node]) < self._phi

    def _intervention_candidancy_condition_nrb(self, node):
        return self.nw.red_out_degree[node] > 0 and np.sum(self._P[node, self.nw.red_group]) / np.sum(self._P[node, :]) < self._phi

    def _intervention_candidancy_condition_rrd(self, node):
        return np.sum(self._P_original[node, self.nw.red_group]) / np.sum(self._P_original[node, :]) < self._phi

    _intervention_candidancy_conditions = {
        "gsir": _intervention_candidancy_condition_gsir,
        "mix": _intervention_candidancy_condition_pgoi,
        "pgoi": _intervention_candidancy_condition_pgoi,
        "nrb": _intervention_candidancy_condition_rrd,
        "rrd": _intervention_candidancy_condition_rrd
    }

    def _determine_intervention_candidate_nodes(self, intervention):
        self._intervention_candidate_nodes = [node for node in self.nw.nodes if self._intervention_candidancy_conditions[intervention](self, node)]
    
    def _compute_pagerank(self):
        # I = np.eye(self.nw.n_nodes) # sps.eye_array(self.nw.n_nodes, format = "csr")
        # self._p = 0.15 * self._v @ np.linalg.inv(I - (1 - 0.15) * self._P) # original pagerank

        # self._p = from_dict_to_array(nx.pagerank(nx.from_numpy_array(self._P, create_using = nx.DiGraph), personalization = from_array_to_dict(self._v)))

        p_previous = np.zeros(self.nw.n_nodes)
        while np.linalg.norm(self._p - p_previous) > 10**-9:
            p_previous = self._p
            self._p = (1 - self._gamma) * p_previous @ self._P + self._gamma * self._v
    
    def _compute_utility_losses(self):
        # print("Computing utility losses...", end = " ")
        # start_time = timer()
        c = self._p[self._intervention_candidate_nodes] / (self._gamma/(1-self._gamma) - np.sum(self._D[self._intervention_candidate_nodes, :] * self._Q[:, self._intervention_candidate_nodes].transpose(), axis = 1))
        dQ = self._D[self._intervention_candidate_nodes, :] @ self._Q
        self._utility_loss = np.empty(self.nw.n_nodes)
        self._utility_loss[self._intervention_candidate_nodes] = (c * c) * np.sum(dQ * dQ, axis = 1) + 2 * c * (dQ @ (self._p - self._p_original)) + np.linalg.norm(self._p - self._p_original)**2
        # end_time = timer()
        # print("Done after", end_time - start_time, "seconds!")

    def _compute_exact_fairness_gains(self):
        # print("Computing fairness gains...", end = " ")
        # start_time = timer()
        self._fairness_gain = np.empty(self.nw.n_nodes)
        self._fairness_gain[self._intervention_candidate_nodes] = self._p[self._intervention_candidate_nodes] * (self._D[self._intervention_candidate_nodes, :] @ self._Q__R)\
            / (self._gamma/(1-self._gamma) - np.sum(self._D[self._intervention_candidate_nodes, :] * self._Q[:, self._intervention_candidate_nodes].transpose(), axis = 1))
        #         / (self._gamma[self._intervention_candidate_nodes] - np.sum(self._D[self._intervention_candidate_nodes, :] * self._Q[:, self._intervention_candidate_nodes].transpose(), axis = 1))
        # end_time = timer()
        # print("Done after", end_time - start_time, "seconds!")
        
        # self._sum_social_influence_from_red_group_to_self = np.zeros(self.nw.n_nodes)
        # self._sum_social_influence_from_blue_group_to_self = np.zeros(self.nw.n_nodes)
        # self._sum_network_influence_from_self_to_other = np.zeros(self.nw.n_nodes)
        # self._sum_network_influence_from_self_to_red_out_neighbors = np.zeros(self.nw.n_nodes)
        # self._sum_network_influence_from_self_to_blue_out_neighbors = np.zeros(self.nw.n_nodes)
        # self._sum_network_influence_from_red_group_to_red_out_neighbors = np.zeros(self.nw.n_nodes)
        # self._sum_network_influence_from_red_group_to_blue_out_neighbors = np.zeros(self.nw.n_nodes)

        # sum_network_influence_from_red_group_to_self = np.sum(Sigma[:, self.nw.red_group] * Q[:, self.nw.red_group], axis = 1)
        # for node in nodes:
        #     for red_out_neighbor in self.nw.red_out_neighborhood[node]:
        #         self._sum_social_influence_from_red_group_to_self[node] += Sigma[node, red_out_neighbor]
        #         self._sum_network_influence_from_self_to_other[node] += np.sum(Q[:, node])
        #         self._sum_network_influence_from_self_to_red_out_neighbors[node] += Q[red_out_neighbor, node]
        #     self._sum_network_influence_from_red_group_to_red_out_neighbors[node] = np.sum(sum_network_influence_from_red_group_to_self[self.nw.red_out_neighborhood[node]])
        #     for blue_out_neighbor in self.nw.blue_out_neighborhood[node]:
        #         self._sum_social_influence_from_blue_group_to_self[node] += Sigma[node, blue_out_neighbor]
        #         self._sum_network_influence_from_self_to_blue_out_neighbors[node] += Q[blue_out_neighbor, node]
        #     self._sum_network_influence_from_red_group_to_blue_out_neighbors[node] = np.sum(sum_network_influence_from_red_group_to_self[self.nw.blue_out_neighborhood[node]])

        # Lambda_numerator = self._sum_network_influence_from_red_group_to_red_out_neighbors / self._sum_social_influence_from_red_group_to_self\
        #     - self._sum_network_influence_from_red_group_to_blue_out_neighbors / self._sum_social_influence_from_blue_group_to_self
        # Lambda_denominator = gamma / ((1-gamma) * (phi - self._sum_social_influence_from_red_group_to_self / (1-gamma)))\
        #     - self._sum_network_influence_from_self_to_red_out_neighbors / self._sum_social_influence_from_red_group_to_self\
        #         + self._sum_network_influence_from_self_to_blue_out_neighbors / self._sum_social_influence_from_blue_group_to_self
        # self._fairness_gain = (self._sum_network_influence_from_self_to_other / self.nw.n_nodes) * (Lambda_numerator / Lambda_denominator)

    def _compute_approximate_fairness_gains(self):
        self._fairness_gain = np.empty(self.nw.n_nodes)
        self._fairness_gain[self._intervention_candidate_nodes] = np.sum(self._Q[:, self._intervention_candidate_nodes], axis = 0) / self.nw.n_nodes

    def _intervention_target_node_generator_defg(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]))

    def _intervention_target_node_generator_dafg(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_approximate_fairness_gains()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]))

    def _intervention_target_node_generator_sefg(self):
        self._compute_exact_fairness_gains()
        self._intervention_candidate_nodes.sort(key = lambda node: (self._fairness_gain[node], self.nw.n_nodes - node)) # ascending order
        while len(self._intervention_candidate_nodes) > 0:
            yield self._intervention_candidate_nodes.pop() # popping last node for maximum fairness gain

    def _intervention_target_node_generator_safg(self):
        self._compute_approximate_fairness_gains()
        self._intervention_candidate_nodes.sort(key = lambda node: (self._fairness_gain[node], self.nw.n_nodes - node)) # ascending order
        while len(self._intervention_candidate_nodes) > 0:
            yield self._intervention_candidate_nodes.pop() # popping last node for maximum fairness gain

    def _intervention_target_node_generator_spr(self):
        self._compute_pagerank()
        self._intervention_candidate_nodes.sort(key = lambda node: (self._p[node], node)) # ascending order
        while len(self._intervention_candidate_nodes) > 0:
            yield self._intervention_candidate_nodes.pop() # pop last node for maximum pagerank

    def _intervention_target_node_generator_dpr(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_pagerank()
            self._intervention_candidate_nodes.sort(key = lambda node: (self._p[node], node)) # ascending order
            yield self._intervention_candidate_nodes.pop() # pop last node for maximum pagerank

    def _intervention_target_node_generator_rp(self):
        # self._fairness_gain = np.empty(self.nw.n_nodes)
        # self._fairness_gain[self._intervention_candidate_nodes] = np.full(len(self._intervention_candidate_nodes), np.nan)
        rng = np.random.default_rng(seed = None) # randomly constructing a random number generator
        while len(self._intervention_candidate_nodes) > 0:
            intervention_target_node = rng.choice(self._intervention_candidate_nodes) # uniform distribution
            self._intervention_candidate_nodes.remove(intervention_target_node)
            yield intervention_target_node

    def _intervention_target_node_generator_suli(self):
        self._compute_utility_losses()
        utility_loss_previous = np.linalg.norm(self._p - self._p_original)**2
        self._intervention_candidate_nodes.sort(key = lambda node: (self._utility_loss[node] - utility_loss_previous, self.nw.n_nodes - node), reverse = True) # descending order
        while len(self._intervention_candidate_nodes) > 0:
            yield self._intervention_candidate_nodes.pop() # popping last node for minimum utility loss

    def _intervention_target_node_generator_sgli(self):
        self._compute_exact_fairness_gains()
        self._compute_utility_losses()
        utility_loss_previous = np.linalg.norm(self._p - self._p_original)**2
        self._intervention_candidate_nodes.sort(key = lambda node: (self._fairness_gain[node] / (self._utility_loss[node] - utility_loss_previous), self.nw.n_nodes - node)) # ascending order
        while len(self._intervention_candidate_nodes) > 0:
            yield self._intervention_candidate_nodes.pop() # popping last node for maximum fairness gain / utility loss
    
    def _intervention_target_node_generator_dul(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmin(self._utility_loss[self._intervention_candidate_nodes]))
    
    def _intervention_target_node_generator_duli(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmin(self._utility_loss[self._intervention_candidate_nodes] - np.linalg.norm(self._p - self._p_original)**2))
    
    def _intervention_target_node_generator_duls(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmin(np.sqrt(self._utility_loss[self._intervention_candidate_nodes])))
    
    def _intervention_target_node_generator_dulis(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmin(np.sqrt(self._utility_loss[self._intervention_candidate_nodes]) - np.linalg.norm(self._p - self._p_original)))
    
    def _intervention_target_node_generator_dgl(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]\
                                                                   / (self._utility_loss[self._intervention_candidate_nodes])))
    
    def _intervention_target_node_generator_dgli(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]\
                                                                   / (self._utility_loss[self._intervention_candidate_nodes] - np.linalg.norm(self._p - self._p_original)**2)))
    
    def _intervention_target_node_generator_dgls(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]\
                                                                   / np.sqrt(self._utility_loss[self._intervention_candidate_nodes])))
    
    def _intervention_target_node_generator_dglis(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax(self._fairness_gain[self._intervention_candidate_nodes]\
                                                                   / (np.sqrt(self._utility_loss[self._intervention_candidate_nodes]) - np.linalg.norm(self._p - self._p_original))))

    def _intervention_target_node_generator_f4(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax((self._fairness_gain[self._intervention_candidate_nodes] - np.min(self._fairness_gain[self._intervention_candidate_nodes])) / (np.max(self._fairness_gain[self._intervention_candidate_nodes]) - np.min(self._fairness_gain[self._intervention_candidate_nodes])) + (np.max(self._utility_loss[self._intervention_candidate_nodes]) - self._utility_loss[self._intervention_candidate_nodes]) / (np.max(self._utility_loss[self._intervention_candidate_nodes]) - np.min(self._utility_loss[self._intervention_candidate_nodes]))))

    def _intervention_target_node_generator_f5(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax((self._fairness_gain[self._intervention_candidate_nodes] - np.min(self._fairness_gain[self._intervention_candidate_nodes])) / (np.max(self._fairness_gain[self._intervention_candidate_nodes]) - np.min(self._fairness_gain[self._intervention_candidate_nodes])) * (np.max(self._utility_loss[self._intervention_candidate_nodes]) - self._utility_loss[self._intervention_candidate_nodes]) / (np.max(self._utility_loss[self._intervention_candidate_nodes]) - np.min(self._utility_loss[self._intervention_candidate_nodes]))))

    def _intervention_target_node_generator_f6(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmin(((np.max(self._fairness_gain[self._intervention_candidate_nodes]) - self._fairness_gain[self._intervention_candidate_nodes]) / (np.max(self._fairness_gain[self._intervention_candidate_nodes]) - np.min(self._fairness_gain[self._intervention_candidate_nodes])))**2 + ((self._utility_loss[self._intervention_candidate_nodes] - np.min(self._utility_loss[self._intervention_candidate_nodes])) / (np.max(self._utility_loss[self._intervention_candidate_nodes]) - np.min(self._utility_loss[self._intervention_candidate_nodes])))**2))
    
    def _intervention_target_node_generator_f7(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            yield self._intervention_candidate_nodes.pop(np.argmin(((np.max(self._fairness_gain[self._intervention_candidate_nodes]) - self._fairness_gain[self._intervention_candidate_nodes]) / (np.max(self._fairness_gain[self._intervention_candidate_nodes]) - np.min(self._fairness_gain[self._intervention_candidate_nodes])))**2\
                                                                   + ((np.max(self.nw.red_out_degree[self._intervention_candidate_nodes]) - self.nw.red_out_degree[self._intervention_candidate_nodes]) / (np.max(self.nw.red_out_degree[self._intervention_candidate_nodes]) - np.min(self.nw.red_out_degree[self._intervention_candidate_nodes])))**2\
                                                                    + ((np.max(self.nw.blue_out_degree[self._intervention_candidate_nodes]) - self.nw.blue_out_degree[self._intervention_candidate_nodes]) / (np.max(self.nw.blue_out_degree[self._intervention_candidate_nodes]) - np.min(self.nw.blue_out_degree[self._intervention_candidate_nodes])))**2))

    # def _intervention_target_node_generator_f7(self):
    #     counterray = np.zeros(self.nw.n_nodes)
    #     while len(self._intervention_candidate_nodes) > 0:
    #         self._compute_exact_fairness_gains()
    #         counterray_cover = np.sum(counterray.reshape((1, self.nw.n_nodes)) * self.nw.adjacency_matrix, axis = 1)
    #         # intervention_target_node = self._intervention_candidate_nodes.pop(np.argmin(((np.max(self._fairness_gain[self._intervention_candidate_nodes]) - self._fairness_gain[self._intervention_candidate_nodes]) / (np.max(self._fairness_gain[self._intervention_candidate_nodes]) - np.min(self._fairness_gain[self._intervention_candidate_nodes])))\
    #         #                                                                             * ((np.max(self.nw.red_out_degree[self._intervention_candidate_nodes]) - self.nw.red_out_degree[self._intervention_candidate_nodes]) / (np.max(self.nw.red_out_degree[self._intervention_candidate_nodes]) - np.min(self.nw.red_out_degree[self._intervention_candidate_nodes])))\
    #         #                                                                                 * ((np.max(self.nw.blue_out_degree[self._intervention_candidate_nodes]) - self.nw.blue_out_degree[self._intervention_candidate_nodes]) / (np.max(self.nw.blue_out_degree[self._intervention_candidate_nodes]) - np.min(self.nw.blue_out_degree[self._intervention_candidate_nodes])))\
    #         #                                                                                     * ((counterray_cover - np.min(counterray_cover)) / (np.max(counterray_cover) - np.min(counterray_cover)))))
    #         intervention_target_node = sorted(self._intervention_candidate_nodes, key = lambda node: (- counterray_cover[node], self.nw.red_out_degree[node] + self.nw.blue_out_degree[node], self._fairness_gain[node])).pop()
    #         self._intervention_candidate_nodes.remove(intervention_target_node)
    #         if np.sum(self.nw.adjacency_matrix[intervention_target_node, :]) == 0:
    #             counterray += np.ones(self.nw.n_nodes)
    #         else:
    #             counterray += self.nw.adjacency_matrix[intervention_target_node, :]
    #         yield intervention_target_node

    def _intervention_target_node_generator_f8(self):
        while len(self._intervention_candidate_nodes) > 0:
            self._compute_exact_fairness_gains()
            self._compute_utility_losses()
            yield self._intervention_candidate_nodes.pop(np.argmax(np.log10(self._fairness_gain[self._intervention_candidate_nodes])**2 / np.log10(self._utility_loss[self._intervention_candidate_nodes])))
    
    _intervention_target_node_generators = {
        "defg": _intervention_target_node_generator_defg,
        "dafg": _intervention_target_node_generator_dafg,
        "sefg": _intervention_target_node_generator_sefg,
        "safg": _intervention_target_node_generator_safg,
        "dpr": _intervention_target_node_generator_dpr,
        "spr": _intervention_target_node_generator_spr,
        "rp": _intervention_target_node_generator_rp,
        "dul": _intervention_target_node_generator_dul,
        "duli": _intervention_target_node_generator_duli,
        "duls": _intervention_target_node_generator_duls,
        "dulis": _intervention_target_node_generator_dulis,
        "dgl": _intervention_target_node_generator_dgl,
        "dgli": _intervention_target_node_generator_dgli,
        "dgls": _intervention_target_node_generator_dgls,
        "dglis": _intervention_target_node_generator_dglis,
        "suli": _intervention_target_node_generator_suli,
        "sgli": _intervention_target_node_generator_sgli,
        "f4": _intervention_target_node_generator_f4,
        "f5": _intervention_target_node_generator_f5,
        "f6": _intervention_target_node_generator_f6,
        "f7": _intervention_target_node_generator_f7,
        "f8": _intervention_target_node_generator_f8
    }
    
    def _compute_intervention_vectors_gsir(self):
        alpha = self._phi * (1 - self._gamma[0]) / np.sum(self._Sigma[self._intervention_candidate_nodes, :][:, self.nw.red_group], axis = 1)
        beta = (1- self._phi) * (1 - self._gamma[0]) / np.sum(self._Sigma[self._intervention_candidate_nodes, :][:, self.nw.blue_group], axis = 1)
        n_intervention_candidate_nodes = len(self._intervention_candidate_nodes)
        self._D = np.empty((self.nw.n_nodes, self.nw.n_nodes))
        self._D[self._intervention_candidate_nodes, :] = ((alpha - 1).reshape((n_intervention_candidate_nodes, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))
                                                          + (beta - 1).reshape((n_intervention_candidate_nodes, 1)) @ self._blue_group_id_vector.reshape((1, self.nw.n_nodes))) * self._Sigma[self._intervention_candidate_nodes, :]

    def _compute_intervention_vectors_mix(self):
        self._D = np.empty((self.nw.n_nodes, self.nw.n_nodes))
        intervention_candidate_nodes_gsir = [node for node in self._intervention_candidate_nodes if self._intervention_candidancy_condition_gsir(node)]
        n_intervention_candidate_nodes_gsir = len(intervention_candidate_nodes_gsir)
        alpha = self._phi * (1 - self._gamma[0]) / np.sum(self._Sigma[intervention_candidate_nodes_gsir, :][:, self.nw.red_group], axis = 1)
        beta = (1- self._phi) * (1 - self._gamma[0]) / np.sum(self._Sigma[intervention_candidate_nodes_gsir, :][:, self.nw.blue_group], axis = 1)
        self._D[intervention_candidate_nodes_gsir, :] = ((alpha - 1).reshape((n_intervention_candidate_nodes_gsir, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))
                                                          + (beta - 1).reshape((n_intervention_candidate_nodes_gsir, 1)) @ self._blue_group_id_vector.reshape((1, self.nw.n_nodes))) * self._Sigma[intervention_candidate_nodes_gsir, :]
        intervention_candidate_nodes_pgoi = [node for node in self._intervention_candidate_nodes if not self._intervention_candidancy_condition_gsir(node)]
        n_intervention_candidate_nodes_pgoi = len(intervention_candidate_nodes_pgoi)
        beta = (1- self._phi) * (1 - self._gamma[0]) / np.sum(self._Sigma[intervention_candidate_nodes_pgoi, :][:, self.nw.blue_group], axis = 1)
        delta = self._phi * (1 - self._gamma[0]) - beta * np.sum(self._Sigma[intervention_candidate_nodes_pgoi, :][:, self.nw.red_group], axis = 1)
        self._D[intervention_candidate_nodes_pgoi, :] = (beta - 1).reshape((n_intervention_candidate_nodes_pgoi, 1)) * self._Sigma[intervention_candidate_nodes_pgoi, :]\
            + (delta / self.nw.red_group.shape[0]).reshape((n_intervention_candidate_nodes_pgoi, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))

    def _compute_intervention_vectors_pgoi(self):
        beta = (1- self._phi) * (1 - self._gamma[0]) / np.sum(self._Sigma[self._intervention_candidate_nodes, :][:, self.nw.blue_group], axis = 1)
        delta = self._phi * (1 - self._gamma[0]) - beta * np.sum(self._Sigma[self._intervention_candidate_nodes, :][:, self.nw.red_group], axis = 1)
        n_intervention_candidate_nodes = len(self._intervention_candidate_nodes)
        self._D = np.empty((self.nw.n_nodes, self.nw.n_nodes))
        self._D[self._intervention_candidate_nodes, :] = (beta - 1).reshape((n_intervention_candidate_nodes, 1)) * self._Sigma[self._intervention_candidate_nodes, :]\
            + (delta / self.nw.red_group.shape[0]).reshape((n_intervention_candidate_nodes, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))

    def _compute_intervention_vectors_nrb(self):
        self._D = np.empty((self.nw.n_nodes, self.nw.n_nodes))
        out_degrees = np.sum(self._P_original[self._intervention_candidate_nodes, :], axis = 1)
        out_degrees_to_red_group = np.sum(self._P_original[self._intervention_candidate_nodes, :][:, self.nw.red_group], axis = 1)
        out_degrees_to_blue_group = np.sum(self._P_original[self._intervention_candidate_nodes, :][:, self.nw.blue_group], axis = 1)
        # case 1
        intervention_candidate_node_indices = [i for i in range(len(self._intervention_candidate_nodes)) if out_degrees_to_red_group[i] > 0]
        intervention_candidate_nodes = [self._intervention_candidate_nodes[i] for i in intervention_candidate_node_indices]
        n_intervention_candidate_nodes = len(intervention_candidate_nodes)
        rho = out_degrees_to_red_group[intervention_candidate_node_indices] / out_degrees[intervention_candidate_node_indices]
        beta = out_degrees_to_blue_group[intervention_candidate_node_indices] / out_degrees[intervention_candidate_node_indices]
        self._D[intervention_candidate_nodes, :] = ((self._phi/rho - 1).reshape((n_intervention_candidate_nodes, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))
                                                    + ((1-self._phi)/beta - 1).reshape((n_intervention_candidate_nodes, 1)) @ self._blue_group_id_vector.reshape((1, self.nw.n_nodes))) * self._P_original[intervention_candidate_nodes, :].todense()
        # case 2
        intervention_candidate_node_indices = [i for i in range(len(self._intervention_candidate_nodes)) if out_degrees_to_red_group[i] == 0]
        intervention_candidate_nodes = [self._intervention_candidate_nodes[i] for i in intervention_candidate_node_indices]
        n_intervention_candidate_nodes = len(intervention_candidate_nodes)
        self._D[intervention_candidate_nodes, :] = np.repeat(self._phi / self.nw.red_group.shape[0], n_intervention_candidate_nodes).reshape((n_intervention_candidate_nodes, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))\
            + (- self._phi / out_degrees_to_blue_group[intervention_candidate_node_indices]).reshape((n_intervention_candidate_nodes, 1)) @ self._blue_group_id_vector.reshape((1, self.nw.n_nodes)) * self._P_original[intervention_candidate_nodes, :].todense()
        
    def _compute_intervention_vectors_rrd(self):
        rho = np.sum(self._P_original[self._intervention_candidate_nodes, :][:, self.nw.red_group], axis = 1) / np.sum(self._P_original[self._intervention_candidate_nodes, :], axis = 1)
        delta = (self._phi - rho) / (1 - rho) # candidates always have rho < phi
        n_intervention_candidate_nodes = len(self._intervention_candidate_nodes)
        self._D = np.empty((self.nw.n_nodes, self.nw.n_nodes))
        self._D[self._intervention_candidate_nodes, :] = (- delta).reshape((n_intervention_candidate_nodes, 1)) * self._P_original[self._intervention_candidate_nodes, :]\
            + (delta / self.nw.red_group.shape[0]).reshape((n_intervention_candidate_nodes, 1)) @ self._red_group_id_vector.reshape((1, self.nw.n_nodes))
        
    _compute_intervention_vectors = {
        "gsir": _compute_intervention_vectors_gsir,
        "mix": _compute_intervention_vectors_mix,
        "pgoi": _compute_intervention_vectors_pgoi,
        "nrb": _compute_intervention_vectors_nrb,
        "rrd": _compute_intervention_vectors_rrd
    }

    def _initialize_intervention_attributes(self, config_index):
        self._determine_intervention_candidate_nodes(self.configs[config_index].intervention)
        self._n_intervention_candidate_nodes = len(self._intervention_candidate_nodes)
        if self.configs[config_index].selector != "ref":
            self._intervention_target_node_generator = self._intervention_target_node_generators[self.configs[config_index].selector](self)
        self._red_group_id_vector = np.array([1 if node in self.nw.red_group else 0 for node in self.nw.nodes])
        self._blue_group_id_vector = np.array([1 if node in self.nw.blue_group else 0 for node in self.nw.nodes])
        self._compute_intervention_vectors[self.configs[config_index].intervention](self)

    def _perform_intervention(self, config_index, node):
        # self._Sigma[node, self.nw.red_group] = self._Sigma[node, self.nw.red_group].multiply((self._phi * (1 - self._delta) * (1 - self._gamma[node])) / np.sum(self._Sigma[node, self.nw.red_group]))
        # self._Sigma[node, self.nw.blue_group] = self._Sigma[node, self.nw.blue_group].multiply(((1 - self._phi) * (1 - self._delta) * (1 - self._gamma[node])) / np.sum(self._Sigma[node, self.nw.blue_group]))
        # self._Q = np.linalg.inv(np.eye(self.nw.n_nodes)-self._Sigma) @ self._Gamma
        # self._Sigma[[node], :] += self._D[[node], :]
        # self._Q += (self._Q[:, [node]] @ (self._D[[node], :] @ self._Q)) / (self._gamma[node] - self._D[[node], :] @ self._Q[:, [node]])
        self._P[[node], :] += csr_array(self._D[[node], :])
        if self.configs[config_index].selector in ["defg", "duli", "dgli"]:
            numerator = self._D[[node], :] @ self._Q
            denominator = self._gamma/(1-self._gamma) - np.dot(self._D[node, :], self._Q[:, node])
            self._Q += self._Q[:, [node]] @ numerator / denominator
            self._p += self._p[node] * numerator.reshape((self.nw.n_nodes,)) / denominator
            self._Q__R += self._Q[:, node] * np.dot(self._D[node, :], self._Q__R) / denominator
        elif self.configs[config_index].selector not in ["dpr"]:
            self._compute_pagerank()

    def _perform_run(self,
                     config_index: int,
                     run_index: int):
        if self.settings.verbose:
            print("Run with index " + str(run_index))

        initial_time = timer() # for tracking total running time

        self._phi = self.configs[config_index].phi
        self._gamma = self.configs[config_index].gamma
        # self._gamma = np.ones(self.nw.n_nodes) * self.configs[config_index].gamma
        # s = np.empty(self.nw.n_nodes)
        # for node in self.nw.nodes:
        #     if node in self.nw.red_group:
        #         s[node] = 1
        #     elif node in self.nw.blue_group:
        #         s[node] = -1
        #     else:
        #         s[node] = 0
        # self._Gamma = np.diag(self._gamma)
        # In the element-wise matrix multiplication that follows, values of the second matrix are broadcasted column-wise by numpy
        # self._Sigma = self.nw.adjacency_matrix * ((1 - self._gamma) / np.sum(self.nw.adjacency_matrix, axis = 1)).reshape((self.nw.n_nodes,1))
        I = np.eye(self.nw.n_nodes) # sps.eye_array(self.nw.n_nodes, format = "csr")
        self._v_original = np.ones(self.nw.n_nodes) / self.nw.n_nodes
        out_degrees = np.sum(self.nw.adjacency_matrix, axis = 1)
        adjacency_matrix = deepcopy(self.nw.adjacency_matrix)
        for node in range(self.nw.n_nodes):
            if out_degrees[node] == 0:
                adjacency_matrix[node, :] = np.ones(self.nw.n_nodes)
        self._P_original = csr_array(adjacency_matrix / np.sum(adjacency_matrix, axis = 1).reshape((self.nw.n_nodes,1)))

        start_time = timer()
        # self._Q = np.linalg.inv(I-self._Sigma) @ self._Gamma
        self._Q_original = self._gamma * np.linalg.inv(I - (1-self._gamma) * self._P_original)
        self._Q__R = np.sum(self._Q_original[:, self.nw.red_group], axis = 1)
        self._p_original = self._v_original @ self._Q_original
        red_group_pagerank_original = np.sum(self._p_original[self.nw.red_group])
        # z_original = self._Q @ s
        # red_group_influence_original = self._compute_red_group_influence()
        # end_time = timer()

        if self._phi > 1: # dynamically set target phi
            self._phi = red_group_pagerank_original + 0.1
            print("Dynamically Set Target Phi = " + str(self._phi))

        # print("Computing intervention target nodes and respective intervention vectors...", end = " ", flush = True)
        # start_time = timer()
        self._initialize_intervention_attributes(config_index)
        self._v = (self._phi / self.nw.red_group.shape[0]) * self._red_group_id_vector + ((1 - self._phi) / self.nw.blue_group.shape[0]) * self._blue_group_id_vector # fair jump vector
        end_time = timer()
        # print("Done after", end_time - start_time, "seconds!")

        # DataFrame.from_dict({"Intervention Target": self._intervention_candidate_nodes}).to_csv(sys.path[0] + "/Results/" + self.dataset + "/15/candidates.csv")

        self.results[run_index]["Iteration"].append(0)
        self.results[run_index]["Intervention Target"].append(np.nan)
        # self.results[run_index]["Expected Fairness Gain"].append(.0)
        # self.results[run_index]["Red Group Influence"].append(red_group_influence_original)
        self.results[run_index]["Red Group Pagerank"].append(red_group_pagerank_original)
        self.results[run_index]["Fairness Gain"].append(.0)
        self.results[run_index]["Utility Loss"].append(.0)
        self.results[run_index]["Performance Time"].append(end_time - start_time)
        if self.settings.verbose:
            # print("Iteration " + str(0).rjust(7) + ": Red Group Influence = " + str(red_group_influence_original).ljust(23) + " | Performance Time = " + str(end_time - start_time))
            print("Iteration " + str(0).rjust(7) + ": Red Group Pagerank = " + str(red_group_pagerank_original).ljust(23) + " | Performance Time = " + str(end_time - start_time))

        # testing whether intervening to all candidates is sufficient
        self._P = deepcopy(self._P_original)
        self._p = self._p_original
        start_time = timer()
        self._P[self._intervention_candidate_nodes, :] += csr_array(self._D[self._intervention_candidate_nodes, :])
        self._compute_pagerank()
        # self._Q = self._gamma * np.linalg.inv(I - (1-self._gamma) * self._P)
        # self._p = self._v @ self._Q
        end_time = timer()

        # plotting fairness gain vs utility loss
        # fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(36, 12))

        n_iterations = self._n_intervention_candidate_nodes
        red_group_pagerank = np.sum(self._p[self.nw.red_group])
        fairness_gain = red_group_pagerank - red_group_pagerank_original
        utility_loss = np.dot(self._p - self._p_original, self._p - self._p_original)

        if red_group_pagerank < self._phi or self.configs[config_index].selector == "ref":
            self.results[run_index]["Iteration"].append(n_iterations)
            self.results[run_index]["Intervention Target"].append(np.nan)
            self.results[run_index]["Red Group Pagerank"].append(red_group_pagerank)
            self.results[run_index]["Fairness Gain"].append(fairness_gain)
            self.results[run_index]["Utility Loss"].append(utility_loss)
            self.results[run_index]["Performance Time"].append(end_time - start_time)
            if self.settings.verbose:
                print("Iteration " + str(n_iterations).rjust(7) + ": Red Group Pagerank = " + str(red_group_pagerank).ljust(23) + " | Fairness Gain = " + str(fairness_gain).ljust(23) + " | Utility Cost = " + str(utility_loss).ljust(23) + " | Performance Time = " + str(end_time - start_time))
        else:
        # for i in range(1):
            # red_group_influence = red_group_influence_original
            # z = z_original
            # self._p = deepcopy(self._p_original)
            self._P = deepcopy(self._P_original)
            self._Q = deepcopy(self._Q_original)
            self._p = self._p_original
            self._compute_pagerank()
            # self._p = self._v @ self._Q # fair jump vector pagerank
            n_iterations = 0
            red_group_pagerank = red_group_pagerank_original
            # while red_group_influence < self._phi and len(self._intervention_candidate_nodes) > 0:
            while red_group_pagerank < self._phi and len(self._intervention_candidate_nodes) > 0:
            # while len(self._intervention_candidate_nodes) > 0:
                # if the run times out, we add a dummy row to our results and abort
                if timer() - initial_time > 24 * 60 * 60: # 24 hours in seconds
                    self.results[run_index]["Iteration"].append(np.nan)
                    self.results[run_index]["Intervention Target"].append(np.nan)
                    # self.results[run_index]["Expected Fairness Gain"].append(np.nan)
                    # self.results[run_index]["Red Group Influence"].append(np.nan)
                    self.results[run_index]["Red Group Pagerank"].append(np.nan)
                    self.results[run_index]["Fairness Gain"].append(np.nan)
                    self.results[run_index]["Utility Loss"].append(np.nan)
                    self.results[run_index]["Performance Time"].append(np.nan)
                    break

                # plot fairness gain vs utility loss
                # if n_iterations < 3:
                #     self._compute_exact_fairness_gains()
                #     self._compute_utility_losses()
                #     df = DataFrame({"Fairness Gain": self._fairness_gain, "Utility Loss": self._utility_loss}).dropna(how='all')
                #     print(df.agg({"Fairness Gain": ["min", "max"], "Utility Loss": ["min", "max"]}))
                #     df.plot.scatter(y = "Fairness Gain", x = "Utility Loss", color = plt.cm.rainbow(0.1), ax = ax[n_iterations])
                #     ax[n_iterations].set_xscale("log")
                #     ax[n_iterations].set_yscale("log")

                start_time = timer()
                intervention_target_node = next(self._intervention_target_node_generator) # removes the returned node from the intervention candidate node list
                # fairness_gain = self._compute_fairness_gain(intervention_target_node)
                # utility_loss = self._compute_utility_loss(intervention_target_node)
                self._perform_intervention(config_index, intervention_target_node) # updates P, Q, Q^R and p
                end_time = timer()

                # self._Q = np.linalg.inv((I-self._Sigma).todense() - self._delta * (1 - self._gamma[0]) * self._J) @ self._Gamma
                # self._Q = np.linalg.inv((I-self._Sigma).todense()) @ self._Gamma
                # z = self._Q @ s

                # red_group_influence = self._compute_red_group_influence()
                # fairness_gain = red_group_influence - red_group_influence_original
                # utility_loss = np.linalg.norm(z - z_original) ** 2 / self.nw.n_nodes

                n_iterations += 1
                red_group_pagerank = np.sum(self._p[self.nw.red_group])
                fairness_gain = red_group_pagerank - red_group_pagerank_original
                utility_loss = np.dot(self._p - self._p_original, self._p - self._p_original)

                self.results[run_index]["Iteration"].append(n_iterations)
                self.results[run_index]["Intervention Target"].append(intervention_target_node)
                # self.results[run_index]["Expected Fairness Gain"].append(self._fairness_gain[intervention_target_node])
                # self.results[run_index]["Red Group Influence"].append(red_group_influence)
                self.results[run_index]["Red Group Pagerank"].append(red_group_pagerank)
                self.results[run_index]["Fairness Gain"].append(fairness_gain)
                self.results[run_index]["Utility Loss"].append(utility_loss)
                self.results[run_index]["Performance Time"].append(end_time - start_time)
                if self.settings.verbose:
                    # print("Iteration " + str(n_iterations).rjust(7) + ": Red Group Influence = " + str(red_group_influence).ljust(23) + " | Fairness Gain = " + str(fairness_gain).ljust(23) + " | Utility Cost = " + str(utility_loss).ljust(23) + " | Performance Time = " + str(end_time - start_time))
                    print("Iteration " + str(n_iterations).rjust(7) + ": Red Group Pagerank = " + str(red_group_pagerank).ljust(23) + " | Fairness Gain = " + str(fairness_gain).ljust(23) + " | Utility Cost = " + str(utility_loss).ljust(23) + " | Performance Time = " + str(end_time - start_time))
        if self.settings.verbose:
            if timer() - initial_time > 24 * 60 * 60: # 24 hours in seconds
                print("Run exceeded 24 hours and was terminated!")
            else:
                # print("Achieved a ratio of " + str(red_group_influence) + " for the red group after " + str(n_iterations) + " interventions!")
                print("Achieved a ratio of " + str(red_group_pagerank) + " for the red group after " + str(n_iterations) + " interventions!")
        self.stats["Phi"].append(self._phi)
        self.stats["Gamma"].append(self._gamma)
        # self.stats["Gamma"].append(self._gamma[0])
        # self.stats["Original Red Group Influence"].append(red_group_influence_original)
        # self.stats["Achieved Red Group Influence"].append(red_group_influence)
        self.stats["Original Red Group Pagerank"].append(red_group_pagerank_original)
        self.stats["Achieved Red Group Pagerank"].append(red_group_pagerank)
        self.stats["# Available Intervention Candidates"].append(self._n_intervention_candidate_nodes)
        self.stats["# Selected Intervention Candidates"].append(n_iterations)
        self.stats["Utility Loss"].append(utility_loss)

        # plot fairness gain vs utility loss
        # fig.tight_layout()
        # plt.savefig(sys.path[0] + "/Results/" + self.dataset + "." + digits_after_point(self._gamma) + "." + digits_after_point(self._phi) + "." + self.configs[config_index].intervention + ".fairness_gain_vs_utility_loss.png")

    def _cancel_run(self,
                     config_index: int,
                     run_index: int):
        if self.settings.verbose:
            print("Run with index " + str(run_index) + " was canceled!")
        
        self.results[run_index]["Iteration"].append(np.nan)
        self.results[run_index]["Intervention Target"].append(np.nan)
        # self.results[run_index]["Expected Fairness Gain"].append(np.nan)
        # self.results[run_index]["Red Group Influence"].append(np.nan)
        self.results[run_index]["Red Group Pagerank"].append(np.nan)
        self.results[run_index]["Fairness Gain"].append(np.nan)
        self.results[run_index]["Utility Loss"].append(np.nan)
        self.results[run_index]["Performance Time"].append(np.nan)
    
    def _save_reference(self,
                  config_index: int):
        folder_path = sys.path[0] + "/Results/" + self.dataset + "/" + digits_after_point(self.configs[config_index].gamma)
        # if self.configs[config_index].delta > 0:
        #     folder_path += "/" + digits_after_point(self.configs[config_index].delta)
        file_name = digits_after_point(self.configs[config_index].phi) + "." + str(self.configs[config_index].intervention) + ".ref.0"
        os.makedirs(folder_path, exist_ok = True)
        # DataFrame.from_dict(self.local_values).to_csv(sys.path[0] + "/Results/" + self.dataset + "/influence.csv", index = False)
        DataFrame.from_dict(self.results[0]).to_csv(folder_path + "/" + file_name + ".csv")
        file_name = digits_after_point(self.configs[config_index].phi) + "." + str(self.configs[config_index].intervention) + ".ref.f"
        os.makedirs(folder_path, exist_ok = True)
        # DataFrame.from_dict(self.local_values).to_csv(sys.path[0] + "/Results/" + self.dataset + "/influence.csv", index = False)
        DataFrame.from_dict({"Node": self._intervention_candidate_nodes, "Fairness Gain": self._fairness_gain[self._intervention_candidate_nodes], "Utility Loss": self._utility_loss[self._intervention_candidate_nodes]}).to_csv(folder_path + "/" + file_name + ".csv")
    
    def _save_run(self,
                  config_index: int,
                  run_index: int):
        folder_path = sys.path[0] + "/Results/" + self.dataset + "/" + digits_after_point(self.configs[config_index].gamma)
        # if self.configs[config_index].delta > 0:
        #     folder_path += "/" + digits_after_point(self.configs[config_index].delta)
        file_name = phi_to_str(self.configs[config_index].phi) + "." + str(self.configs[config_index].intervention) + "." + str(self.configs[config_index].selector) + "." + str(run_index)
        os.makedirs(folder_path, exist_ok = True)
        # DataFrame.from_dict(self.local_values).to_csv(sys.path[0] + "/Results/" + self.dataset + "/influence.csv", index = False)
        DataFrame.from_dict(self.results[run_index]).to_csv(folder_path + "/" + file_name + ".csv")

    def _compute_statistics(self,
                            config_index: int):
        n_rows = np.max([len(self.results[run_index]["Iteration"]) for run_index in range(self.configs[config_index].n_runs)])
        for stat_name, stat_func in [("min", np.min), ("max", np.max), ("avg", np.average)]:
            self.results[stat_name] = dict(self._results_template)
            for column in self._results_template.keys():
                self.results[stat_name][column] = [np.nan]
            for column in ["Iteration", "Red Group Pagerank", "Fairness Gain", "Utility Loss"]:
                # self.results[stat_name][column] = [stat_func([self.results[run_index][column][i] for run_index in range(self.configs[config_index].n_runs) if len(self.results[run_index][column]) > i]) for i in range(n_rows)]
                self.results[stat_name][column] = [stat_func([self.results[run_index][column][-1] for run_index in range(self.configs[config_index].n_runs)])]
            # self.results[stat_name]["Iteration"] = list(range(n_rows))
            # self.results[stat_name]["Intervention Target"] = [np.nan for i in range(n_rows)]
    
    def _save_statistics(self,
                         config_index: int):
        folder_path = sys.path[0] + "/Results/" + self.dataset + "/" + digits_after_point(self.configs[config_index].gamma)
        # if self.configs[config_index].delta > 0:
        #     folder_path += "/" + digits_after_point(self.configs[config_index].delta)
        os.makedirs(folder_path, exist_ok = True)
        for stat in ["min", "max", "avg"]:
            file_name = phi_to_str(self.configs[config_index].phi) + "." + str(self.configs[config_index].intervention) + "." + str(self.configs[config_index].selector) + "." + stat
            DataFrame.from_dict(self.results[stat]).to_csv(folder_path + "/" + file_name + ".csv")

    def _perform(self,
                 config_index: int):
        self.results = {run_index: deepcopy(self._results_template) for run_index in range(self.configs[config_index].n_runs)}

        if self.settings.verbose:
            print("Fair Opinion Formation via node interventions on the " + str(self.dataset) + " dataset with gamma = " + str(self.configs[config_index].gamma))
            print("Target phi = " + str(self.configs[config_index].phi) + ", Intervention Target Selector = " + self.selector_descriptions[self.configs[config_index].selector])
            print("Intervention Procedure = " + self.intervention_descriptions[self.configs[config_index].intervention])
        timed_out = False
        for run_index in range(self.configs[config_index].n_runs):
            if not timed_out:
                self._perform_run(config_index, run_index)
            else:
                self._cancel_run(config_index, run_index)
            if self.settings.save:
                self._save_run(config_index, run_index)
            if self.results[run_index]["Iteration"] == np.nan:
                timed_out = True # if a run timed out, we cancel subsequent runs
        if self.settings.save and self.configs[config_index].n_runs > 1:
            self._compute_statistics(config_index)
            self._save_statistics(config_index)

    def perform(self):
        if self.settings.verbose:
            for config_index in range(len(self.configs)):
                self._perform(config_index)
        else:
            for config_index in tqdm(list(range(len(self.configs))), desc = self.description.rjust(23)):
                self._perform(config_index)
    
    def save(self):
        os.makedirs(sys.path[0] + "/Results/" + self.dataset, exist_ok = True)
        DataFrame.from_dict(self.local_values).to_csv(sys.path[0] + "/Results/" + self.dataset + "/influence.csv", index = False)
        if len(self.stats["Phi"]) == 0:
            print("Nothing to save...")
        if len(self.stats["Phi"]) == 1:
            self._save_run(0, 0)
        else:
            DataFrame.from_dict(self.stats).to_csv(sys.path[0] + "/Results/" + self.dataset + "/results_comparison.csv")

# Our predefined experiments
# vanilla_synthetic_dataset_configurations = {}
# for i in range(1, 10):
#     P = [[.1, .01 * i], [.01 * i, .1]]
#     name = "synthetic_1K_" + "-".join([str(int(p * 100)) for row in P for p in row])
#     # vanilla_synthetic_dataset_configurations[name] = Configuration(name, [.5, .6, .7, .8, .9], np.arange(0.1, 1.01, 0.1), [1.], "Greedy", True, name + " for 5 Phi * 10 Gamma")
#     vanilla_synthetic_dataset_configurations[name] = Experiment(name, [.6], [.1], [.1], ["Greedy"], True, name + " for Greedy")
# for i in range(1, 10):
#     P = [[.1, .01 * i], [.01 * (10 - i), .1]]
#     name = "synthetic_1K_" + "-".join([str(int(p * 100)) for row in P for p in row])
#     # vanilla_synthetic_dataset_configurations[name] = Configuration(name, [.1, .3, .5, .7, .9], np.arange(0.1, 1.01, 0.1), [1.], "Greedy", True, name + " for 5 Phi * 10 Gamma")
#     vanilla_synthetic_dataset_configurations[name] = Configuration(name, [.1], [.5], [.5], "Greedy", True, name + " for Influence")
# symmetric_synthetic_datasets = []
# for i in range(1, 6): # from 1 to 5
#     R = [.3, .7]
#     P = [[.3 - .05 * i, .05 * i], [.05 * i, .7 - .05 * i]]
#     symmetric_synthetic_datasets.append("synthetic_1K_" + "-".join([str(int(p * 100)) for p in R]) + "_" + "-".join([str(int(p * 100)) for row in P for p in row]))
# asymmetric_synthetic_datasets = {}
# for i in range(1, 10): # from 1 to 9
#     R = [.3, .7]
#     P = [[.25, .01 * i], [.01 * (10 - i), .65]]
#     asymmetric_synthetic_datasets.append("synthetic_1K_" + "-".join([str(int(p * 100)) for p in R]) + "_" + "-".join([str(int(p * 100)) for row in P for p in row]))
R = [.5, .5]
symmetric_synthetic_datasets = []
for i in range(1, 10): # from 1 to 9
    P = [[.1, .01 * i], [.01 * i, .1]]
    symmetric_synthetic_datasets.append("synthetic_1K_" + "-".join([str(int(p * 100)) for p in R]) + "_" + "-".join([str(int(p * 100)) for row in P for p in row]))
asymmetric_synthetic_datasets = []
for i in range(1, 10): # from 1 to 9
    P = [[.1, .01 * i], [.01 * (10 - i), .1]]
    asymmetric_synthetic_datasets.append("synthetic_1K_" + "-".join([str(int(p * 100)) for p in R]) + "_" + "-".join([str(int(p * 100)) for row in P for p in row]))
synthetic_MPA_datasets = {"symmetric": [], "asymmetric": []}
for i in range(1,10):
    for j in range(5):
	    synthetic_MPA_datasets["symmetric"].append("synthetic_MPA-5-" + str(i) + "-" + str(i) + "_" + str(j))
for i in range(1,5):
	for j in range(5):
		synthetic_MPA_datasets["asymmetric"].append("synthetic_MPA-5-" + str(i) + "-5_" + str(j))
for j in range(5):
    synthetic_MPA_datasets["asymmetric"].append("synthetic_MPA-5-5-5_" + str(j))
for i in range(6,10):
	for j in range(5):
		synthetic_MPA_datasets["asymmetric"].append("synthetic_MPA-5-5-" + str(i) + "_" + str(j))
synthetic_GPA_datasets = {"homophily": [], "red homophily": [], "blue homophily": [], "red size": [], "scale": []}
for i in range(1,10):
    for j in range(10):
	    synthetic_GPA_datasets["homophily"].append("synthetic_GPA-5-" + str(i) + "-" + str(i) + "_" + str(j))
for i in range(1,10):
	for j in range(10):
		synthetic_GPA_datasets["red homophily"].append("synthetic_GPA-5-" + str(i) + "-5_" + str(j))
for i in range(1,10):
	for j in range(10):
		synthetic_GPA_datasets["blue homophily"].append("synthetic_GPA-5-5-" + str(i) + "_" + str(j))
for i in range(1,5):
	for j in range(10):
		synthetic_GPA_datasets["red size"].append("synthetic_GPA-" + str(i) + "-7-7_" + str(j))
for i in [2**k for k in range(4)]:
	for j in range(10):
		synthetic_GPA_datasets["scale"].append("synthetic_GPA-" + str(i) + "K-3-7-7_" + str(j))