import os, sys, numpy as np, pandas as pd
from collections.abc import Sequence

_assign_random_label = lambda rng, probability: 0 if rng.random() < probability else 1

def _save_graph(labels, adjacency_list, path):
	os.makedirs(path, exist_ok = True) # creating path if it does not exist
	
	# save nodes
	pd.DataFrame.from_dict({"Node": list(range(len(labels))), "Label": labels})\
		.to_csv(path + "/nodes.csv", index = False)
	
	# save edges
	source_list = []; target_list = []
	for source in range(len(adjacency_list)):
		source_list += [source] * len(adjacency_list[source])
		target_list += adjacency_list[source]
	pd.DataFrame.from_dict({"Source": source_list, "Target": target_list})\
		.to_csv(path + "/edges.csv", index = False)

def _print_graph_characteristics(labels, adjacency_list):
	nodes = list(range(len(labels)))
	red_nodes = [node for node in nodes if labels[node] == 0]
	blue_nodes = [node for node in nodes if labels[node] == 1]
	
	n_edges = np.sum([len(adjacency_list[node]) for node in nodes]) / 2
	# n_edges = np.sum([len([other_node for other_node in adjacency_list[this_node] if this_node in adjacency_list[other_node]]) for this_node in nodes]) / 2
	n_red_edges = np.sum([len([other_node for other_node in adjacency_list[this_node] if other_node in red_nodes]) for this_node in red_nodes]) / 2
	n_blue_edges = np.sum([len([other_node for other_node in adjacency_list[this_node] if other_node in blue_nodes]) for this_node in blue_nodes]) / 2
	n_mixed_edges = np.sum([len([other_node for other_node in adjacency_list[this_node] if other_node in blue_nodes]) for this_node in red_nodes])
	
	print("Constructed graph has the following characteristics:")
	print(str(len(nodes)) + " nodes, " + str(n_edges) + " edges")
	print(str(len(red_nodes)) + " red nodes, " + str(len(blue_nodes)) + " blue nodes")
	print(str(n_red_edges) + " red edges, " + str(n_blue_edges) + " blue edges, " + str(n_mixed_edges) + " mixed edges")
	


def generate_network(
	expected_ratio: float = .5, # of the red group
	expected_homophily: Sequence[Sequence[float]] = [[.5, .5], [.5, .5]], # of the network
	n_nodes_final: int = 1000,
	max_degree_initial: int = 10,
	path: str | None = None
):
	rng = np.random.default_rng(seed = None) # initializing the random number generator
	
	labels = [] # 0 for red, 1 for blue
	adjacency_list = []
	
	# construct the initial graph
	n_nodes_initial = max_degree_initial + 1
	for this_node in range(n_nodes_initial):
		labels.append(_assign_random_label(rng, expected_ratio))
		adjacency_list.append(list(range(this_node)) + list(range(this_node + 1, n_nodes_initial))) # initial nodes form a clique
		
    # construct the rest of the graph at random
	for this_node in range(n_nodes_initial, n_nodes_final):
		all_other_nodes = np.array(range(this_node))
		this_label = _assign_random_label(rng, expected_ratio)
		
		other_nodes = []
		for i in range(max_degree_initial): # add max_degree_initial edges
			other_label = _assign_random_label(rng, expected_homophily[this_label][0])
			one_hot_neighbors = [other_nodes.count(other_node) for other_node in all_other_nodes]
			non_neighbors = all_other_nodes[(np.array(labels) == other_label) & (np.array(one_hot_neighbors) == 0)] # other group non neighbors
			if non_neighbors.shape[0] == 0: # if no other group non neighbors exist
				non_neighbors = all_other_nodes[np.array(one_hot_neighbors) == 0] # switch to all non neighbors
			degrees = np.array([len(adjacency_list[node]) for node in non_neighbors])
			preferences = degrees / np.sum(degrees)
			other_nodes.append(rng.choice(non_neighbors, p = preferences))
		
		labels.append(this_label)
		adjacency_list.append(sorted(other_nodes))
		for other_node in adjacency_list[this_node]:
			adjacency_list[other_node].append(this_node)
	
	# save the constructed graph
	if path is None:
		name = "synthetic"
		path = sys.path[0] + "/Datasets/" + name
	_save_graph(labels, adjacency_list, path)
	
	# report the constructed graph characteristics
	_print_graph_characteristics(labels, adjacency_list)

# # homophily
# for i in range(1,10):
# 	p = np.round(i * .1, 1)
# 	h = [[p, 1 - p], [1 - p, p]]
# 	for j in range(10):
# 		path = sys.path[0] + "/Datasets/synthetic_GPA-5-" + str(i) + "-" + str(i) + "_" + str(j)
# 		generate_network(.5, h, path = path)
# # red homophily
# for i in range(1,10):
# 	p = np.round(i * .1, 1)
# 	h = [[p, 1 - p], [.5, .5]]
# 	for j in range(10):
# 		path = sys.path[0] + "/Datasets/synthetic_GPA-5-" + str(i) + "-5_" + str(j)
# 		generate_network(.5, h, path = path)
# # blue homophily
# for i in range(1,10):
# 	p = np.round(i * .1, 1)
# 	h = [[.5, .5], [1 - p, p]]
# 	for j in range(10):
# 		path = sys.path[0] + "/Datasets/synthetic_GPA-5-5-" + str(i) + "_" + str(j)
# 		generate_network(.5, h, path = path)

# # red size
# for i in range(1,5):
# 	r = np.round(i * .1, 1)
# 	p = .7
# 	h = [[p, 1 - p], [1 - p, p]]
# 	for j in range(10):
# 		path = sys.path[0] + "/Datasets/synthetic_GPA-" + str(i) + "-7-7_" + str(j)
# 		generate_network(r, h, path = path)

# scale
for i in [2**k for k in range(4)]:
	r = .3
	p = .7
	h = [[p, 1 - p], [1 - p, p]]
	for j in range(10):
		path = sys.path[0] + "/Datasets/synthetic_GPA-" + str(i) + "K-3-7-7_" + str(j)
		generate_network(r, h, i * 1000, path = path)