import os
import networkx as nx
# import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

def all_pairs(l):
	n = len(l)
	final = set([])
	for i in range(0, n):
		for j in range(0, n):
			if (not ((l[j], l[i]) in final)) and (l[i] != l[j]):
				final.add((l[i], l[j]))
	return final

all_user_pairs = set([])

keywords_magafuli = ["john","magufuli", "pombe", "ccm", "chama", 
			         "cha", "mapinduz", "president", "rais"]

keywords_corona = ["corona", "virus", "covid", "covid-19", "pandemic", 
                   "sars‑cov‑2", "janga", "kubwa", "ventilator", "ventilators", 
                   "masks", "kiingilizi", "virusi"]

def create_graph(keywords):
	nodes = set([])
	edges = set([])

	for i, filename in enumerate(os.listdir("JF_Data")):
		if filename.endswith("graph.txt"):
			f = open("JF_Data/" + filename, "r", encoding="utf-8")
			# f_cleaned =  open("Cleaned/" + filename[:-4] + "_cleaned.txt", "r", encoding="utf-8")
			try:
				# print(filename[:-10])
				f_cleaned = open("Filtered/" + filename[:-10] + "_data.txt", "r", encoding="utf-8")
			except FileNotFoundError:
				continue

			f_lines = f_cleaned.readlines()
			top = f.readline()

			main_guy = top.split(":")[0]
			nodes.add(main_guy)

			for line in f_lines:
				if len(line) > 1:
					line_split = line.split(",")
					score = int(line_split[2])
					sentence = line_split[3]

					tokens = word_tokenize(sentence)
					if (len(list(filter(lambda x : x in keywords, tokens))) > 0):
						nodes.add(line_split[0])
						edges.add((main_guy, line_split[0], score))
				
			f.close()
			f_cleaned.close()

	nodes_list = list(nodes)
	edge_weights = list(edges)
	# edges_list = list(map(lambda t: (t[0], t[1]), edges_list_temp))
	# edges_weights = list(map(lambda t: t[2], edges_list_temp))

	return nodes_list, edge_weights

def top_n(d, n):
	l = list(d.items())
	l.sort(reverse = True, key = lambda x: x[1])
	return list(map(lambda x: x[0], l[:n]))

# Slides from https://www.cl.cam.ac.uk/teaching/1314/L109/tutorial.pdf
# were very helpful here.
def analyze(nodes, edge_weights, filename = "graph.txt", num_users = 10):
	f = open(filename, "w+", encoding="utf-8")
	# nodes_i = range(len(nodes))

	# look_up = dict(zip(nodes, nodes_i))
	# edges_i = list(map(lambda t: (look_up[t[0]], look_up[t[1]]), edges))

	G = nx.DiGraph()
	G.add_nodes_from(nodes)
	# G.add_edges_from(edges)
	G.add_weighted_edges_from(edge_weights)

	num_scc = nx.number_strongly_connected_components(G)
	f.write("Number of Strongly Connected Components: " + str(num_scc) + "\n")
	
	num_wcc = nx.number_weakly_connected_components(G)
	f.write("Number of Weakly Connected Components: " + str(num_wcc) + "\n")
	
	# Largest component of G
	uF = G.to_undirected()
	F = max((G.subgraph(c) for c in nx.connected_components(uF)), key=len)

	# A measure of how much nodes cluster together
	clustering_coef = nx.average_clustering(uF)
	f.write("Clustering Coefficient: " + str(clustering_coef) + "\n")

	nodes_out_degrees = G.out_degree()
	df_out = pd.DataFrame.from_dict({"Nodes": list(map(lambda t: t[0], nodes_out_degrees)),
								 "Out_Degree": list(map(lambda t: t[1], nodes_out_degrees))})
	ax_out = df_out.hist(column="Out_Degree")
	ax_out[0][0].set_yscale('log')

	fig_out = ax_out[0][0].get_figure()
	fig_out.savefig(filename+"_Out_Degree_Histogram.pdf")

	nodes_in_degrees = G.in_degree()
	df_in = pd.DataFrame.from_dict({"Nodes": list(map(lambda t: t[0], nodes_in_degrees)),
							 "In_Degree": list(map(lambda t: t[1], nodes_in_degrees))})
	
	ax_in = df_in.hist(column="In_Degree")
	ax_in[0][0].set_yscale('log')
	fig_in = ax_in[0][0].get_figure()
	fig_in.savefig(filename+"_In_Degree_Histogram.pdf")

	# Compute central nodes to the network
	nodes_out_degrees = G.out_degree()
	weighted_out_degrees = {}
	
	total = 0
	for node in nodes:
		total = 0
		for neighbor in G[node]:
			total += G[node][neighbor]["weight"]
		weighted_out_degrees[node] = total
		
	eigenvector_centrality = nx.eigenvector_centrality(F)
	
	df_eigen = pd.DataFrame.from_dict({"Nodes": list(eigenvector_centrality.keys()),
								 	   "Centrality": list(eigenvector_centrality.values())})
	ax_eigen = df_eigen.hist(column="Centrality")
	ax_eigen[0][0].set_yscale('log')

	fig_eigen = ax_eigen[0][0].get_figure()
	fig_eigen.savefig(filename+"_Centrality_Histogram.pdf")

	l = top_n(eigenvector_centrality, n = num_users)
	l1 = list(map(lambda name: (name, weighted_out_degrees[name]), l))

	def tuple_list_to_string(l):
		final = []
		for t in l:
			final.append(t[0] + " with total setiment: " + str(t[1]))
		return final

	f.write("The top " + str(num_users) + " users are: " + 
			", ".join(tuple_list_to_string(l1)) + "\n")

	f.write(str(nx.info(G)))

	f.close()

	return 0 

m_nodes, m_edge_weights = create_graph(keywords_magafuli)
c_nodes, c_edge_weights = create_graph(keywords_corona)
analyze(m_nodes, m_edge_weights, filename = "Magufuli_Graph_Analysis.txt")
analyze(c_nodes, c_edge_weights, filename = "COVID_Graph_Analysis.txt")

# print(g.maxdegree())
# print(g.vs.select(_degree = g.vs.degree())["name"])
	# all_user_pairs = all_user_pairs.union(users)

# f1 = open("graph_magafuli.txt", "w+", encoding="utf-8")
# f2 = open("graph_covid.txt", "w+", encoding="utf-8")

# for (user_1, user_2), score in zip(m_edges_list, m_edges_weights) :
# 	f1.write(user_1 + "," + user_2.replace("\n","") + "," + str(score) + "\n")

# for (user_1, user_2), score in zip(c_edges_list, c_edges_weights):
# 	f2.write(user_1 + "," + user_2.replace("\n","") + "," + str(score) + "\n")

# f1.close()
# f2.close()
