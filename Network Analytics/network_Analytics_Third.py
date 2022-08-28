# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:55:37 2022

@author: Dell
"""



import networkx as nx
 #import matplotlib.pyplot as plt
# import networkx library

 
# create an empty undirected graph
G = nx.Graph()
 
# adding edge in graph G
G.add_edge(1, 2)
G.add_edge(2, 3, weight=0.9)

# import matplotlib.pyplot library
import matplotlib.pyplot as plt
 
# import networkx library
import networkx as nx
 
# create a cubical empty graph
G = nx.cubical_graph()
 
# plotting the graph
plt.subplot(122)
 
# draw a graph with red
# node and value edge color
nx.draw(G, pos = nx.circular_layout(G),node_color = 'r',edge_color = 'b')
# node and value edge color
nx.draw(G, pos = nx.circular_layout(G),
        node_color = 'r',
        edge_color = 'b')
# create a cubical empty graph




G = nx.cubical_graph()
 # adding edge in graph G
G.add_edge(1, 2)
G.add_edge(1, 9)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4,5)
G.add_edge(5, 6)
G.add_edge(6, 7)
G.add_edge(7, 8)
G.add_edge(8, 9)
# plotting the graph
plt.subplot(122)
 
# draw a graph with red
# node and value edge color
nx.draw(G, pos = nx.circular_layout(G),
        node_color = 'r',
        edge_color = 'b')

g=nx.from_pandas_edgelist(facebook,source='',target='9')
print(nx.info(g))
d=nx.degree_centrality(g)
print(d)
pos=nx.spring_layout(g)
nx.draw_networkx(g,pos,node_size=25,node_color='blue')
# to check closeness centrality
closeness=nx.closeness_centrality(g)
print(closeness)
# to check betweenness centrality
b=nx.betweenness_centrality(g)
print(b)
#Average clustering
cc=nx.average_clustering(g)
print(cc)
