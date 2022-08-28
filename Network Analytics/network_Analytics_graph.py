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
import matplotlib.pyplot as plt
# face book
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

######################################
G = nx.star_graph(8)
# adding edge in graph G
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)
G.add_edge(1, 5)
G.add_edge(1, 6)
G.add_edge(1, 7)
G.add_edge(1, 8)

# illustrate graph
nx.draw(G, node_color = 'green',
        node_size = 100)
######################################
####linkedin star graph
G = nx.star_graph(23)
# adding edge in graph G
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(3, 4)
G.add_edge(2, 3)
G.add_edge(2, 4)
G.add_edge(4, 1)
G.add_edge(5, 2)
G.add_edge(5, 3)
G.add_edge(5, 13)
G.add_edge(6, 7)
G.add_edge(7, 8)
G.add_edge(8, 5)
G.add_edge(8, 6)
G.add_edge(9, 6)
G.add_edge(9, 7)
G.add_edge(9, 13)
G.add_edge(10, 11)
G.add_edge(11, 12)
G.add_edge(12, 9)
G.add_edge(12, 10)
G.add_edge(13, 10)
G.add_edge(13, 11)
# illustrate graph
nx.draw(G, node_color = 'green',
        node_size = 100)










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
