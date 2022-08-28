# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:55:37 2022

@author: Dell
"""


import pandas as pd
import networkx as nx
G=pd.read_csv("c:/360DG/Datasets/routes.csv")
G=G.iloc[0:500,1:10]
G.info()
g=nx.Graph()
g=nx.from_pandas_edgelist(G,source='Source Airport',target='Destination Airport')
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
