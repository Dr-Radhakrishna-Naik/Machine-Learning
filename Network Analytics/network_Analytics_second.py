# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:55:37 2022

@author: Dell
"""


import pandas as pd
import networkx as nx
G=pd.read_csv("c:/360DG/Datasets/flight_hault.csv")
G.shape
G.columns
Flight_hault=G.rename(({"1":"ID","Goroka":"Name","Goroka.1":"City","Papua New Guinea":"Country","GKA":"IATA_FAA","AYGA":"ICAO","-6.081689":"Latitude","145.391881":"Longitude","5282":"Altitude","10":"Time"}),axis=1)

Flight_hault=Flight_hault.iloc[0:500,:]
Flight_hault.info()
g=nx.Graph()
Flight_hault.columns
g=nx.from_pandas_edgelist(Flight_hault,source='Name',target='Pacific/Port_Moresby')
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
