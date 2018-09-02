import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def data_corelation_spring_layout(file,value):

    df = pd.read_csv(file)

    correlation = df.corr()
    links = correlation.stack().reset_index()

    print(links)
    links.columns = ['var1', 'var2','value']

    links_filtered=links.loc[ (links['value'] > value) & (links['var1'] != links['var2']) ]
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(40, 40))
    nx.draw(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos)
    plt.show()


def data_dependency_kamada_kawai_layout(file,value):
    df = pd.read_csv(file)

    correlation = df.corr()
    links = correlation.stack().reset_index()

    print(links)
    links.columns = ['var1', 'var2', 'value']

    links_filtered = links.loc[(links['value'] > value) & (links['var1'] != links['var2'])]
    G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

    pos = nx.kamada_kawai_layout(G)
    plt.figure(3, figsize=(40, 40))
    nx.draw(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos)
    plt.show()



data_dependency_kamada_kawai_layout("D:\Thesis\Sampled monte carlo Data from PF.csv",1)
data_dependency_kamada_kawai_layout("D:\Thesis\Sampled Realtime Data from PF.csv",0.8)


# data_corelation_spring_layout("D:\Thesis\Sampled monte carlo Data from PF.csv",1)
# data_corelation_spring_layout("D:\Thesis\Sampled Realtime Data from PF.csv",0.8)




