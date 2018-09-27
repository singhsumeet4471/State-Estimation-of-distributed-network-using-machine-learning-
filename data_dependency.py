import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def data_corelation_spring_layout(file,value):

    df = pd.read_csv(file)

    correlation = df.corr()
    #correlation.to_csv('D:\Thesis\Sampled monte carlo Data_correaltion_revised.csv')
    links = correlation.stack().reset_index()

    print(links)
    links.columns = ['var1', 'var2','value']

    links_filtered=links.loc[ (links['value'] > value) & (links['var1'] != links['var2']) ]
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    g = G.to_directed()
    gen = nx.dfs_tree(g)
    print(nx.algorithms.dag.is_directed_acyclic_graph(gen))


    pos = nx.spring_layout(gen, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(40, 40))
    nx.draw(gen, pos=pos)
    nx.draw_networkx_labels(gen, pos=pos,arrows=True)
    plt.show()


def data_dependency_kamada_kawai_layout(file,value):
    df = pd.read_csv(file)

    correlation = df.corr()
    links = correlation.stack().reset_index()

    print(links)
    links.columns = ['var1', 'var2', 'value']

    links_filtered = links.loc[(links['value'] > value) & (links['var1'] != links['var2'])]
    G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    g = G.to_directed()
    gen = nx.dfs_tree(g)
    print(nx.algorithms.dag.is_directed_acyclic_graph(gen))

    pos = nx.kamada_kawai_layout(gen)
    plt.figure(3, figsize=(40, 40))
    nx.draw(gen, pos=pos)
    nx.draw_networkx_labels(gen, pos=pos)
    plt.show()

def data_corelation_network_grid_layout(file,value):

    df = pd.read_csv(file)
    correlation = df.corr()

    links = correlation.stack().reset_index()

    #print(links)


    links_filtered=links.loc[ (links['value'] > value) & (links['var1'] != links['var2']) ]


    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')



    for i in range(11):
        for j in range (11):
            if ((G.has_edge('{}{}'.format("p", i),'{}{}'.format("p", j))) ):
                G.remove_edge('{}{}'.format("p", i),'{}{}'.format("p", j))
            elif(G.has_edge('{}{}'.format("q", i),'{}{}'.format("q", j))):
                G.remove_edge('{}{}'.format("q", i), '{}{}'.format("q", j))
            elif (G.has_edge('{}{}'.format("p", i), '{}{}'.format("q", j))):
                G.remove_edge('{}{}'.format("p", i), '{}{}'.format("q", j))

    g = G.to_directed()
    pos = nx.spring_layout(g, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(40, 40))
    nx.draw(g, pos=pos)
    nx.draw_networkx_labels(g, pos=pos,arrows=True)
    plt.show()

#data_dependency_kamada_kawai_layout("D:\Thesis\Sampled monte carlo Data from PF.csv",1)
#data_dependency_kamada_kawai_layout("D:\Thesis\Sampled Realtime Data from PF.csv",0.8)


#data_corelation_spring_layout("D:\Thesis\Sampled monte carlo Data from PF.csv",0.3)
#data_corelation_spring_layout("D:\Thesis\Sampled Realtime Data from PF.csv",0.5)
data_corelation_network_grid_layout("D:\Thesis\Sampled sensitivity analysis from PF.csv",0.5)




