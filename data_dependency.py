import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from randmise import calculate_max_min_absolute_values, get_top_abs_correlations


def load_csv():
    csv_list = []
    p1 = pd.read_csv('D:\Thesis\Sensitivity analysis\P1 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p1)
    q1 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q1 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q1)
    p2 = pd.read_csv('D:\Thesis\Sensitivity analysis\P2 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p2)
    q2 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q2 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q2)
    p3 = pd.read_csv('D:\Thesis\Sensitivity analysis\P3 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p3)
    q3 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q3 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q3)
    p4 = pd.read_csv('D:\Thesis\Sensitivity analysis\P4 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p4)
    q4 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q4 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q4)
    p5 = pd.read_csv('D:\Thesis\Sensitivity analysis\P5 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p5)
    q5 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q5 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q5)
    p6 = pd.read_csv('D:\Thesis\Sensitivity analysis\P6 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p6)
    q6 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q6 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q6)
    p7 = pd.read_csv('D:\Thesis\Sensitivity analysis\P7 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p7)
    q7 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q7 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q7)
    p8 = pd.read_csv('D:\Thesis\Sensitivity analysis\P8 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p8)
    q8 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q8 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q8)
    p9 = pd.read_csv('D:\Thesis\Sensitivity analysis\P9 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p9)
    q9 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q9 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q9)
    p10 = pd.read_csv('D:\Thesis\Sensitivity analysis\P10 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(p10)
    q10 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q10 Sampled sensitivity analysis_constant from PF.csv')
    csv_list.append(q10)

    return csv_list


def data_corelation_spring_layout(file):

    data = pd.read_csv(file)

    df = get_top_abs_correlations(data)

    G=nx.from_pandas_edgelist(df, 'var1', 'var2')

    for i in range(11):
        for j in range(11):
            if ((G.has_edge('{}{}'.format("p", i), '{}{}'.format("p", j)))):
                G.remove_edge('{}{}'.format("p", i), '{}{}'.format("p", j))
            elif (G.has_edge('{}{}'.format("q", i), '{}{}'.format("q", j))):
                G.remove_edge('{}{}'.format("q", i), '{}{}'.format("q", j))
    g = G.to_directed()

    pos = nx.spring_layout(g, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(40, 40))
    nx.draw(g, pos=pos)
    nx.draw_networkx_labels(g, pos=pos,arrows=True)
    plt.show()
    return g,df


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

def data_corelation_network_grid_layout():

    csv_list = load_csv()
    corr_list = []
    name_list = ['p1','q1','p2','q2','p3','q3','p4','q4','p5','q5','p6','q6','p7','q7','p8','q8','p9','q9','p10','q10']
    final_df = []


    for df_name in csv_list:
        df = df_name.corr()
        corr_list.append(df)


    for corr,name in zip(corr_list,name_list):

        links = corr.stack().reset_index()
        links.columns = ['var1', 'var2', 'value']

        links_filtered = links.loc[(links['var1'] != links['var2'])]

        dataf = pd.DataFrame(links_filtered.loc[(links_filtered['var1'] == name)])
        dataf['value'] = dataf.value.abs()
        dataf = dataf.sort_values('value').tail(5)
        print(dataf)
        final_df.append(dataf)

    print(final_df)
    graph_df = pd.concat(final_df)
    graph_df.to_csv('D:\Thesis\Combined_sensitivity_correlation.csv')

    G = nx.from_pandas_edgelist(graph_df, 'var1', 'var2')

    for i in range(11):
        for j in range(11):
            if ((G.has_edge('{}{}'.format("p", i), '{}{}'.format("p", j)))):
                G.remove_edge('{}{}'.format("p", i), '{}{}'.format("p", j))
            elif (G.has_edge('{}{}'.format("q", i), '{}{}'.format("q", j))):
                G.remove_edge('{}{}'.format("q", i), '{}{}'.format("q", j))

    g = G.to_directed()

    pos = nx.spring_layout(g, k=0.3 * 1 / np.sqrt(len(G.nodes())), iterations=20)

    plt.figure(3, figsize=(20, 20))
    nx.draw(g, pos=pos)
    nx.draw_networkx_labels(g, pos=pos, arrows=True)
    plt.show()

def data_absolute_diff_network_grid_layout():

     df = calculate_max_min_absolute_values()

     #normalized_df[df_name] = (df[df_name] - df[df_name].min()) / (df[df_name].max() - df[df_name].min())

     column_name = list(df)
     final_val = get_top_abs_correlations(df)
     #print(final_val)





     # graph_df = pd.concat(final_df)
     # graph_df.to_csv('D:\Thesis\min_max_top_correlation.csv')
     G = nx.convert_matrix.from_pandas_edgelist(final_val, 'var1', 'var2')

     for i in range(11):
         for j in range(11):
             if ((G.has_edge('{}{}'.format("p", i), '{}{}'.format("p", j)))):
                 G.remove_edge('{}{}'.format("p", i), '{}{}'.format("p", j))
             elif (G.has_edge('{}{}'.format("q", i), '{}{}'.format("q", j))):
                 G.remove_edge('{}{}'.format("q", i), '{}{}'.format("q", j))

     g = G.to_directed()
     pos = nx.spring_layout(g, k=0.3 * 1 / np.sqrt(len(G.nodes())), iterations=20)

     plt.figure(3, figsize=(20, 20))
     nx.draw(g, pos=pos)
     nx.draw_networkx_labels(g, pos=pos, arrows=True)
     plt.show()



     #final_df.to_csv("D:\Thesis\Absolute_diff__min_max_sensitivity_analysis.csv")






#data_corelation_spring_layout("D:\Thesis\Sampled monte carlo Data from PF.csv",0.3)
#data_corelation_spring_layout("D:\Thesis\Sampled Realtime Data from PF.csv",0.5)
#data_absolute_diff_network_grid_layout()




