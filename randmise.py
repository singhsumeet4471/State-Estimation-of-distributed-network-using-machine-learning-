import pandas as pd


def merge(plist, qlist, index=None):
    """Returns list of unique pairs of events and actors, none of which may on index"""
    if len(plist) == 0:
        return []
    if index is None:
        index = set()

    merged = None
    tried_pairs = set()
    for pvalue in plist:
        for i, qvalue in enumerate(qlist):
            pair = (pvalue, qvalue)
            if pair not in index and pair not in tried_pairs:
                new_index = index.union([pair])
                rest = merge(plist[1:], qlist[:i]+qlist[i+1:], new_index)

                if rest is not None:
                    # Found! Done.
                    merged = [pair] + rest
                    break
                else:
                    tried_pairs.add(pair)
        if merged is not None:
            break

    return merged


def add_csv():

    csv1 =  pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL1.csv")
    pf1 = pd.DataFrame(csv1)
    csv2 = pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL2.csv")
    pf2 = pd.DataFrame(csv2)
    csv3 = pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL3.csv")
    pf3 = pd.DataFrame(csv3)

    combined_pf = pf1 + pf2.values
    combined_df = combined_pf + pf3.values
    combined_df = combined_df.apply(pd.to_numeric, errors='ignore')
    combined_df.to_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final.csv")
    print(combined_df)

    csv1 = pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\QL1.csv")
    qf1 = pd.DataFrame(csv1)
    csv2 = pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\QL2.csv")
    qf2 = pd.DataFrame(csv2)

    combined_qf = qf1 + qf2.values
    combined_qf = combined_qf.apply(pd.to_numeric, errors='ignore')
    combined_qf.to_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final.csv")
    print(combined_qf)
    plist = []
    qlist = []

    for column in combined_df:
       plist.extend(combined_df[column].tolist())

    for column in combined_qf:
        qlist.extend(combined_qf[column].tolist())

    return plist,qlist


def concat_df():




    df1 = pd.read_csv('D:\Thesis\Sensitivity analysis\P1 Sampled sensitivity analysis_constant from PF.csv')
    df2 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q1 Sampled sensitivity analysis_constant from PF.csv')
    df3 = pd.read_csv('D:\Thesis\Sensitivity analysis\P2 Sampled sensitivity analysis_constant from PF.csv')
    df4 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q2 Sampled sensitivity analysis_constant from PF.csv')
    df5 = pd.read_csv('D:\Thesis\Sensitivity analysis\P3 Sampled sensitivity analysis_constant from PF.csv')
    df6 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q3 Sampled sensitivity analysis_constant from PF.csv')
    df7 = pd.read_csv('D:\Thesis\Sensitivity analysis\P4 Sampled sensitivity analysis_constant from PF.csv')
    df8 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q4 Sampled sensitivity analysis_constant from PF.csv')
    df9 = pd.read_csv('D:\Thesis\Sensitivity analysis\P5 Sampled sensitivity analysis_constant from PF.csv')
    df10 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q5 Sampled sensitivity analysis_constant from PF.csv')
    df11 = pd.read_csv('D:\Thesis\Sensitivity analysis\P6 Sampled sensitivity analysis_constant from PF.csv')
    df12 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q6 Sampled sensitivity analysis_constant from PF.csv')
    df13 = pd.read_csv('D:\Thesis\Sensitivity analysis\P7 Sampled sensitivity analysis_constant from PF.csv')
    df14 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q7 Sampled sensitivity analysis_constant from PF.csv')
    df15 = pd.read_csv('D:\Thesis\Sensitivity analysis\P8 Sampled sensitivity analysis_constant from PF.csv')
    df16 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q8 Sampled sensitivity analysis_constant from PF.csv')
    df17 = pd.read_csv('D:\Thesis\Sensitivity analysis\P9 Sampled sensitivity analysis_constant from PF.csv')
    df18 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q9 Sampled sensitivity analysis_constant from PF.csv')
    df19 = pd.read_csv('D:\Thesis\Sensitivity analysis\P10 Sampled sensitivity analysis_constant from PF.csv')
    df20 = pd.read_csv('D:\Thesis\Sensitivity analysis\Q10 Sampled sensitivity analysis_constant from PF.csv')

    bigdata = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20], ignore_index=True)
    bigdata.to_csv('D:\Thesis\Sensitivity analysis final.csv')



def calculate_max_min_absolute_values():
    df = pd.read_csv('D:\Thesis\Sensitivity analysis final.csv')
    pmax, pmin = 0, 100
    qmax, qmin = 0,100
    vmax, vmin = 0, 100
    vamax, vamin = 0, 100
    normalized_df = pd.DataFrame()
    for i in range(1,10):
        pmaxtemp, pmintemp = 0, 0
        pmaxtemp = df['{}{}'.format("p", i)].max()
        pmintemp = df['{}{}'.format("p", i)].min()
        if (pmaxtemp > pmax):
            pmax = pmaxtemp
        if (pmintemp < pmin):
            pmin = pmintemp

    for i in range(1, 10):
        normalized_df['{}{}'.format("p", i)] = (df['{}{}'.format("p", i)] - pmin) / (pmax - pmin)

    for i in range(1,10):
        maxtemp, mintemp = 0, 0
        maxtemp = df['{}{}'.format("q", i)].max()
        mintemp = df['{}{}'.format("q", i)].min()
        if (maxtemp > qmax):
            qmax = maxtemp
        if (mintemp < qmin):
            qmin = mintemp

    for i in range(1, 10):
        normalized_df['{}{}'.format("q", i)] = (df['{}{}'.format("q", i)] - qmin) / (qmax - qmin)


    for i in range(1,10):
        maxtemp, mintemp = 0, 0
        maxtemp = df['{}{}'.format("Voltage", i)].max()
        mintemp = df['{}{}'.format("Voltage", i)].min()
        if (maxtemp > vmax):
            vmax = maxtemp
        if (mintemp < vmin):
            vmin = mintemp

    for i in range(1, 10):
        normalized_df['{}{}'.format("Voltage", i)] = (df['{}{}'.format("Voltage", i)] - vmin) / (vmax - vmin)



    for i in range(1,10):
        maxtemp, mintemp = 0, 0
        maxtemp = df['{}{}'.format("Voltage angle", i)].max()
        mintemp = df['{}{}'.format("Voltage angle", i)].min()
        if (maxtemp > vamax):
            vamax = maxtemp
        if (mintemp < vamin):
            vamin = mintemp

    for i in range(1, 10):
        normalized_df['{}{}'.format("Voltage angle", i)] = (df['{}{}'.format("Voltage angle", i)] - vamin) / (vamax - vamin)


    print(normalized_df)
    return normalized_df




def get_top_abs_correlations(df):
    column_name = list(df)
    df_list =[]
    au_corr = df.corr().abs().unstack().reset_index()
    au_corr.columns = ['var1', 'var2', 'value']
    temp_df = pd.DataFrame()
    for names in column_name:
        temp_df = au_corr.loc[au_corr['var1']== names]
        temp_df = temp_df.nlargest(5,'value')
        df_list.append(temp_df)
    final_df = pd.concat(df_list)
    #labels_to_drop = get_redundant_pairs(df)
    #au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    print(final_df)

    return final_df

calculate_max_min_absolute_values()

