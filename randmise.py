import numpy  as np
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

    csv1 =  pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL1.csv")
    pf1 = pd.DataFrame(csv1)
    csv2 = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL2.csv")
    pf2 = pd.DataFrame(csv2)
    csv3 = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\PL3.csv")
    pf3 = pd.DataFrame(csv3)

    combined_pf = pf1 + pf2.values
    combined_df = combined_pf + pf3.values
    combined_df = combined_df.apply(pd.to_numeric, errors='ignore')
    combined_df.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final.csv")
    print(combined_df)

    csv1 = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\QL1.csv")
    qf1 = pd.DataFrame(csv1)
    csv2 = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\QL2.csv")
    qf2 = pd.DataFrame(csv2)

    combined_qf = qf1 + qf2.values
    combined_qf = combined_qf.apply(pd.to_numeric, errors='ignore')
    combined_qf.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final.csv")
    print(combined_qf)
    plist = []
    qlist = []

    for column in combined_df:
       plist.extend(combined_df[column].tolist())

    for column in combined_qf:
        qlist.extend(combined_qf[column].tolist())

    return plist,qlist



def split():
    csv1 = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final.csv")


    df = np.array_split(csv1,4)
    df1 = df[0]
    df2 = df[1]
    df3 = df[2]
    df4 = df[3]

    df1.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final_FRaction1.csv")
    df2.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final_FRaction2.csv")
    df3.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final_FRaction3.csv")
    df4.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final_FRaction4.csv")

    csv = pd.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final.csv")

    dfQ = np.array_split(csv,4)
    df1Q = dfQ[0]
    df2Q = dfQ[1]
    df3Q = dfQ[2]
    df4Q = dfQ[3]
    df1Q.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final_FRaction1.csv")
    df2Q.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final_FRaction2.csv")
    df3Q.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final_FRaction3.csv")
    df4Q.to_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final_FRaction4.csv")




split()





