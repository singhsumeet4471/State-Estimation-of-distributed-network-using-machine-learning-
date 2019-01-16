import sys

import pandas as pd

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP3\Python\3.6")
import powerfactory as pf
import matplotlib.pyplot as plt


app = pf.GetApplication()
app.Show()
user = app.GetCurrentUser()
project = app.ActivateProject("StateEstimationThesis")
prj = app.GetActiveProject()

studycase= app.GetProjectFolder('study')

app.PrintPlain(studycase)

AllStudyCasesInProject= studycase.GetContents()
elmRes = app.GetFromStudyCase('Results.ElmRes')
for StudyCase in AllStudyCasesInProject:
   app.PrintPlain(StudyCase)
   StudyCase.Activate()




app.PrintPlain(prj)

ldf = app.GetFromStudyCase("ComLdf")
LineObj = app.GetCalcRelevantObjects("*.ElmLne")

loads = app.GetCalcRelevantObjects("*.ElmLod")
terms = app.GetCalcRelevantObjects("*.ElmTerm")
syms = app.GetCalcRelevantObjects("*.ElmSym")
print(loads)

for dislod in loads:
    print(dislod.cDisplayName)
    print(dislod.cDisplayName)

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
    csv3 = pd.read_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\QL3.csv")
    qf3 = pd.DataFrame(csv3)

    combined_qf = qf1 + qf2.values
    combined_dfq = combined_pf + qf3.values
    combined_df = combined_dfq.apply(pd.to_numeric, errors='ignore')
    combined_df.to_csv("D:\Thesis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final.csv")
    print(combined_df)
    plist = []
    qlist = []

    for column in combined_df:
       plist.extend(combined_df[column].tolist())

    for column in combined_df:
        qlist.extend(combined_qf[column].tolist())

    return plist,qlist

def sample_relatimedata():
    # plist = []
    # qlist = []
    # combined_df = pandas.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\P_final.csv")
    # combined_qf = pandas.read_csv("D:\Thessis\Sumeet\Sumeet\CSV_74_Loadprofiles_1min_W_var\Q_final.csv")
    #
    # for column in combined_df:
    #     plist.extend(combined_df[column].tolist())
    #
    # for column in combined_qf:
    #     qlist.extend(combined_qf[column].tolist())

    plist,qlist = add_csv()

    plt.scatter(qlist, plist)
    plt.show()
    dfvolt = [[] for x in range(11)]
    dfvolt_angle = [[] for x in range(11)]
    dfpower_factor = [[] for x in range(11)]
    pvalue = [[] for x in range(11)]
    qvalue = [[] for x in range(11)]
    p1 = []
    q1 = []
    i = 0
    ploadList = []
    qloadlist = []
    w = 0


    for p, q in zip(plist, qlist):

     w+=1
     if(i<10):
        i+=1
        ploadList.append(p)
        qloadlist.append(q)




     if(w==1500 and i==10):

        w = 0

        for ploop,qlopp,load in zip(ploadList,qloadlist,loads):
            load.plini = ploop
            load.qlini = ploop


        i =0
        ploadList =[]
        qloadlist = []
        ldf.Execute()

        pval = [Lod.GetAttribute('m:P:bus1') for Lod in loads]

        p1.append(0.0)
        for pvar, ptemp in zip(pval, pvalue):
            ptemp.append(pvar)

        q1.append(0.0)
        qval = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
        for qvar, qtemp in zip(qval, qvalue):
            qtemp.append(qvar)

        Voltages = [Volt.GetAttribute('m:U') for Volt in terms]
        for vvar, vlist in zip(Voltages, dfvolt):
            vlist.append(vvar)

        volt_angle = [Volt.GetAttribute('m:phiu') for Volt in terms]
        for vavar, valist in zip(volt_angle, dfvolt_angle):
            valist.append(vavar)

        power_factor = [Volt.GetAttribute('m:cosphiout') for Volt in terms]
        for pfvar, pflist in zip(power_factor, dfpower_factor):
            pflist.append(pfvar)

    pcheck = [p1] + pvalue
    qcheck = [q1] + qvalue
    dflist = []
    i = 0

    for pfinal, qfinal, vfinal, vafinal, pffinal in zip(pcheck, qcheck, dfvolt, dfvolt_angle, dfpower_factor):
        df = pd.DataFrame({'{}{}'.format("p", i): pd.Series(pfinal), '{}{}'.format("q", i): pd.Series(qfinal),
                 '{}{}'.format("Voltage", i): pd.Series(vfinal),
                 '{}{}'.format("Voltage angle", i): pd.Series(vafinal),
                 '{}{}'.format("Powerfactor", i): pd.Series(pffinal)})
        i += 1
        dflist.append(df)

    final_df = pd.concat(dflist, axis=1)
    final_df.to_csv('D:\Thesis\Sampled RealTime Data from PF.csv')