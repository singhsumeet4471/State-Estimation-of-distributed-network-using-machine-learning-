import sys

import pandas

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP3\Python\3.6")
import powerfactory as pf
from sample_values import sample_montecarlo
from random import choice
from randmise import add_csv
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






def sample_moduledata():
    df = sample_montecarlo()

    plist = df.p_w.tolist()
    qlist = df.q_var.tolist()

    dfvolt = [[] for x in range(11)]
    dfvolt_angle = [[] for x in range(11)]
    dfpower_factor = [[] for x in range(11)]
    pvalue = [[] for x in range(11)]
    qvalue = [[] for x in range(11)]
    #print(pvalue)
    skip = False
    p1 = []
    q1 = []


    for i in range(len(plist)):
        #pvalue.append(0.0)
        #qvalue.append(0.0)
        for load in loads:
            p = choice(plist)
            q = choice(qlist)
            load.plini = p
            load.qlini = q
            #pvalue.append(p)
            #qvalue.append(q)
            # app.PrintPlain(load.loc_name)

        ldf.Execute()

        pval = [Lod.GetAttribute('m:P:bus1') for Lod in loads]
        skip = False
        p1.append(0.0)
        for pvar, plist in zip(pval, pvalue):
            plist.append(pvar)





        q1.append(0.0)
        q = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
        for qvar, qlist in zip(q, qvalue):
            qlist.append(qvar)


        Voltages = [Volt.GetAttribute('m:U') for Volt in terms]
        for vvar, vlist in zip(Voltages, dfvolt):
            vlist.append(vvar)

        volt_angle = [Volt.GetAttribute('m:phiu') for Volt in terms]
        for vavar, valist in zip(volt_angle, dfvolt_angle):
            valist.append(vavar)

        power_factor = [Volt.GetAttribute('m:cosphiout') for Volt in terms]
        for pfvar, pflist in zip(power_factor, dfpower_factor):
            pflist.append(pfvar)








    pcheck = [p1]+pvalue
    qcheck = [q1]+qvalue
    dflist = []
    i=0

    for pfinal, qfinal ,vfinal,vafinal,pffinal in zip(pcheck,qcheck,dfvolt,dfvolt_angle,dfpower_factor):

        df = pandas.DataFrame( {'{}{}'.format("p", i): pandas.Series(pfinal), '{}{}'.format("q", i):pandas.Series(qfinal), '{}{}'.format("Volatge", i):pandas.Series(vfinal),
                                '{}{}'.format("Volatage angle", i):pandas.Series(vafinal), '{}{}'.format("Powerfactor", i):pandas.Series(pffinal)})
        i +=1
        dflist.append(df)

    final_df = pandas.concat(dflist,axis=1)
    final_df.to_csv('D:\Thessis\Sampled monte carlo Data from PF.csv')


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
        for pvar, plist in zip(pval, pvalue):
            plist.append(pvar)

        q1.append(0.0)
        q = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
        for qvar, qlist in zip(q, qvalue):
            qlist.append(qvar)

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
        df = pandas.DataFrame({'{}{}'.format("p", i): pandas.Series(pfinal), '{}{}'.format("q", i): pandas.Series(qfinal),
                 '{}{}'.format("Volatge", i): pandas.Series(vfinal),
                 '{}{}'.format("Volatage angle", i): pandas.Series(vafinal),
                 '{}{}'.format("Powerfactor", i): pandas.Series(pffinal)})
        i += 1
        dflist.append(df)

    final_df = pandas.concat(dflist, axis=1)
    final_df.to_csv('D:\Thessis\Sampled RealTime Data from PF.csv')


sample_relatimedata()


# Loads = []
#
# print(Voltages)
# print(volt_angle)
# print(power_factor)
#
# p = [Lod.Set('m:P:bus1') for Lod in loads]
# q = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
#
#
#
# print(p)
# print(q)
# print(power_factor)
