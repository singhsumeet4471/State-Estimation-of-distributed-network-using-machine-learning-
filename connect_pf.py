import sys

import pandas

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP3\Python\3.6")
import powerfactory as pf
from sample_values import sample_montecarlo
import random
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

    plist = df['p_w'].tolist()
    qlist = df['q_var'].tolist()

    dfvolt = [[] for x in range(11)]
    dfvolt_angle = [[] for x in range(11)]
    dfpower_factor = [[] for x in range(11)]
    pvalue = [[] for x in range(11)]
    qvalue = [[] for x in range(11)]
    #print(pvalue)
    skip = False
    p1 = []
    q1 = []


    for i in range(3000):

        for load in loads:
            p = random.choice(plist)
            q = random.choice(qlist)
            load.plini = p
            load.qlini = q

            # app.PrintPlain(load.loc_name)

        ldf.Execute()

        pval = [Lod.GetAttribute('m:P:bus1') for Lod in loads]

        p1.append(0.0)
        for pvar, pnew in zip(pval, pvalue):
            pnew.append(pvar)

        q1.append(0.0)
        q = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
        for qvar, qnew in zip(q, qvalue):
            qnew.append(qvar)


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
    i=0

    for pfinal, qfinal ,vfinal,vafinal,pffinal in zip(pcheck,qcheck,dfvolt,dfvolt_angle,dfpower_factor):

        df = pandas.DataFrame( {'{}{}'.format("p", i): pandas.Series(pfinal), '{}{}'.format("q", i):pandas.Series(qfinal), '{}{}'.format("Volatge", i):pandas.Series(vfinal),
                                '{}{}'.format("Volatage angle", i):pandas.Series(vafinal), '{}{}'.format("Powerfactor", i):pandas.Series(pffinal)})
        i +=1
        dflist.append(df)

    final_df = pandas.concat(dflist,axis=1)
    final_df.to_csv('D:\Thesis\Sampled monte carlo Data from PF.csv')


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
    final_df.to_csv('D:\Thesis\Sampled RealTime Data from PF.csv')


def sample_sensitive_analysis():
    plist, qlist = add_csv()

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

        w += 1
        if (i < 10):
            i += 1
            if(i==5):
                ploadList.append(1716)
                qloadlist.append(206)
            else:
                ploadList.append(p)
                qloadlist.append(q)

        if (w == 1500 and i == 10):

            w = 0

            for ploop, qlopp, load in zip(ploadList, qloadlist, loads):
                load.plini = ploop
                load.qlini = ploop

            i = 0
            ploadList = []
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
        df = pandas.DataFrame(
            {'{}{}'.format("p", i): pandas.Series(pfinal), '{}{}'.format("q", i): pandas.Series(qfinal),
             '{}{}'.format("Volatge", i): pandas.Series(vfinal),
             '{}{}'.format("Volatage angle", i): pandas.Series(vafinal),
             '{}{}'.format("Powerfactor", i): pandas.Series(pffinal)})
        i += 1
        dflist.append(df)

    final_df = pandas.concat(dflist, axis=1)
    final_df.to_csv('D:\Thesis\Sampled sensitivity analysis from PF.csv')


sample_sensitive_analysis()


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
