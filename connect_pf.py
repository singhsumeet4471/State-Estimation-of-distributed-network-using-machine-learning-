import sys

import pandas

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP2\Python\3.6")
import powerfactory as pf
from sample_values import sample_montecarlo
from random import choice


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






df = sample_montecarlo()

plist = df.p_w.tolist()
qlist = df.q_var.tolist()

dfplist=[]
dfqlist=[]
dfvolt =[]
dfvolt_angle=[]
dfpower_factor =[]
for i in range(10):
   for load in loads:
      p= choice(plist)
      q= choice(qlist)
      load.plini = p
      load.qlini = q
      #app.PrintPlain(load.loc_name)

   ldf.Execute()
   Voltages = [Volt.GetAttribute('m:U') for Volt in terms]
   volt_angle = [Volt.GetAttribute('m:phiu') for Volt in terms]
   power_factor = [Volt.GetAttribute('m:cosphiout') for Volt in terms]
   print(volt_angle, volt_angle, power_factor)
   ptemp = [Lod.GetAttribute('m:P:bus1') for Lod in loads]
   qtemp = [Lod.GetAttribute('m:Q:bus1') for Lod in loads]
   dfplist.extend(ptemp)
   dfqlist.extend(qtemp)
   dfvolt.extend(Voltages)
   dfvolt_angle.extend(volt_angle)
   dfpower_factor.extend(power_factor)


df = pandas.DataFrame({"P_w":dfplist,"q_var":dfqlist,"Voltages":dfvolt,"volt_angle":dfvolt_angle,"power_factor":dfpower_factor})
df.to_csv('D:\Thessis\Sampled Data from PF.csv')


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
