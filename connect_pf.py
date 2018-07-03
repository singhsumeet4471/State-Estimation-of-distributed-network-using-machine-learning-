import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP2\Python\3.6")
import powerfactory as pf

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




ldf.Execute()

for load in loads:
   app.PrintPlain(load.loc_name)



# Voltages=[Volt.GetAttribute('m:U') for Volt in terms ]
# volt_angle=[Volt.GetAttribute('m:phiu') for Volt in terms ]
# power_factor=[Volt.GetAttribute('m:cosphiout') for Volt in terms ]
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
