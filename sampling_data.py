import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP2\Python\3.6")
import powerfactory as pf


app = pf.GetApplication()

#Variables definition

sLoads = {}


#check input parameters



oLdf = app.GetFromStudyCase('ComLdf')

oLdf.Execute()

#get all the loads
sLoads = app.GetCalcRelevantObjects('ElmLod');


#clear the results file
MyResults =app.MyResults()
MyResults.Clear()

#add the variables to the results file
MyResults.AddVars( 'b:dScale') #result variable of this script defined in the Advanced Options page

for x in sLoads :
    MyResults.AddVars(x, 'c:scale0', 'm:P:bus1', 'm:Q:bus1')




print(MyResults)

app.oExp.Execute();
