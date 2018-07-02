from __future__ import division

import math
import random
from itertools import chain

import matplotlib.pyplot as plt
import pandapower as p
import pandapower.networks as pp
import pandas as pd

net = pp.create_cigre_network_lv()

buses = chain([0], range(23, 43))

net1 = p.select_subnet(net, buses, True, False, False)

p.runpp(net1)

print(net1.load)
load_list = []

load_list.append(net1.res_load)
# load_list.remove(2)
# load_list = load_list[1:]

print(load_list)

for line in load_list:
    p_kwar = line["p_kw"]
    q_kwar = line["q_kvar"]

p_kwar = p_kwar.drop([7])
q_kwar = q_kwar.drop([7])

print(p_kwar, q_kwar)

min_pvar = min(p_kwar)*1000
max_pvar = max(p_kwar)*1000

max_qvar = max(q_kwar)*1000
min_qvar = - max_qvar

def montecarlo(min, max,limit):
    randmaise = []
    randmaise = random.sample(range(min, max),limit)
    return randmaise


def powerfactor(p, q):
    s = math.sqrt((p **2) + (q **2))
    power_factor =  p/s
    return power_factor


pvar_list = []

pvar_list = montecarlo(min_pvar, max_pvar,100)

print(pvar_list)

qvar_list = []


qvar_list = montecarlo(min_qvar, max_qvar,100)


print(qvar_list)

final_result =[]
validvalues = []
plist , qlist = [], []
for p in pvar_list:
    #print(p)
    for q in qvar_list:
        value = powerfactor(p, q)

        if (value > 0.75):
            plist.append(p)
            qlist.append(q)
            #print("Correct value is :", value)
            validvalues = (p, q, value)
            final_result.append(validvalues)



final_result = map(list, final_result)

print(pd.DataFrame(list(final_result),columns = ("p_w","q_var","power_factor")))


plt.scatter(qlist,plist)
plt.show()
