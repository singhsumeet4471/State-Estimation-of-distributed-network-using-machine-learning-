import math
# sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2018 SP2\Python\3.6")
# import powerfactory as pf
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd

from randmise import merge

# max_pvar = max(p)*1000000
#
# max_qvar = max(q)*1000000
# min_qvar = - max_qvar
sys.setrecursionlimit(1500)

def montecarlo(min, max,limit):
    randmaise = []
    randmaise = random.sample(range(min, max),limit)
    return randmaise


def powerfactor(p, q):
    s = math.sqrt((p **2) + (q **2))
    power_factor =  p/s
    return power_factor


def sample_montecarlo():
    pvar_list = []
    pvar_list = montecarlo(0, 30000, 1400)
    qvar_list = []
    qvar_list = montecarlo(-10000, 10000, 1400)

    final_result = merge(pvar_list, qvar_list)
    print(final_result)
    plist, qlist = zip(*final_result)

    i = len(plist)
    final_result = []
    validvalues = []
    pfinallist, qfinallist = [], []
    for x in range(i):
        value = powerfactor(plist[x], qlist[x])
        if (value > 0.75):
            pfinallist.append(plist[x])
            qfinallist.append(qlist[x])
            # print("Correct value is :", value)
            validvalues = (plist[x], qlist[x], value)
            final_result.append(validvalues)

    final_result = map(list, final_result)

    df = pd.DataFrame(list(final_result), columns=("p_w", "q_var", "power_factor"))
    df.to_csv('D:\Thessis\Sampled Data.csv')
    plt.scatter(qlist, plist)
    plt.show()


x = sample_montecarlo()
