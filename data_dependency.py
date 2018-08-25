import pandas as pd
from pomegranate import *

df = pd.read_csv("D:\Thesis\Sampled Realtime Data from PF.csv")
x = df.as_matrix()

model = BayesianNetwork.from_samples(x,algorithm='chow-liu')








