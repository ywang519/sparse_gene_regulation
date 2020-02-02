'''
Created on Feb 1, 2020

@author: mrwan
'''
import pickle
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

## 10 trials, each trial has 21 time step



file = open('./data/dataDream.pickle', 'rb')
GE = pickle.load(file)
file.close

file = open('./data/REQDream.pickle', 'rb')
bounds = pickle.load(file)
file.close


GEF = GE[0]


df = pd.DataFrame(GEF)


print(df.head())
print(df.columns)
maxlag=5
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df
 
a = grangers_causation_matrix(df, variables = df.columns)  


print(a['0_x'])