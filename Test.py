from LNS import *
from Data import *
import pandas as pd

set_list= [r'scp49.txt',r'scpnrg1.txt']
t = [30,60]
lamb = 0.5
n= 20
k = 10
w = [0,0.2,0.4,1.5]
allowance = 0.1

result = np.empty((2,10))
iter = np.empty((2,10))

for j ,set in enumerate(set_list):
    data = DataObject(set)
    for i in range(10):
        ALNS(data, t[j], n, lamb, k , w)
        result[j][i] = data.cost_function(data.s_best)
        iter[j][i] = data.iteration

np.savetxt('test_result.txt', result)
np.savetxt('test_iter.txt', iter)
