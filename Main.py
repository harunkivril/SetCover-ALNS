from LNS import *
from Data import *
import time

buyuk = r'scpnrg2.txt'
kucuk = r'scp410.txt'
data = DataObject(buyuk)
LNS(data,300, 20, 0.8, 10, [0,0.8, 1.2, 2])
#print(data.s_best)
#print(len(data.s_best))
print(data.cost_function(data.s_best))
print(len(data.s_best),' ', len(np.unique(data.s_best)))
# data.random_initial_solution()
# print(np.random.random(data.s_current.shape[0]).shape[0])
# #print(data.M, " ", data.N)
# data.destroy(10, method = 'Mixed')
# data.repair(method = 'Greedy')
# #print(data.s_current)
# #print(len(data.temp_sol))
