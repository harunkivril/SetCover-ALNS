from LNS import *
from Data import *
import time

buyuk = r'scpnrg2.txt'
kucuk = r'scp410.txt'
data = DataObject(buyuk)
LNS(data,60, 10, 'Mixed', 'Greedy' )
#print(data.s_best)
#print(len(data.s_best))
print(data.cost_function(data.s_best))
# data.random_initial_solution()
# #print((data.current_vector))
# #print(data.M, " ", data.N)
# data.destroy(10, method = 'Mixed')
# data.repair(method = 'Greedy')
# #print(data.s_current)
# #print(len(data.temp_sol))
