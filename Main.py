from LNS import *
from Data import *
import time

data = DataObject(r'scpnrg2.txt')
data.random_initial_solution()
print(len(data.current_sets))
data.destroy(100)
print(len(data.temp_sol))
