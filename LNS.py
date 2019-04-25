from Data import *
import time

def LNS(data, t , n, m1, m2):

    data.random_initial_solution()
    start_time = time.time()
    iter = 0
    while (time.time() - start_time) < t:

        data.destroy( n,m1)
        data.repair(m2)

        data.set_current()
        data.set_best()
        iter += 1
    print(iter)
