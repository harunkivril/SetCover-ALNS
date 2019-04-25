from Data import *
import time

def LNS(data, t , *args):

    data.random_initial_solution()
    start_time = time.time()

    while (time.time() - start_time) < t:

        data.destroy(*args)
        data.repair(*args)

        if data.acceptance():
            data.place_temp()
        if data.isBest():
            data.set_best()

        
