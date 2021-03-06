from Data import *
import time
import matplotlib.pyplot as plt

def ALNS(data, t , n, lamda, k, w):
    data.w = w
    best_val = []
    data.random_initial_solution()
    start_time = time.time()

    data.iteration = 0
    while (time.time() - start_time) < t:

        data.destroy(n)
        data.repair(k)

        accept = data.acceptance(0.1)
        best = data.isBest()
        if accept:
             data.set_current()
        if best:
            data.set_best()
        data.UpdateProbs(accept,best, lamda)
        best_val.append(data.cost_function(data.s_best))
        data.iteration += 1

    # data.LocalSearch()
    # if best:
    #      data.set_best()
    plt.plot(best_val)
    plt.show()
    print(data.iteration)
