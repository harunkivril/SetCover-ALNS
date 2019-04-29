from LNS import *
from Data import *
import pandas as pd

set_list= [r'scp410.txt',r'scpnrg2.txt']

dongu = 3
ts=[30,60]
ns=[5, 10, 20]
lambdas = [0.1, 0.5, 0.9]
w1 = 0
w2 = 0.2
w3 = [0.4, 0.6]
w4 = [1, 1.5]
ks = [3, 10]
allowances = 0.1


results = np.empty((3,3,2,2,2,len(set_list),dongu))

for a, set in enumerate(set_list):
    data = DataObject(set)
    t = ts[a]
    for b, n in enumerate(ns):
        for c, lamb in enumerate(lambdas):
            for d, w_3 in enumerate(w3):
                for e, w_4 in enumerate(w4):
                    for f, k in enumerate(ks):
                        for i in range(dongu):
                            LNS(data, t, n, lamb, k , [w1,w2,w_3,w_4])
                            results[b][c][d][e][f][a][i] = data.cost_function(data.s_best)
                            print('Params:', b,c,d,e,f,a,i, 'Result: ',results[b][c][d][e][f][a][i])
np.savetxt('results.txt',results.reshape(len(set_list)*3*3*2*2*2*dongu))

results = np.loadtxt('results.txt').reshape((3,3,2,2,2,len(set_list),dongu))
best = []
relatives = np.empty((3,3,2,2,2,len(set_list),dongu))
averages = np.empty((3,3,2,2,2,len(set_list)))
stds = np.empty((3,3,2,2,2,len(set_list)))
scores = np.empty((3,3,2,2,2,len(set_list)))
for b, n in enumerate(ns):
    for c, lamb in enumerate(lambdas):
        for d, w_3 in enumerate(w3):
            for e, w_4 in enumerate(w4):
                for f, k in enumerate(ks):
                    for a, set in enumerate(set_list):
                        best.append(np.max(results[a].ravel()))
                        for i in range(dongu):
                            relatives[b][c][d][e][f][a][i] = 100*np.abs(results[b][c][d][e][f][a][i] - best[a])/best[a]
                        averages[b][c][d][e][f][a] = np.mean(relatives[b][c][d][e][f][a])
                        stds[b][c][d][e][f][a] = np.std(relatives[b][c][d][e][f][a])
                    scores[b][c][d][e][f] = np.mean(averages[b][c][d][e][f])/np.mean(stds[b][c][d][e][f])

np.savetxt('scores.txt',results.reshape(3*3*2*2*2))
