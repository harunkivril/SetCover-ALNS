import numpy as np
import pandas as pd


class DataObject:

    def __init__(self, file):
        self._file = file
        self.read_data()
        self.s_best = np.array([], dtype=int)
        self.s_temp = np.array([], dtype=int)
        self.s_current = np.array([], dtype=int)
        self.v_best = np.zeros(self.N, dtype=int)
        self.v_current = np.zeros(self.N, dtype=int)
        self.v_temp = np.zeros(self.N, dtype=int)

        self.stuck = False

    def read_data(self):
        with open(self._file) as f:
            self.M, self.N = [int(x) for x in next(f).split()]
            self.data = np.zeros((self.M,self.N), dtype=bool)
            self.weights = []
            while(len(self.weights) < self.N):
                [self.weights.append(int(x)) for x in next(f).split()]
            for _set in range(self.M):
                n = int(next(f))
                a = 0
                while(a < n):
                    for num in next(f).split():
                        self.data[_set][int(num)-1] = True
                        a+=1
        self.data = pd.DataFrame(self.data)
        self.weights = np.array(self.weights)

    def random_initial_solution(self):
        b = self.M + 1
        while(not np.all(self.v_current)):
            a = np.random.randint(0, self.M) ## FIXME: aynı sayıdan üretmememiz lazım
            if not a == b:
                self.s_current = np.append(a, self.s_current)
                self.s_current = np.unique(self.s_current)
                self.v_current += self.data.iloc[a]
            b = a
        self.v_best = self.v_current
        self.s_best = self.s_current

    def repair(self, method):
        if method == 'Greedy':
            self.GreedyRepair()
        elif method == 'Other':
            self.OtherRepair()
        else:
            print('Please select an available method. Options: Greedy, Other')

    def GreedyRepair(self):
        while(not np.all(self.v_temp)):
            score = []
            missing = (self.v_temp == 0)
            for n_set, sub_set in enumerate(self.data.values):
                n_missing = np.sum(sub_set[missing])
                if n_missing == 0:
                    n_missing = 1.0e-8
                score.append(self.weights[n_set]/n_missing)
            a = np.argmin(score)
            self.s_temp = np.append(self.s_temp, a)
            self.v_temp += self.data.iloc[a]

    def destroy(self, n, method):
        score = self.setScoreMethod(method)
        self.s_temp = self.s_current
        if self.stuck:
            for i in range(n):
                self.s_temp = np.delete(self.s_temp, np.random.randint(0, len(self.s_temp)))
        else:
            print(np.argpartition(score, -n)[-n:])
            self.s_temp = np.delete(self.s_temp, np.argpartition(score, -n)[-n:])
        self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)

    def setScoreMethod(self, method):
        if method == 'Freq':
            return self.FreqScore()
        elif method == 'Weight':
            return self.WeightScore()
        else:
            return self.MixedScore()

    def FreqScore(self):
        score = []
        for sub_set in self.s_current:
            score.append(np.dot(self.data.iloc[sub_set], self.v_current))
        return score

    def WeightScore(self):
        score = []
        for sub_set in self.s_current:
            reg = np.dot(self.data.iloc[sub_set], (self.v_current == 1))
            score.append(self.weights[sub_set] - (np.max(self.weights)/np.mean(self.weights))**reg)
        return score

    def MixedScore(self):
        freq = self.FreqScore()
        weight = self.WeightScore()
        return (freq-np.mean(freq))/np.std(freq)+(weight-np.mean(weight))/np.std(weight)

    def cost_function(self,set):
        cost = 0
        for i in set:
            cost += self.weights[i]
        return cost


    # def isBest(self):
    #     return self.cost_function(self.s_temp) < self.cost_function(self.s_best)
    #
    # def acceptance(self):
    #     return self.cost_function(self.s_temp) < self.cost_function(self.s_current)

    def set_best(self):
        if (self.cost_function(self.s_temp) < self.cost_function(self.s_best)):
            self.s_best = self.s_temp
            self.v_best = self.v_temp

    def set_current(self):
        if (self.cost_function(self.s_temp) < self.cost_function(self.s_current)):
            self.s_current = self.s_temp
            self.v_current = self.v_temp
            self.stuck = False
        else:
            self.stuck = True
