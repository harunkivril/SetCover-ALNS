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
        self.p_repair = [0.5, 0.5]
        self.p_destroy = [0.25]*4
        self.w = np.array([0,0.5,0.8,2])

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
        #self.LocalSearch()

    def repair(self, n):
        self.r_method = np.random.choice(range(2), p=self.p_repair)
        if self.r_method == 0:
            self.GreedyRepair()
        else:
            self.OtherRepair(n)

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
            while a in self.s_temp:
                score = np.delete(score, a)
                a = np.argmin(score)
            self.s_temp = np.append(self.s_temp, a)
            self.v_temp += self.data.iloc[a]

    def OtherRepair(self, k):
        iter = 0
        while(not np.all(self.v_temp)):
            score = []
            missing0 = (self.v_temp == 0)
            missing1 = (self.v_temp == 1)
            for n_set, sub_set in enumerate(self.data.values):
                n_missing = np.any(sub_set[missing0]) +np.any(sub_set[missing1])/k
                score.append(n_missing/self.weights[n_set])
                #print(score[n_set])
            a = np.argmax(score)
            while (a in self.s_temp):
                score[a] = 1.0e-8
                a = np.argmax(score)
            self.s_temp = np.append(self.s_temp, a)
            self.v_temp += self.data.iloc[a]
            iter +=1
        #print(iter)


    def destroy(self, n):
        self.d_method = np.random.choice(range(4), p=self.p_destroy)
        if self.d_method == 0:
            return self.Freq(n)
        elif self.d_method == 1:
            return self.Weight(n)
        elif self.d_method == 2:
            return self.Random(n)
        elif self.d_method == 3:
            return self.Mixed(n)

    def Freq(self, n):
        score = []
        for sub_set in self.s_current:
            score.append(np.dot(self.data.iloc[sub_set], self.v_current))
        self.s_temp = self.s_current
        self.s_temp = np.delete(self.s_temp, np.argpartition(score, -n)[-n:])
        self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)

    def Weight(self, n):
        score = []
        for sub_set in self.s_current:
            reg = np.dot(self.data.iloc[sub_set], (self.v_current == 1))
            score.append(self.weights[sub_set]/(reg+1))
        self.s_temp = self.s_current
        self.s_temp = np.delete(self.s_temp, np.argpartition(score, -n)[-n:])
        self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)

    def Mixed(self,n):
        freq = []
        weight = []
        for sub_set in self.s_current:
            weight.append(self.weights[sub_set])
        for sub_set in self.s_current:
            freq.append(np.dot(self.data.iloc[sub_set], self.v_current))
        score = (freq-np.mean(freq))/np.std(freq)+(weight-np.mean(weight))/np.std(weight)
        self.s_temp = self.s_current
        self.s_temp = np.delete(self.s_temp, np.argpartition(score, -n)[-n:])
        self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)

    def Random(self,n):
        score = np.random.random(self.s_current.shape[0])
        self.s_temp = self.s_current
        self.s_temp = np.delete(self.s_temp, np.argpartition(score, -n)[-n:])
        self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)

    def LocalSearch(self):
        self.s_temp = self.s_current
        self.v_temp = self.v_current
        while(np.all(self.v_temp)):
            score = []
            unique = (self.v_temp == 1)
            for n_set, sub_set in enumerate(self.s_temp):
                n_unique = np.sum(self.data.values[sub_set][unique])
                score.append(n_unique/self.weights[sub_set])
            a = np.argmin(score)
            self.s_temp = np.delete(self.s_temp,a)
            self.v_temp = np.sum(self.data.iloc[self.s_temp], axis=0)
        self.v_current = self.v_temp
        self.s_current = self.s_temp


    def cost_function(self,set):
        cost = 0
        for i in set:
            cost += self.weights[i]
        return cost

    def UpdateProbs(self, accept, best, lamda):
        if best:
            w = self.w[3]
        elif self.acceptance(0):
            w = self.w[2]
        elif accept:
            w = self.w[1]
        else:
            w = self.w[0]
        self.p_destroy[self.d_method] = self.p_destroy[self.d_method]*lamda + w*(1-lamda)
        self.p_destroy = self.p_destroy/np.sum(self.p_destroy)
        self.p_repair[self.r_method] = self.p_repair[self.r_method]*lamda + w*(1-lamda)
        self.p_repair = self.p_repair/np.sum(self.p_repair)
        print(self.d_method, ': ' ,self.p_destroy,'***', self.r_method, ': ' ,self.p_repair)

    def isBest(self):
        return self.cost_function(self.s_temp) < self.cost_function(self.s_best)

    def acceptance(self, allowance):
        return self.cost_function(self.s_temp) < self.cost_function(self.s_best)*(1+allowance)

    def set_best(self):
        self.s_best = self.s_temp
        self.v_best = self.v_temp

    def set_current(self):
        self.s_current = self.s_temp
        self.v_current = self.v_temp
