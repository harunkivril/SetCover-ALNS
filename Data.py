import numpy as np
import pandas as pd

class DataObject:

    def __init__(self,file):
        self._file = file
        self.read_data()
        self.best_sets = np.array([], dtype=int)
        self.temp_sol = np.array([], dtype=int)
        self.current_sets = np.array([], dtype=int)
        self.best_vector = np.zeros(self.N, dtype= int)
        self.current_vector = np.zeros(self.N, dtype=int)
        self.temp_vector = np.zeros(self.N, dtype= int)


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
        while(not np.all(self.current_vector) ):
            a = np.random.randint(0,self.M) ## FIXME: aynı sayıdan üretmememiz lazım
            self.current_sets = np.append(a, self.current_sets)
            self.current_sets = np.unique(self.current_sets)
            self.current_vector = np.sum(self.data.iloc[self.current_sets], axis = 0)
        self.best_vector = self.current_vector
        self.best_sets = self.current_sets

    def search_neighbor(self, k):
        n = len(self.current_sets)
        temp_vector = np.zeros(self.N)
        temp_sets = list(self.current_sets)
        [temp_sets.pop(np.random.randint(0,n-i)) for i in range(k)]
        for i in temp_sets:
            temp_vector = np.logical_or(temp_vector,self.data[i])
        cont = True
        while(cont):
            temp_sets2 = temp_sets.copy()
            temp_vector2 = temp_vector.copy()
            while(sum(temp_vector2) < self.N):
                ran = np.random.randint(0,self.M)
                temp_sets2.append(ran)
                temp_vector2 = np.logical_or(temp_vector2, self.data[ran])
            temp_sets2 = np.unique(temp_sets2)
            cont = (temp_sets2.shape[0] >= n)

        return temp_sets2, temp_vector2

    def destroy(self, n):
        score = []
        for sub_set in self.current_sets:
            score.append(np.dot(self.data.iloc[sub_set] , self.current_vector))

        self.temp_sol = self.current_sets
        print(np.argpartition(score , -n)[-n:])
        self.temp_sol = np.delete(self.temp_sol ,np.argpartition(score , -n)[-n:])
        self.temp_vector = np.sum(self.data.iloc[self.temp_sol], axis = 0)
        missing = np.array(range(self.N))[self.temp_vector == 0]) # FIXME: Repaire taşınacak

    def cost_function(self,set):
        cost = 0
        for i in set:
            cost += self.weights[i]
        return cost

    def isBest(self):
        return cost_function(self.temp_sol) < cost_function(self.best_sets)

    def acceptance(self):
        return cost_function(self.temp_sol) < cost_function(self.current_sets)

    def set_best(self):
        if (cost_function(self.temp_sol) < cost_function(self.best_sets)):
            self.best_sets = self.temp_sol

    def set_current(self):
        if (cost_function(self.temp_sol) < cost_function(self.current_sets)):
            self.current_sets = self.temp_sol
