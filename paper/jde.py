import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

path1 = os.path.abspath('.')
path2 = os.path.abspath('..')

"""
在基本算法基础上改进参数选择策略。
"""


def jde(fobj, bounds, mut=0.9, cr=0.1, popsize=100, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), min_b, max_b)
            cross_points = np.random.rand(dimensions) < cr
            randj = np.random.rand(4)
            if randj[0] < 0.1:
                mut = 0.1 + randj[1] * 0.9
            if randj[2] < 0.1:
                cr = randj[3]
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
        yield best, fitness[best_idx]


def jde_test(fun, bounds, mut=0.9, cr=0.1, its=3000, log=0):
    it = list(jde(fun, bounds, mut=mut, cr=cr, popsize=100, its=its))
    print(it[-1])
    x, f = zip(*it)
    plt.plot(f, label='jde')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass


def jde_test_50(fun, bounds, its):
    result = []
    for num in range(50):
        it = list(jde(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['JDE', fun.__name__, its, i] for i in result])
    data.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['JDE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', index=False, header=False)
    pass
