import numpy as np
import matplotlib.pyplot as plt
import datetime

"""
策略池 rand1 best2 current to rand1 随机选择
参数池 f : 0.1 - 0.9 step 0.1
     cr : 0.4 - 0.9 step 0.1  随机选择
存储成功的策略和参数 没实现
失败时要重新选择新的策略和参数 没实现
"""


def epsde(fobj, bounds, popsize=20, its=1000, goal=0):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    fitness = np.asarray([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx]
    for i in range(its):
        for j in range(popsize):
            rand_algo = np.random.randint(1, 4, 1)
            rand_f = 0.1 * np.random.randint(1, 9, 1)
            rand_cr = 0.1 * np.random.randint(4, 9, 1)
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c, d = population[np.random.choice(idxs, 4, replace=False)]
            mutant = np.clip(a + rand_f * (b - c), min_b, max_b)
            cross_points = np.random.rand(dimensions) < rand_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
                pass
            trial = np.where(cross_points, mutant, population[j])
            if rand_algo == 2:
                mutant = np.clip(best + rand_f * (a - b) + rand_f * (c - d), min_b, max_b)
                trial = np.where(cross_points, mutant, population[j])
                pass
            elif rand_algo == 3:
                k = np.random.rand()
                trial = np.clip(population[j] + k * (a - population[j]) + rand_f * (b - c), min_b, max_b)
                pass
            else:
                pass
            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population_new[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
                    pass
                pass
            else:
                population_new[j] = population[j]
                pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
        if np.fabs(min(fitness) - goal) < 1e-8:
            print(i)
            break
        yield best, fitness[best_idx]
    pass


it = list(epsde(lambda x: sum(x ** 2), [(-100, 100)] * 30, popsize=100, its=1000))
print(it[-1])
