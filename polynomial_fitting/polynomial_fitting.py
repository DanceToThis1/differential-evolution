# -*- coding: utf-8 -*-
"""
使用差分进化算法优化
五次多项式拟合cos函数 范围[0,10]

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy


def fun_fifth_polynomial(x, w):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * x ** 4 + w[5] * x ** 5
    pass


def loss(w):
    _x = np.linspace(0, 10, 500)
    y_predict = fun_fifth_polynomial(_x, w)
    return np.sqrt(sum((np.cos(_x) - y_predict) ** 2) / len(_x))
    pass


def jade_2(fobj, bounds, popsize=100, its=1000, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mean_cr = 0.5
    mean_mut = 0.5
    for i in range(its):
        s_mut = []
        s_cr = []  # 存储成功的cr值，每代清空
        population = list(population)
        population.sort(key=fobj)
        population = np.array(population)
        best = population[0]
        fitness_best = fobj(best)
        fitness = np.asarray([fobj(ind) for ind in population])
        for j in range(popsize):
            p = 0.05 * popsize
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mean_mut, scale=0.1)
                else:
                    mut = 1
            mutant = population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2)
            for mutant_i in range(len(mutant)):
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            cr = np.clip(random.gauss(mean_cr, 0.1), 0, 1)
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
                s_cr.append(cr)
                s_mut.append(mut)
            else:
                population_new[j] = population[j]
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        yield best, fitness_best
        # yield population, fitness
        pass
    pass


def test_1():
    it = list(jade_2(loss, [(-10, 10)] * 6, popsize=20))
    print(it[-1])
    xx = np.linspace(0, 10, 500)
    plt.plot(xx, np.cos(xx), label='cos(x)')
    plt.plot(xx, fun_fifth_polynomial(xx, it[-1][0]), label='result')
    plt.title('polynomial_fitting')
    plt.legend()
    plt.savefig('polynomial_fitting')
    plt.show()


test_1()
