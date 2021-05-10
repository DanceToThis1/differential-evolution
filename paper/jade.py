import datetime
import numpy as np
import random
from scipy.stats import cauchy
import matplotlib.pyplot as plt


def jade(fobj, bounds, popsize=20, its=1000, goal=0, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mut = 0.5
    mean_cr = 0.5
    mean_mut = 0.5
    a = []  # 定义一个新种群A初始化为空
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
            p = 0.05 * popsize  # p的设置，固定值0.05-0.2 * NP，或者自适应调整。
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            idx_x_r2 = random.randint(0, len(population) + len(a) - 3)
            if idx_x_r2 >= (len(population) - 2):
                x_r2 = a[idx_x_r2 - len(population) + 2]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mean_mut, scale=0.1)
                else:
                    mut = 1
            mutant = np.clip(population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2), min_b, max_b)
            cr = random.gauss(mean_cr, 0.1)
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
                a.append(population[j])
                s_cr.append(cr)
                s_mut.append(mut)
            else:
                population_new[j] = population[j]
        while len(a) > popsize:
            index = random.randint(0, len(a) - 1)
            a.pop(index)
            pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        if np.fabs(fitness_best - goal) < 1e-6:
            print(i)
            break
        yield best, fitness_best


def jade_test(fun, bounds, its=3000, goal=0, log=0):
    start = datetime.datetime.now()
    it = list(jade(fun, bounds, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='jade')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with jade')
    plt.show()
    pass


def rastrigin_jade_test_20(fun, bounds):
    result = []
    for num in range(20):
        it = list(jade(fun, bounds, popsize=100, its=3000, goal=0))
        result.append(it[-1][-1])
        pass
    mean_result = np.mean(result)
    std_result = np.std(result)
    success_num = 0
    for i in result:
        if np.fabs(i - 0) < 1e-8:
            success_num += 1
            pass
        pass
    return mean_result, std_result, success_num
