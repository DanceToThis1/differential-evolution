import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
from scipy.stats import cauchy


def jade(fobj, bounds, popsize=20, its=1000, goal=0, c=0.5):
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
        matrix_sort(population, fobj, pop)  # 为了得到种群中前百分之几的个体需要对种群中的个体进行排序
        best = population[0]
        fitness_best = fobj(best)
        fitness = np.asarray([fobj(ind) for ind in population])
        for j in range(popsize):
            p = 0.2 * popsize  # p的设置，固定值0.2 * NP，或者自适应调整。
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            idx_x_r2 = random.randint(0, len(population) + len(a) - 3)
            if idx_x_r2 >= (len(population) - 2):
                x_r2 = a[idx_x_r2 - len(population) + 2]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)
            while mut < 0 or mut > 1:
                mut = cauchy.rvs(loc=mean_mut, scale=0.1)
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
        mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
        mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        if np.fabs(fitness_best - goal) < 1e-6:
            print(i)
            break
        yield best, fitness_best


def matrix_sort(matrix, fobj, pop):
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if fobj(matrix[i]) > fobj(matrix[j]):
                pop[i] = matrix[i]
                pop[j] = matrix[j]
                matrix[i] = pop[j]
                matrix[j] = pop[i]
    return matrix


def rastrigin(x1):
    return sum(x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10)


start = datetime.datetime.now()
it = list(jade(rastrigin, [(-5.12, 5.12)] * 30, popsize=100, its=3000))
print(it[-1])
end = datetime.datetime.now()
print(end - start)
x, f = zip(*it)
plt.plot(f, label='rastrigin with jade')
plt.yscale('log')
plt.legend()
plt.show()
