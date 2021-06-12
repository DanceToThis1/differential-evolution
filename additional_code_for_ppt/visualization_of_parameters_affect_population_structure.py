# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
import os
path1 = os.path.abspath('.')
path2 = os.path.abspath('..')


def de_magic_change(fobj, bounds, popsize=100, its=1000):
    dimensions = len(bounds)
    population = np.zeros(popsize * dimensions).reshape(popsize, dimensions)
    for pop_i in range(len(population)):
        for pop_j in range(dimensions):
            population[pop_i][pop_j] = random.gauss(-2, 0.5)
    min_b, max_b = np.asarray(bounds).T
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    for i in range(its):
        population = list(population)
        population.sort(key=fobj)
        population = np.array(population)
        fitness = np.asarray([fobj(ind) for ind in population])
        for j in range(popsize):
            p = 0.05 * popsize
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            mut = 0.9
            mutant = population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2)
            for mutant_i in range(len(mutant)):
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            # 手动调节参数1
            cr = 0.1
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
            else:
                population_new[j] = population[j]
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        yield population, fitness
        pass
    pass


def fun_rastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10)


# 记录在算法迭代过程中个体的位置变化情况,如果使用matplotlib.animation应该能做成动画。
def test_2():
    it = list(de_magic_change(fun_rastrigin, [(-5, 5)] * 2, popsize=500, its=10))
    # print(it[-1])
    xx = np.arange(-5, 5, 0.1)
    yy = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(xx, yy)
    z = 2 * 10 + xx ** 2 - 10 * np.cos(2 * np.pi * xx) + yy ** 2 - 10 * np.cos(2 * np.pi * yy)
    x = it[1][0][:, 0]
    y = it[1][0][:, 1]
    plt.contour(xx, yy, z, levels=10, alpha=0.3)
    plt.scatter(x, y)
    # 手动调节参数2
    plt.title('how cr influence population, cr=0.1')
    # plt.savefig(path2 + '/image/image_in_ppt/how cr influence population' + str(random.randint(1, 1000)))
    plt.show()
    pass


test_2()
