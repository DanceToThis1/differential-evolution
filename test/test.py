# -*- coding: utf-8 -*-
import random

import numpy as np
import matplotlib.pyplot as plt


def jade_2(fobj, bounds, popsize=100, its=1000):
    dimensions = len(bounds)
    # pop = np.random.rand(popsize, dimensions)
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


# JADE算法 记录在算法过程中个体的位置变化情况,matplotlib.animation应该能做成动画。
def test_2():
    it = list(jade_2(fun_rastrigin, [(-5, 5)] * 2, popsize=500, its=10))
    # print(it[-1])
    xx = np.arange(-5, 5, 0.1)
    yy = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(xx, yy)
    # z = (xx ** 2 + yy - 11) ** 2 + (xx + yy ** 2 - 7) ** 2
    z = 2 * 10 + xx ** 2 - 10 * np.cos(2 * np.pi * xx) + yy ** 2 - 10 * np.cos(2 * np.pi * yy)
    # z = xx ** 2 + yy ** 2
    x = it[1][0][:, 0]
    y = it[1][0][:, 1]
    plt.contour(xx, yy, z, levels=10, alpha=0.3)
    plt.scatter(x, y)
    plt.title('how cr influence population, cr=0.1')
    plt.savefig('C:\\Users\\zhang\\PycharmProjects\\differentialEvolution\\image\\image_in_ppt\\' + 'how cr influence population' + str(random.randint(1, 1000)))
    plt.show()
    pass


test_2()
#
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')
#
#
# def init():
#     ax.set_xlim(0, 2 * np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
#
# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
#                     init_func=init, blit=True)
# plt.show()
