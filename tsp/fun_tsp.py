# -*- coding: utf-8 -*-
"""
tsp function:
旅行商问题
x 随机生成的向量
y x从小到大排序后的向量
x_index y每个维度依次在x中的索引位置
x_0 x中最小的值变成0，第二小的变成1，依次，最大的值变成29，由随机向量对应的可以看作一个解的向量。
hde:
将差分进化算法的选择操作改为遗传算法中的操作，不将试验向量与父代相比而是存入新种群，截取前半部分。
"""
import random

import numpy as np
import matplotlib.pyplot as plt


def x_x0(x):
    x = list(x)
    set_x = set(x)
    x_index = [0] * len(x)
    x_0 = [0] * len(x)
    if len(x) != len(set_x):
        x_copy = x
        for item in set_x:
            x.remove(item)
            pass
        for x_item in x_copy:
            x_item += 0.01
            set_x.add(x_item)
        x = list(set_x)
    else:
        pass
    y = list(np.sort(x))
    for i1 in range(len(x)):
        x_index[i1] = x.index(y[i1])
        pass
    point_1 = 0
    for i1 in range(len(x)):
        x_0[x_index[i1]] = point_1
        point_1 += 1
        pass
    pass
    return x_0
    pass


def fun_tsp_oliver_30(x_0):
    # x_0 = list(x_0)
    points = [
        [54, 67],
        [54, 62],
        [37, 84],
        [41, 94],
        [2, 99],
        [7, 64],
        [25, 62],
        [22, 60],
        [18, 54],
        [4, 50],
        [13, 40],
        [18, 40],
        [24, 42],
        [25, 38],
        [44, 35],
        [41, 26],
        [45, 21],
        [58, 35],
        [62, 32],
        [82, 7],
        [91, 38],
        [83, 46],
        [71, 44],
        [64, 60],
        [68, 58],
        [83, 69],
        [87, 76],
        [74, 78],
        [71, 71],
        [58, 69]
    ]
    length = 0
    for i in range(len(x_0) - 1):
        x1 = points[int(x_0[i])][0]
        y1 = points[int(x_0[i])][1]
        x2 = points[int(x_0[i + 1])][0]
        y2 = points[int(x_0[i + 1])][1]
        length += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        pass
    length += np.sqrt((points[int(x_0[0] - 1)][0] - points[int(x_0[-1]) - 1][0]) ** 2 + (points[int(x_0[0]) - 1][1] - points[int(x_0[-1]) - 1][1]) ** 2)
    return length
    pass


def hde(fobj, bounds, mut=0.9, cr=0.1, popsize=200, its=200):
    dimensions = len(bounds)
    population = np.zeros(popsize * dimensions).reshape(popsize, dimensions)
    for pop_i in range(popsize):
        population[pop_i] = np.array(random.sample(range(30), dimensions))
        pass
    new_population = np.zeros(2 * popsize * dimensions).reshape(2 * popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for mutant_i in range(len(mutant)):
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            cross_points = np.random.rand(dimensions) < cr
            cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            trial_0 = x_x0(trial)
            new_population[j] = np.array(trial_0)
            new_population[popsize + j] = population[j]
            pass
        new_population = list(new_population)
        new_population.sort(key=fobj)
        new_population = np.array(new_population)
        for k in range(len(population)):
            population[k] = new_population[k]
            pass
        best = population[0]
        fitness_best = fobj(best)
        yield best, fitness_best
    pass


it = list(hde(fun_tsp_oliver_30, [(-15, 45)] * 30, 0.5, 0.5, popsize=200, its=200))
print(it[-1])
order = list(it[-1][0])
points = [
        [54, 67],
        [54, 62],
        [37, 84],
        [41, 94],
        [2, 99],
        [7, 64],
        [25, 62],
        [22, 60],
        [18, 54],
        [4, 50],
        [13, 40],
        [18, 40],
        [24, 42],
        [25, 38],
        [44, 35],
        [41, 26],
        [45, 21],
        [58, 35],
        [62, 32],
        [82, 7],
        [91, 38],
        [83, 46],
        [71, 44],
        [64, 60],
        [68, 58],
        [83, 69],
        [87, 76],
        [74, 78],
        [71, 71],
        [58, 69]
    ]
points = np.array(points)
xx = points[:, 0]
yy = points[:, 1]
plt.scatter(xx, yy)
plt.show()
