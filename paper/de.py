import numpy as np
from functions import fun_sphere

"""
实现标准差分进化算法，包括三种策略，此项目中后两种策略没有涉及。
"""


def de(fobj, bounds, mut=0.9, cr=0.1, popsize=20, its=1000):
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
    pass


def de_randtobest_1(fobj, bounds, mut=0.5, cr=0.3, popsize=100, its=1000):
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
            a, b = population[np.random.choice(idxs, 2, replace=False)]
            mutant = np.clip(population[j] + mut * (best - population[j]) + mut * (a - b), min_b, max_b)
            cross_points = np.random.rand(dimensions) < cr
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


def de_randtobest_2(fobj, bounds, mut=0.5, cr=0.3, popsize=20, its=1000):
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
            a, b, c, d = population[np.random.choice(idxs, 4, replace=False)]
            mutant = np.clip(population[j] + mut * (best - population[j]) + mut * (a - b) + mut * (c - d), min_b, max_b)
            cross_points = np.random.rand(dimensions) < cr
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
    pass


"""
@function de_rand_1_test:标准差分进化算法测试
@:parameter fun:评价函数，在benchmarkFunctions目录下的functions.py中有定义。或者直接使用'lambda x: sum(x ** 2)'。
@:parameter bounds:评价函数每个维度的限定搜索范围，如[(-100, 100)] * 30
@:parameter mut: 即缩放因子参数F，由于F容易冲突所以改为mut。
@:parameter cr: 交叉概率
@:parameter popsize: 即种群大小NP
@:parameter its: 迭代次数，对于已知全局最优解的函数也可以设置为差异足够小时停止。
@:return 由于采用yield，所以需要再加一层list。返回一个列表，列表中包含两项，第一项为每次迭代中种群的最优向量，第二项为每次迭代最优向量对应的适应值。
"""


def de_rand_1_test(fun=None, bounds=None, mut=0.9, cr=0.1, popsize=100, its=1000):
    if fun is None:
        fun = fun_sphere
    if bounds is None:
        bounds = [(-100, 100)] * 30
    it = list(de(fun, bounds, mut, cr, popsize=popsize, its=its))
    print(it[-1])
    pass

# de_rand_1_test()
