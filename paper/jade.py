import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import cauchy
import pandas as pd
import os

path1 = os.path.abspath('.')
path2 = os.path.abspath('..')

"""
jade函数,实现JADE算法
输入为待优化函数和相应的维度，每个维度的取值范围。
输出为每次迭代中最优向量和最优向量对应的最优解。
此函数在all_algo_test.py中被调用。
"""


def jade(fobj, bounds, popsize=100, its=1000, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.zeros(popsize * dimensions).reshape(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
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
            p = 0.05 * popsize
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            idx_x_r2 = random.randint(0, len(population) + len(a) - 3)
            if idx_x_r2 >= (len(population) - 2):
                x_r2 = a[idx_x_r2 - len(population) + 2]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)  # 柯西分布生成随机数
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mean_mut, scale=0.1)
                else:
                    mut = 1
            mutant = population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2)
            for mutant_i in range(len(mutant)):  # 变异向量越界处理
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            cr = np.clip(random.gauss(mean_cr, 0.1), 0, 1)
            cross_points = np.random.rand(dimensions) < cr
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
        for k in range(len(population)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        # yield best, mean_mut, mean_cr, fitness_best
        yield best, fitness_best
        pass
    pass


"""
jade_test函数
参数随迭代次数的变化情况
使用时还需要改算法的返回值，注释掉现在的，使用被注释的那条。
这个函数画ppt中的图用的，在此程序中没有被调用。
"""


def jade_test(fun, bounds, popsize=100, its=1000):
    it = list(jade(fun, bounds, popsize=popsize, its=its))
    print(it[-1])
    x, mut, cr, f = zip(*it)
    plt.plot(mut, label='F')
    plt.plot(cr, label='CR')
    plt.title('JADE ' + fun.__name__)
    plt.legend()
    plt.show()
    pass


"""
jade_test_50函数
输入为评价函数和与之对应的变量范围和最大迭代次数
对函数进行50次优化，把每次得到的优化结果记录进csv文件中。
这个函数在all_algo_test.py中被调用。
"""


def jade_test_50(fun, bounds, its):
    result = []
    for num in range(50):
        it = list(jade(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['JADE', fun.__name__, its, i] for i in result])
    data.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['JADE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', index=False, header=False)
    pass
