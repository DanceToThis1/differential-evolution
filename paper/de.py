import numpy as np
from functions import fun_sphere

"""
实现标准差分进化算法。
dimensions为问题的维度
pop 为二维数组，popsize行，dimensions列。范围在0-1
min_b,max_b为各维度的上界和下界数组
population为映射到全部取值范围的二维种群数组
fitness为每个向量的适应值数组
best_idx为当前最佳向量在种群中的位置
best为当前最佳向量

进入迭代后：
idxs为除去目标向量序列号的列表。
a,b,c为选择的三个随机向量
mutant为变异向量
cross_points为随机生成的bool向量，长度与问题维度相同。
trial为试验向量

算法用到的函数：
len(x)：求解x的长度
np.random.rand(x,y)：生成大小为x*y的多维数组。
np.fabs：求解绝对值
np.argmin（x）：求解数组x中最小值所在位置。
np.random.choice（x，3）：在x中选择3个值
np.clip(x,0,1)：将x限定在0，1之间
yield:类似return。
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
"""
可能与文献不相符的地方：
交叉操作是否是保证至少一个维度进行变换？
选择操作如果新的试验向量较优则直接替代原来的目标向量，造成种群在本次迭代即发生变化，是否应等到下次迭代再进行替换操作？
"""
