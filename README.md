# differential-evolution 差分进化算法
基于基本DE实现改进算法JDE、SaDE、JADE、SHADE、CoDE。在20个benchmark函数上进行测试。
>基本DE实现参考自：https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

程序结构：  
functions.py为20个基准测试函数的实现。  
additional_code_for_ppt目录下的文件是为了做PPT画图写的，与主程序无关。  
image目录下为程序运行保存的图片。  
jade_test目录中的jadeTest.py为JADE算法的仿真。首先实现文献中提到的20个测试函数，然后分别实现带额外存档和不带额外存档的JADE算法，之后在20个测试函数上进行测试，保存数据到csv文件中。  
paper目录下为标准差分进化算法和五种改进算法的实现，EPSDE算法尚未添加特性。  
test.py对几种算法进行测试，有可视化部分和生成csv数据部分。可视化部分将各种算法在每个测试函数上的表现画在一张图中，清晰的展示每种算法的迭代过程。数据生成部分将每个算法在每个测试函数上分别优化50次，将结果记录在csv文件中。  
程序基于python3.7编写，基本运行库需求：  
numpy               1.20.2  
scipy               1.6.2  
pandas              1.2.4  
matplotlib          3.4.2  
