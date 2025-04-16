
"""
星座优化模块测试脚本
"""

import numpy as np
from constellation_optimization import ConstellationOptimizer
from numerical_methods import OrbitPropagator  # 假设存在轨道传播器
from constellation_design import ConstellationDesigner  # 假设存在星座设计器

# 初始化轨道传播器和星座设计器
propagator = OrbitPropagator()
designer = ConstellationDesigner()

# 创建星座优化器
optimizer = ConstellationOptimizer(propagator, designer)

# 测试网格搜索
print("测试网格搜索算法...")
grid_result = optimizer.optimize_walker_constellation(
    algorithm='grid',
    total_satellites_range=(20, 50),
    num_planes_range=(3, 6),
    inclination_range=(45, 90),
    altitude_range=(500, 1500),
    max_iterations=10
)
print("网格搜索结果:", grid_result)

# 测试粒子群算法
print("\n测试粒子群算法...")
pso_result = optimizer.optimize_walker_constellation(
    algorithm='pso',
    total_satellites_range=(20, 50),
    num_planes_range=(3, 6),
    inclination_range=(45, 90),
    altitude_range=(500, 1500),
    max_iterations=20,
    num_particles=15
)
print("粒子群优化结果:", pso_result)

# 测试遗传算法
print("\n测试遗传算法...")
ga_result = optimizer.optimize_walker_constellation(
    algorithm='genetic',
    total_satellites_range=(20, 50),
    num_planes_range=(3, 6),
    inclination_range=(45, 90),
    altitude_range=(500, 1500),
    max_iterations=20,
    population_size=20
)
print("遗传算法结果:", ga_result)

# 比较不同算法
print("\n比较不同优化算法...")
comparison = optimizer.compare_optimization_algorithms(
    total_satellites_range=(20, 50),
    num_planes_range=(3, 6),
    inclination_range=(45, 90),
    altitude_range=(500, 1500),
    max_iterations=10
)
print("算法比较结果:", comparison)

# 可视化比较结果
fig = optimizer.visualize_optimization_comparison(comparison)
if fig:
    fig.savefig('algorithm_comparison.png')
    print("已保存算法比较图表")
