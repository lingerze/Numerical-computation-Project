"""
优化算法模块，提供多种优化算法类
包括网格搜索、粒子群优化和遗传算法
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class GridSearch:
    """网格搜索优化算法类"""
    
    def __init__(self, objective_func, bounds, steps=None):
        """
        初始化网格搜索
        
        参数:
            objective_func: 目标函数
            bounds: 参数边界列表，每个元素为(min, max)
            steps: 每个维度的步长列表，默认为每个维度10个点
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.steps = steps if steps else [10] * len(bounds)
        self.best_position = None
        self.best_score = -float('inf')
        self.history = []
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, value, message=""):
        """更新进度"""
        if self.progress_callback:
            # 检查回调函数是否仍然有效（例如，UI窗口可能已关闭）
            try:
                 self.progress_callback(value, message)
            except Exception as e:
                 print(f"Error in progress callback: {e}") # 或者使用日志记录
                 pass # 忽略回调错误，继续优化

    def optimize(self):
        """
        执行网格搜索优化
        
        返回:
            dict: 优化结果字典, 包含 'best_position', 'best_score', 'history'
        """
        dimensions = len(self.bounds)
        grid_points = []
        
        # 为每个维度创建网格点
        for i in range(dimensions):
            min_val, max_val = self.bounds[i]
            # 确保 step 是有效的整数
            step_count = int(self.steps[i]) if self.steps[i] >= 2 else 2
            points = np.linspace(min_val, max_val, step_count) 
            grid_points.append(points)
        
        # 创建网格
        try:
            # 使用 sparse=False 生成密集网格，方便迭代
            # 注意：对于高维度或大步数，这可能消耗大量内存
            # 如果内存是大问题，应考虑迭代器或其他方法
            grid = np.meshgrid(*grid_points, indexing='ij', sparse=False)
            # 将网格点重塑为 (N, D) 数组，N是总组合数，D是维度
            positions_to_evaluate = np.vstack([g.ravel() for g in grid]).T
        except MemoryError:
            print("MemoryError: Grid search meshgrid is too large. Consider reducing dimensions or steps.")
            return {'best_position': None, 'best_score': -float('inf'), 'history': []}
        except Exception as e:
             print(f"Error creating meshgrid: {e}")
             return {'best_position': None, 'best_score': -float('inf'), 'history': []}

        total_combinations = len(positions_to_evaluate)
        if total_combinations == 0:
            print("Warning: No grid points to evaluate.")
            return {'best_position': None, 'best_score': -float('inf'), 'history': []}
        
        # 评估每个网格点
        count = 0
        for position in positions_to_evaluate:
            try:
                # 评估目标函数
                score = self.objective_func(position)
                
                # 记录历史
                self.history.append((position.tolist(), score)) # Store position as list
                
                # 更新最佳位置
                if score > self.best_score:
                    self.best_score = score
                    self.best_position = position.tolist() # Store best position as list
                
                # 更新进度
                count += 1
                progress = int(100 * count / total_combinations)
                self._update_progress(progress, f"网格搜索: 已评估 {count}/{total_combinations} 组合, 当前最佳: {self.best_score:.4f}")
            except InterruptedError: # 来自 _update_progress 的取消信号
                 print("GridSearch optimize interrupted.")
                 # 返回当前找到的最佳结果
                 return {
                     'best_position': self.best_position,
                     'best_score': self.best_score,
                     'history': self.history
                 }
            except Exception as e:
                 print(f"Error evaluating position {position}: {e}")
                 # 记录失败，继续下一个点
                 self.history.append((position.tolist(), -float('inf'))) 
                 count += 1 # 也要增加计数器
                 progress = int(100 * count / total_combinations)
                 self._update_progress(progress, f"网格搜索: 评估点 {count}/{total_combinations} 时出错")
                 continue
        
        return {
            'best_position': self.best_position,
            'best_score': self.best_score,
            'history': self.history
        }
    
    def visualize_optimization(self, figsize=(12, 8)):
        """
        可视化优化过程
        
        参数:
            figsize (tuple): 图表大小
        
        返回:
            matplotlib.figure.Figure: matplotlib图表对象
        """
        fig = plt.figure(figsize=figsize)
        
        # 绘制优化历史
        # 过滤掉评估失败的分数
        valid_history = [item for item in self.history if isinstance(item[1], (int, float)) and item[1] > -float('inf')]
        if not valid_history:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有有效的优化数据", ha='center', va='center')
            return fig
            
        scores = [item[1] for item in valid_history]
        
        ax = fig.add_subplot(111)
        ax.plot(scores, '-o', markersize=3, label='评估分数')
        # 绘制累积最佳分数
        if scores:
             best_scores_accum = np.maximum.accumulate(scores)
             ax.plot(best_scores_accum, 'g--', label='累积最佳分数')
             
        ax.set_xlabel('评估次数')
        ax.set_ylabel('目标函数得分')
        ax.set_title('网格搜索优化过程')
        
        # 标记最佳得分
        if self.best_score > -float('inf'):
            ax.axhline(y=self.best_score, color='r', linestyle='--', 
                      label=f'最终最佳得分: {self.best_score:.3f}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


class ParticleSwarmOptimization:
    """粒子群优化算法类"""
    
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=50, 
                w=0.7, c1=1.4, c2=1.4):
        """
        初始化粒子群优化
        
        参数:
            objective_func: 目标函数
            bounds: 参数边界列表，每个元素为(min, max)
            num_particles (int): 粒子数量
            max_iter (int): 最大迭代次数
            w (float): 惯性权重
            c1 (float): 个体学习因子
            c2 (float): 社会学习因子
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds) # Ensure bounds is numpy array
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        
        if self.bounds.shape[1] != 2:
            raise ValueError("Bounds should be a list of (min, max) pairs.")
        self.dimensions = len(self.bounds)
        self.min_bounds = self.bounds[:, 0]
        self.max_bounds = self.bounds[:, 1]
        
        self.best_position = None
        self.best_score = -float('inf')
        
        # 粒子群历史记录
        self.history = [] # List of dicts per iteration: {'positions': ..., 'scores': ...}
        self.best_scores_history = [] # List of best score found so far at each iteration
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, value, message=""):
        """更新进度"""
        if self.progress_callback:
            try:
                 self.progress_callback(value, message)
            except Exception as e:
                 print(f"Error in progress callback: {e}")
                 pass
    
    def optimize(self):
        """
        执行粒子群优化
        
        返回:
            dict: 优化结果字典, 包含 'best_position', 'best_score', 'history', 'best_scores' (迭代最佳)
        """
        # 初始化粒子位置和速度
        # Ensure positions are within bounds initially
        positions = np.random.rand(self.num_particles, self.dimensions) * (self.max_bounds - self.min_bounds) + self.min_bounds
        # Initialize velocities more carefully, perhaps scaled by bounds range
        vel_range = (self.max_bounds - self.min_bounds) * 0.1 # Example scaling
        velocities = np.random.uniform(-vel_range, vel_range, (self.num_particles, self.dimensions))
        
        personal_best_positions = positions.copy()
        personal_best_scores = np.ones(self.num_particles) * -float('inf')
        
        global_best_position = None
        global_best_score = -float('inf')
        
        self.best_scores_history = [] # Reset history for this run
        self.history = []
        
        # 评估初始种群
        initial_scores = np.array([self.objective_func(p) for p in positions])
        personal_best_scores = initial_scores.copy()
        best_initial_idx = np.argmax(initial_scores)
        global_best_score = initial_scores[best_initial_idx]
        global_best_position = positions[best_initial_idx].copy()
        
        # 迭代优化
        for iteration in range(self.max_iter):
            iter_scores = np.zeros(self.num_particles)
            # 评估每个粒子的目标函数并更新pbest/gbest
            for i in range(self.num_particles):
                try:
                     score = self.objective_func(positions[i])
                     iter_scores[i] = score
                     
                     # 更新个体最佳位置
                     if score > personal_best_scores[i]:
                         personal_best_scores[i] = score
                         personal_best_positions[i] = positions[i].copy()
                     
                     # 更新全局最佳位置
                     if score > global_best_score:
                         global_best_score = score
                         global_best_position = positions[i].copy()
                except Exception as e:
                     print(f"PSO: Error evaluating particle {i} at iter {iteration}: {e}")
                     iter_scores[i] = -float('inf') # Mark as invalid
                     # Optionally skip this particle for update or handle differently

            # 记录当前迭代的粒子状态和分数
            current_state = {
                'positions': positions.copy(),
                'scores': iter_scores 
            }
            self.history.append(current_state)
            self.best_scores_history.append(global_best_score) # Record best score *so far*
            
            # 更新速度和位置
            r1 = np.random.random((self.num_particles, self.dimensions))
            r2 = np.random.random((self.num_particles, self.dimensions))
            
            velocities = (self.w * velocities + 
                         self.c1 * r1 * (personal_best_positions - positions) + 
                         self.c2 * r2 * (global_best_position - positions))
            
            positions = positions + velocities
            
            # 边界检查 (使用 numpy clip)
            positions = np.clip(positions, self.min_bounds, self.max_bounds)
            
            # 更新进度
            progress = int(100 * (iteration + 1) / self.max_iter)
            try:
                 self._update_progress(progress, f"粒子群优化: 迭代 {iteration + 1}/{self.max_iter}, 最佳得分: {global_best_score:.4f}")
            except InterruptedError:
                 print("PSO optimize interrupted.")
                 break # 中断循环
        
        # 返回最终结果
        self.best_position = global_best_position.tolist() if global_best_position is not None else None
        self.best_score = global_best_score
        
        return {
            'best_position': self.best_position,
            'best_score': self.best_score,
            'history': self.history,
            'best_scores': self.best_scores_history # Return iteration best scores
        }
    
    def visualize_optimization(self, figsize=(12, 10)):
        """
        可视化优化过程
        
        参数:
            figsize (tuple): 图表大小
        
        返回:
            matplotlib.figure.Figure: matplotlib图表对象
        """
        fig = plt.figure(figsize=figsize)
        
        # 绘制优化历史 (使用 best_scores_history)
        if not self.best_scores_history:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有有效的优化数据", ha='center', va='center')
            return fig
        
        # 绘制收敛曲线
        ax1 = fig.add_subplot(211 if self.dimensions >= 2 else 111) # Only one plot if 1D
        ax1.plot(self.best_scores_history, '-o', markersize=3)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('全局最佳目标函数得分')
        ax1.set_title('粒子群优化收敛过程')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制粒子分布散点图 (最后一次迭代)
        if self.dimensions >= 2 and self.history:
            ax2 = fig.add_subplot(212)
            
            # 获取最后一次迭代的粒子状态
            last_state = self.history[-1]
            positions = last_state.get('positions')
            scores = last_state.get('scores')
            
            if positions is not None and scores is not None:
                 # 过滤掉评估失败的分数，避免颜色映射问题
                 valid_indices = scores > -float('inf')
                 valid_positions = positions[valid_indices]
                 valid_scores = scores[valid_indices]

                 if len(valid_positions) > 0:
                     # 绘制散点图 (取前两个维度)
                     scatter = ax2.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                                          c=valid_scores, cmap='viridis', 
                                          s=50, alpha=0.6, edgecolors='w')
                     # 添加颜色条
                     plt.colorbar(scatter, ax=ax2, label='目标函数得分')
                 else:
                      ax2.text(0.5, 0.5, '无有效粒子数据', ha='center', va='center')

                 # 标记最终全局最佳位置
                 if self.best_position is not None:
                     ax2.scatter(self.best_position[0], self.best_position[1], 
                                color='red', s=200, marker='*', 
                                label=f'最佳位置 ({self.best_position[0]:.2f}, {self.best_position[1]:.2f})')
                     ax2.legend()
            else:
                 ax2.text(0.5, 0.5, '无最终粒子状态数据', ha='center', va='center')
            
            # 设置标签和标题
            ax2.set_xlabel(f'参数 1 ({self.bounds[0,0]:.1f} to {self.bounds[0,1]:.1f})')
            ax2.set_ylabel(f'参数 2 ({self.bounds[1,0]:.1f} to {self.bounds[1,1]:.1f})')
            ax2.set_title('最终粒子分布 (前两维)')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


class GeneticAlgorithm:
    """遗传算法优化类"""
    
    def __init__(self, objective_func, bounds, population_size=50, max_generations=50,
                crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, elitism=True):
        """
        初始化遗传算法
        
        参数:
            objective_func: 目标函数
            bounds: 参数边界列表，每个元素为(min, max)
            population_size (int): 种群大小
            max_generations (int): 最大代数
            crossover_rate (float): 交叉率 (0 to 1)
            mutation_rate (float): 变异率 (0 to 1)
            tournament_size (int): 锦标赛选择大小
            elitism (bool): 是否启用精英保留
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = int(tournament_size)
        self.elitism = elitism
        
        if self.bounds.shape[1] != 2:
            raise ValueError("Bounds should be a list of (min, max) pairs.")
        self.dimensions = len(self.bounds)
        self.min_bounds = self.bounds[:, 0]
        self.max_bounds = self.bounds[:, 1]

        self.best_individual = None
        self.best_score = -float('inf')
        
        # 历史记录
        self.history = [] # List of dicts per generation: {'population': ..., 'scores': ...}
        self.best_scores_history = [] # List of best score found so far at each generation
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, value, message=""):
        """更新进度"""
        if self.progress_callback:
            try:
                 self.progress_callback(value, message)
            except Exception as e:
                 print(f"Error in progress callback: {e}")
                 pass
    
    def optimize(self):
        """
        执行遗传算法优化
        
        返回:
            dict: 优化结果字典, 包含 'best_position', 'best_score', 'history', 'best_scores' (每代最佳)
        """
        # 初始化种群
        population = np.random.rand(self.population_size, self.dimensions) * (self.max_bounds - self.min_bounds) + self.min_bounds
        scores = np.ones(self.population_size) * -float('inf')

        self.best_scores_history = []
        self.history = []
        current_best_individual = None
        current_best_score = -float('inf')
        
        # 迭代优化
        for generation in range(self.max_generations):
            # 评估适应度
            for i in range(self.population_size):
                try:
                    score = self.objective_func(population[i])
                    scores[i] = score
                    # 更新当前代最佳
                    if score > current_best_score:
                         current_best_score = score
                         current_best_individual = population[i].copy()
                except Exception as e:
                     print(f"GA: Error evaluating individual {i} at gen {generation}: {e}")
                     scores[i] = -float('inf') # Mark as invalid

            # 更新全局最佳
            if current_best_score > self.best_score:
                 self.best_score = current_best_score
                 self.best_individual = current_best_individual.copy()
            
            # 记录历史
            current_state = {
                'population': population.copy(),
                'scores': scores.copy()
            }
            self.history.append(current_state)
            self.best_scores_history.append(self.best_score) # Record best score *so far*
            
            # 如果已经达到最大代数，结束优化
            if generation == self.max_generations - 1:
                break
            
            # --- 生成下一代 --- 
            new_population = np.zeros_like(population)
            start_index = 0
            
            # 精英保留
            if self.elitism:
                 # 找到当前代的最佳个体索引
                 elite_idx = np.argmax(scores)
                 new_population[0] = population[elite_idx].copy()
                 start_index = 1
            
            # 填充新种群
            for i in range(start_index, self.population_size):
                # 选择父代 (锦标赛选择)
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)
                
                # 交叉
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                    # 选择其中一个子代 (或者两个都用，如果种群大小允许)
                    new_population[i] = child1 # 简化处理，只用一个
                else:
                    new_population[i] = parent1.copy() # 未交叉则直接继承父代
                    
                # 变异
                new_population[i] = self._mutate(new_population[i])

                # 边界检查
                new_population[i] = np.clip(new_population[i], self.min_bounds, self.max_bounds)

            # 更新种群
            population = new_population
            # 重置分数，下一代重新评估
            scores.fill(-float('inf')) 
            current_best_score = -float('inf') # 重置当前代最佳
            
            # 更新进度
            progress = int(100 * (generation + 1) / self.max_generations)
            try:
                 self._update_progress(progress, f"遗传算法: 代数 {generation + 1}/{self.max_generations}, 最佳得分: {self.best_score:.4f}")
            except InterruptedError:
                 print("GA optimize interrupted.")
                 break # 中断循环
        
        # 返回最终结果 (best_individual 是 NumPy 数组)
        best_pos_list = self.best_individual.tolist() if self.best_individual is not None else None
        return {
            'best_position': best_pos_list, # 优化算法通常返回 position
            'best_score': self.best_score,
            'history': self.history,
            'best_scores': self.best_scores_history # 每代最佳
        }

    def _tournament_selection(self, population, scores):
        """锦标赛选择"""
        # 随机选择 tournament_size 个个体
        selection_ix = np.random.randint(len(population), size=self.tournament_size)
        # 从选择的个体中找到适应度最高的
        best_ix_in_tournament = selection_ix[np.argmax(scores[selection_ix])]
        return population[best_ix_in_tournament]

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if self.dimensions < 2:
             return parent1.copy(), parent2.copy()
        # 随机选择交叉点 (不包括第一个和最后一个基因位)
        pt = np.random.randint(1, self.dimensions)
        # 执行交叉
        child1 = np.hstack((parent1[:pt], parent2[pt:]))
        child2 = np.hstack((parent2[:pt], parent1[pt:]))
        return child1, child2

    def _mutate(self, individual):
        """均匀变异"""
        for i in range(self.dimensions):
            if np.random.random() < self.mutation_rate:
                # 在边界内随机生成一个新值
                individual[i] = np.random.uniform(self.min_bounds[i], self.max_bounds[i])
        return individual
    
    def visualize_optimization(self, figsize=(12, 10)):
        """
        可视化优化过程
        
        参数:
            figsize (tuple): 图表大小
        
        返回:
            matplotlib.figure.Figure: matplotlib图表对象
        """
        fig = plt.figure(figsize=figsize)
        
        # 绘制优化历史 (使用 best_scores_history)
        if not self.best_scores_history:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有有效的优化数据", ha='center', va='center')
            return fig
        
        # 绘制最佳适应度曲线
        ax1 = fig.add_subplot(211 if self.dimensions >= 2 else 111)
        ax1.plot(self.best_scores_history, '-o', markersize=3)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('全局最佳适应度')
        ax1.set_title('遗传算法优化收敛过程')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制最后一代种群分布
        if self.dimensions >= 2 and self.history:
            ax2 = fig.add_subplot(212)
            
            # 获取最后一代种群状态
            last_state = self.history[-1]
            population = last_state.get('population')
            scores = last_state.get('scores')
            
            if population is not None and scores is not None:
                 # 过滤无效分数
                 valid_indices = scores > -float('inf')
                 valid_population = population[valid_indices]
                 valid_scores = scores[valid_indices]
                 
                 if len(valid_population) > 0:
                     # 绘制散点图 (取前两个维度)
                     scatter = ax2.scatter(valid_population[:, 0], valid_population[:, 1], 
                                          c=valid_scores, cmap='viridis', 
                                          s=50, alpha=0.6, edgecolors='w')
                     # 添加颜色条
                     plt.colorbar(scatter, ax=ax2, label='适应度')
                 else:
                      ax2.text(0.5, 0.5, '无有效种群数据', ha='center', va='center')

                 # 标记最终全局最佳个体
                 if self.best_individual is not None:
                     ax2.scatter(self.best_individual[0], self.best_individual[1], 
                                color='red', s=200, marker='*', 
                                label=f'最佳个体 ({self.best_individual[0]:.2f}, {self.best_individual[1]:.2f})')
                     ax2.legend()
            else:
                  ax2.text(0.5, 0.5, '无最终种群状态数据', ha='center', va='center')

            # 设置标签和标题
            ax2.set_xlabel(f'参数 1 ({self.bounds[0,0]:.1f} to {self.bounds[0,1]:.1f})')
            ax2.set_ylabel(f'参数 2 ({self.bounds[1,0]:.1f} to {self.bounds[1,1]:.1f})')
            ax2.set_title('最终种群分布 (前两维)')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig 

class MultiObjectivePSO:
    """多目标粒子群优化算法(MOPSO)类"""
    
    def __init__(self, objective_funcs, bounds, num_particles=30, max_iter=50, 
                w=0.7, c1=1.4, c2=1.4, weight_method='weighted_sum', archive_size=100):
        """
        初始化多目标粒子群优化
        
        参数:
            objective_funcs: 目标函数列表，或具有多个目标的单一函数
            bounds: 参数边界列表，每个元素为(min, max)
            num_particles: 粒子数量
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 全局学习因子
            weight_method: 权重方法，'weighted_sum'或'pareto'
            archive_size: 当使用pareto方法时，非支配解档案的最大大小
        """
        # 如果只提供了一个函数，假设它返回多个目标值
        if callable(objective_funcs) and not isinstance(objective_funcs, list):
            self.single_func = True
            self.objective_func = objective_funcs
        else:
            self.single_func = False
            self.objective_funcs = objective_funcs if isinstance(objective_funcs, list) else [objective_funcs]
            self.num_objectives = len(self.objective_funcs)
        
        self.bounds = np.array(bounds)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.weight_method = weight_method
        self.archive_size = archive_size
        
        self.dimensions = len(self.bounds)
        self.min_bounds = self.bounds[:, 0]
        self.max_bounds = self.bounds[:, 1]
        
        # 初始化帕累托最优解档案
        self.archive = []  # 存储非支配解
        
        # 结果存储
        self.best_position = None
        self.best_score = None
        self.best_scores_history = []
        self.history = []
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, value, message=""):
        """更新进度"""
        if self.progress_callback:
            try:
                self.progress_callback(value, message)
            except Exception as e:
                print(f"Error in progress callback: {e}")
                pass
    
    def _evaluate(self, position, weights=None):
        """
        评估给定位置的目标函数
        
        参数:
            position: 位置向量
            weights: 各目标的权重列表（如果使用加权和方法）
        
        返回:
            score: 如果是加权和方法，返回加权得分；如果是帕累托方法，返回目标函数值列表
        """
        if self.single_func:
            # 假设函数返回多个目标值
            objectives = self.objective_func(position)
            if not isinstance(objectives, (list, tuple, np.ndarray)):
                objectives = [objectives]  # 处理单目标情况
            objectives = np.array(objectives)
        else:
            # 计算每个目标函数
            objectives = np.array([func(position) for func in self.objective_funcs])
        
        # 如果使用加权和方法，返回加权得分
        if self.weight_method == 'weighted_sum' and weights is not None:
            return np.sum(objectives * weights)
        
        # 否则返回目标值列表
        return objectives
    
    def _dominates(self, scores1, scores2):
        """
        判断解1是否支配解2（越大越好）
        
        参数:
            scores1: 解1的目标函数值列表
            scores2: 解2的目标函数值列表
        
        返回:
            bool: 如果解1支配解2，返回True
        """
        better_in_any = False
        for i in range(len(scores1)):
            if scores1[i] < scores2[i]:
                return False  # 解1在任一目标上更差
            elif scores1[i] > scores2[i]:
                better_in_any = True
        
        return better_in_any  # 解1至少在一个目标上更好且在其他目标上不差
    
    def _update_archive(self, position, scores):
        """
        更新非支配解档案
        
        参数:
            position: 位置向量
            scores: 目标函数值列表
        """
        # 检查当前解是否被档案中的解支配
        dominated = False
        i = 0
        while i < len(self.archive):
            archive_pos, archive_scores = self.archive[i]
            
            # 如果新解支配档案中的解，移除档案中的解
            if self._dominates(scores, archive_scores):
                self.archive.pop(i)
            # 如果档案中的解支配新解，标记为被支配
            elif self._dominates(archive_scores, scores):
                dominated = True
                break
            else:
                i += 1
        
        # 如果新解未被支配，添加到档案
        if not dominated:
            self.archive.append((position.copy(), scores.copy()))
            
            # 如果档案太大，移除拥挤区域的解
            if len(self.archive) > self.archive_size:
                self._reduce_archive()
    
    def _reduce_archive(self):
        """当档案超过最大大小时，移除拥挤区域的解"""
        if len(self.archive) <= self.archive_size:
            return
        
        # 计算每个解的拥挤度
        crowding_distances = []
        for i in range(len(self.archive)):
            pos_i, scores_i = self.archive[i]
            distance = float('inf')
            
            # 找最近的解
            for j in range(len(self.archive)):
                if i != j:
                    pos_j, scores_j = self.archive[j]
                    d = np.linalg.norm(scores_i - scores_j)
                    distance = min(distance, d)
            
            crowding_distances.append(distance)
        
        # 按拥挤度排序（越小越拥挤）
        sorted_indices = np.argsort(crowding_distances)
        
        # 移除最拥挤的解
        to_remove = sorted_indices[0]
        self.archive.pop(to_remove)
    
    def _select_leader(self):
        """从非支配解档案中选择领导者"""
        if not self.archive:
            return None, None
        
        # 随机选择一个非支配解作为领导者
        # 可以使用更复杂的策略，如轮盘赌或基于拥挤度的选择
        idx = np.random.randint(0, len(self.archive))
        return self.archive[idx][0], self.archive[idx][1]
    
    def _apply_weight_method(self, weights):
        """使用加权和方法选择最佳解"""
        if not self.archive:
            return None, -float('inf')
        
        best_position = None
        best_weighted_score = -float('inf')
        
        for position, scores in self.archive:
            weighted_score = np.sum(scores * weights)
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_position = position
        
        return best_position, best_weighted_score
    
    def optimize(self, weights=None):
        """
        执行多目标粒子群优化
        
        参数:
            weights: 当使用weighted_sum方法时，各目标的权重列表
        
        返回:
            dict: 优化结果字典, 包含 'best_position', 'best_score', 'pareto_front', 'history'
        """
        # 权重检查和初始化
        if self.weight_method == 'weighted_sum' and weights is None:
            if self.single_func:
                # 尝试从单函数中获取默认权重
                test_result = self.objective_func(self.min_bounds)
                if isinstance(test_result, (list, tuple, np.ndarray)):
                    num_objectives = len(test_result)
                    weights = np.ones(num_objectives) / num_objectives  # 平均权重
                else:
                    # 单目标情况
                    weights = np.array([1.0])
            else:
                # 多函数情况
                num_objectives = len(self.objective_funcs)
                weights = np.ones(num_objectives) / num_objectives  # 平均权重
        
        # 初始化粒子位置和速度
        positions = np.random.rand(self.num_particles, self.dimensions) * (self.max_bounds - self.min_bounds) + self.min_bounds
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dimensions)) * (self.max_bounds - self.min_bounds)
        
        # 个体最佳位置和分数
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(p, weights) for p in positions])
        
        # 初始化档案
        self.archive = []
        for i in range(self.num_particles):
            if self.weight_method == 'pareto':
                # 对于帕累托方法，评估所有目标
                scores = self._evaluate(positions[i])
                self._update_archive(positions[i], scores)
            else:
                # 对于加权和方法，简单地记录初始解
                self.archive.append((positions[i].copy(), np.array([personal_best_scores[i]])))
        
        # 迭代优化
        iteration_best_scores = []
        for iteration in range(self.max_iter):
            # 更新进度
            progress = int(100 * (iteration + 1) / self.max_iter)
            self._update_progress(progress, f"MOPSO 迭代 {iteration+1}/{self.max_iter}")
            
            # 对每个粒子
            for i in range(self.num_particles):
                # 选择领导者
                if self.weight_method == 'pareto':
                    global_best_position, _ = self._select_leader()
                    if global_best_position is None:
                        continue
                else:
                    # 加权和方法使用当前最佳解
                    global_best_position, _ = self._apply_weight_method(weights)
                    if global_best_position is None:
                        continue
                
                # 更新速度
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                              self.c1 * r1 * (personal_best_positions[i] - positions[i]) + 
                              self.c2 * r2 * (global_best_position - positions[i]))
                
                # 限制速度
                velocities[i] = np.clip(velocities[i], -0.2 * (self.max_bounds - self.min_bounds), 0.2 * (self.max_bounds - self.min_bounds))
                
                # 更新位置
                positions[i] = positions[i] + velocities[i]
                
                # 边界处理
                positions[i] = np.clip(positions[i], self.min_bounds, self.max_bounds)
                
                # 评估新位置
                if self.weight_method == 'pareto':
                    current_scores = self._evaluate(positions[i])
                    
                    # 更新个体最佳
                    if self._dominates(current_scores, personal_best_scores[i]):
                        personal_best_positions[i] = positions[i].copy()
                        personal_best_scores[i] = current_scores.copy()
                    
                    # 更新档案
                    self._update_archive(positions[i], current_scores)
                else:
                    # 加权和方法
                    current_score = self._evaluate(positions[i], weights)
                    
                    # 更新个体最佳
                    if current_score > personal_best_scores[i]:
                        personal_best_positions[i] = positions[i].copy()
                        personal_best_scores[i] = current_score
                    
                    # 简单记录解
                    self.archive.append((positions[i].copy(), np.array([current_score])))
            
            # 记录当前迭代的最佳分数
            if self.weight_method == 'weighted_sum':
                best_position, best_score = self._apply_weight_method(weights)
                iteration_best_scores.append(best_score)
            else:
                # 对于帕累托方法，记录档案大小
                iteration_best_scores.append(len(self.archive))
            
            # 记录迭代历史
            self.history.append({
                'positions': positions.copy(),
                'scores': personal_best_scores.copy(),
                'archive': [(p.copy(), s.copy()) for p, s in self.archive] if self.weight_method == 'pareto' else []
            })
        
        # 最终解
        if self.weight_method == 'weighted_sum':
            best_position, best_score = self._apply_weight_method(weights)
            
            # 返回结果
            return {
                'best_position': best_position,
                'best_score': best_score,
                'pareto_front': self.archive,
                'history': self.history,
                'best_scores': iteration_best_scores,
                'method': 'weighted_sum'
            }
        else:
            # 返回帕累托前沿
            return {
                'best_position': None,  # 帕累托方法没有单一最佳解
                'best_score': None,
                'pareto_front': self.archive,
                'history': self.history,
                'best_scores': iteration_best_scores,  # 这里存储的是档案大小历史
                'method': 'pareto'
            }
    
    def visualize_optimization(self, figsize=(12, 10)):
        """
        可视化优化过程
        
        参数:
            figsize: 图表大小
        
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        fig = plt.figure(figsize=figsize)
        
        # 检查是否有优化历史
        if not self.history:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有优化历史数据", ha='center', va='center')
            return fig
        
        # 绘制得分历史
        if self.best_scores_history:
            ax1 = fig.add_subplot(221)
            ax1.plot(self.best_scores_history, marker='o', markersize=3)
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('最佳得分' if self.weight_method == 'weighted_sum' else '非支配解数量')
            ax1.set_title('MOPSO优化历史')
            ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制帕累托前沿（如果有多个目标）
        if self.archive and self.weight_method == 'pareto':
            # 提取帕累托前沿的目标值
            pareto_scores = np.array([scores for _, scores in self.archive])
            
            # 如果是2D问题，绘制2D帕累托前沿
            if pareto_scores.shape[1] == 2:
                ax2 = fig.add_subplot(222)
                ax2.scatter(pareto_scores[:, 0], pareto_scores[:, 1], c='b', marker='o')
                ax2.set_xlabel('目标1')
                ax2.set_ylabel('目标2')
                ax2.set_title('帕累托前沿')
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 如果是3D问题，绘制3D帕累托前沿
            elif pareto_scores.shape[1] == 3:
                ax3 = fig.add_subplot(222, projection='3d')
                ax3.scatter(pareto_scores[:, 0], pareto_scores[:, 1], pareto_scores[:, 2], c='b', marker='o')
                ax3.set_xlabel('目标1')
                ax3.set_ylabel('目标2')
                ax3.set_zlabel('目标3')
                ax3.set_title('帕累托前沿')
        
        # 绘制粒子分布（最后一次迭代）
        if self.history:
            ax4 = fig.add_subplot(223)
            last_positions = self.history[-1]['positions']
            
            # 如果是2D参数空间，直接绘制
            if self.dimensions == 2:
                ax4.scatter(last_positions[:, 0], last_positions[:, 1], c='r', marker='x')
                ax4.set_xlabel('参数1')
                ax4.set_ylabel('参数2')
                ax4.set_title('最终粒子分布')
                ax4.grid(True, linestyle='--', alpha=0.7)
                
                # 添加约束边界
                ax4.plot([self.min_bounds[0], self.max_bounds[0], self.max_bounds[0], self.min_bounds[0], self.min_bounds[0]],
                         [self.min_bounds[1], self.min_bounds[1], self.max_bounds[1], self.max_bounds[1], self.min_bounds[1]],
                         'k--')
            
            # 如果是高维参数空间，绘制前两个维度
            elif self.dimensions > 2:
                ax4.scatter(last_positions[:, 0], last_positions[:, 1], c='r', marker='x')
                ax4.set_xlabel('参数1')
                ax4.set_ylabel('参数2')
                ax4.set_title('最终粒子分布 (前两个维度)')
                ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 添加总结信息
        ax5 = fig.add_subplot(224)
        ax5.axis('off')
        
        info_text = "多目标粒子群优化 (MOPSO)\n\n"
        info_text += f"方法: {self.weight_method}\n"
        info_text += f"粒子数: {self.num_particles}\n"
        info_text += f"迭代次数: {self.max_iter}\n"
        
        if self.weight_method == 'pareto':
            info_text += f"非支配解数量: {len(self.archive)}\n"
        elif self.best_position is not None and self.best_score is not None:
            info_text += f"最佳得分: {self.best_score:.6f}\n"
            info_text += f"最佳位置: {np.round(self.best_position, 4)}\n"
        
        ax5.text(0.1, 0.5, info_text, va='center')
        
        plt.tight_layout()
        return fig 