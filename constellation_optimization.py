"""
增强版星座优化模块，提供高效的星座设计和优化工具
支持多线程并行计算和高级优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import time
import threading
import importlib
import functools
import os
import logging
from optimization_algorithms import GridSearch, ParticleSwarmOptimization, GeneticAlgorithm, MultiObjectivePSO

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConstellationOptimizer:
    """增强版星座优化器，提供多种优化方法和并行计算支持"""
    
    def __init__(self, propagator, designer=None, max_workers=None):
        """
        初始化星座优化器
        
        参数:
        - propagator: 轨道传播器对象
        - designer: 星座设计器对象（可选）
        - max_workers: 最大并行工作进程/线程数（默认为CPU核心数）
        """
        self.propagator = propagator
        if designer is None:
            # 如果未提供designer，尝试从main模块导入并创建
            try:
                from main import ConstellationDesigner
                self.designer = ConstellationDesigner(propagator)
                logger.info("成功创建 ConstellationDesigner 实例")
            except ImportError:
                logger.error("无法导入 ConstellationDesigner，请确保 main.py 文件存在")
                raise
        else:
            self.designer = designer
            
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = None
        self.lock = threading.Lock()
        self.progress_callback = None
        self.is_cancelled = False # 添加取消标志
        
        # 存储优化算法类
        self.optimization_algorithms = {
            'grid': GridSearch,
            'pso': ParticleSwarmOptimization,
            'genetic': GeneticAlgorithm,
            'mopso': MultiObjectivePSO  # 添加多目标PSO
        }
        logger.info(f"已加载优化算法: {list(self.optimization_algorithms.keys())}")
        
    def _init_process_pool(self):
        """初始化进程池（延迟初始化以节省资源）"""
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, value, message=""):
        """更新进度，并检查是否取消"""
        if self.is_cancelled:
            raise InterruptedError("优化已取消") # 抛出异常以中断算法
        if self.progress_callback:
            self.progress_callback(value, message)
    
    def cancel(self):
        """设置取消标志"""
        logger.info("收到取消优化请求")
        self.is_cancelled = True
        # 可以考虑关闭进程池等资源，但这可能比较复杂
        # if self.process_pool:
        #     self.process_pool.shutdown(wait=False, cancel_futures=True)
        # self.thread_pool.shutdown(wait=False, cancel_futures=True)


    def optimize_walker_constellation(self, objective='coverage', 
                                    total_satellites_range=(20, 80),
                                    num_planes_range=(3, 8),
                                    inclination_range=(45, 90),
                                    altitude_range=(500, 1500),
                                    weights=None, 
                                    max_iterations=20, # 对于GA/PSO，这是代数/迭代次数
                                    use_multiprocessing=True,
                                    algorithm='grid',
                                    **algorithm_params):
        """
        优化Walker星座配置
        
        返回:
        - 字典: 包含 'best_config', 'best_score', 'history' (包含 'iterations', 'best_scores')
        或 None 如果优化失败或被取消
        """
        self.is_cancelled = False # 重置取消标志
        start_time = time.time()
        logger.info(f"开始星座优化，算法: {algorithm}, 目标: {objective}")
        logger.info(f"参数范围: Sats={total_satellites_range}, Planes={num_planes_range}, Inc={inclination_range}, Alt={altitude_range}")
        
        # 设置默认权重
        if weights is None:
            weights = {
                'coverage': 0.5,
                'collision_risk': 0.3, 
                'cost': 0.2 # 假设有一个成本指标
            }
        logger.info(f"使用权重: {weights}")
        
        # 创建目标函数 (确保传递正确的参数)
        objective_function = functools.partial(
            self._evaluate_configuration,
            total_satellites_range=total_satellites_range,
            num_planes_range=num_planes_range,
            inclination_range=inclination_range,
            altitude_range=altitude_range, # 添加高度范围以供验证
            weights=weights
        )
        
        # 为MOPSO创建多目标函数
        if algorithm == 'mopso':
            logger.info("使用多目标PSO，准备多目标函数")
            
            # 创建多目标评估函数
            def multi_objective_function(params):
                # 调用评估函数，应确保返回单一分数
                single_score = self._evaluate_configuration(
                    params,
                    total_satellites_range=total_satellites_range,
                    num_planes_range=num_planes_range,
                    inclination_range=inclination_range,
                    altitude_range=altitude_range,
                    weights=None  # 这里不传递权重，由MOPSO内部处理
                )
                
                # 如果评估失败，返回无效分数
                if single_score <= -1:
                    return np.array([-1.0, -1.0, -1.0])  # 所有目标都无效
                
                # 计算各个目标
                sats_raw, planes_raw, inc, alt = params
                sats = int(round(sats_raw))
                planes = int(round(planes_raw))
                
                # 自动调整卫星数以满足Walker约束
                sats_per_plane = max(1, sats // planes)
                sats = sats_per_plane * planes
                
                # 创建星座状态
                try:
                    constellation_states = self.designer.create_walker_constellation(
                        total_satellites=sats,
                        num_planes=planes,
                        relative_spacing=max(1, int(round(sats / (2 * planes)))),
                        inclination=np.radians(inc),
                        altitude=alt * 1000
                    )
                    
                    # 评估各个目标
                    coverage_score = self.designer.calculate_coverage(constellation_states)
                    
                    # 计算碰撞风险
                    from main_enhanced import EnhancedCollisionAnalyzer
                    collision_analyzer = EnhancedCollisionAnalyzer(self.propagator)
                    cov_matrices = [np.eye(6) * 100 for _ in range(len(constellation_states))]
                    avg_risk, _, _ = collision_analyzer.calculate_collision_probability_clustered(
                        constellation_states, cov_matrices
                    )
                    collision_score = 1.0 - avg_risk  # 风险越低分数越高
                    
                    # 计算成本（简化模型，基于卫星数量）
                    cost_score = 1.0 - (sats / total_satellites_range[1])
                    
                    # 返回多个目标的分数
                    return np.array([coverage_score, collision_score, cost_score])
                    
                except Exception as e:
                    logger.error(f"多目标评估失败: {e}")
                    return np.array([-1.0, -1.0, -1.0])  # 返回无效分数
        
        result = None
        optimizer_instance = None
        try:
            # 获取选择的算法类
            AlgorithmClass = self.optimization_algorithms.get(algorithm)
            if AlgorithmClass is None:
                raise ValueError(f"不支持的优化算法: {algorithm}")

            bounds = [
                total_satellites_range,
                num_planes_range,
                inclination_range,
                altitude_range
            ]

            logger.info(f"初始化算法 {algorithm}...")
            # 根据算法初始化优化器
            if algorithm == 'grid':
                steps = algorithm_params.get('steps', [
                    min(10, int(total_satellites_range[1] - total_satellites_range[0]) + 1),
                    min(6, int(num_planes_range[1] - num_planes_range[0]) + 1),
                    min(10, int((inclination_range[1] - inclination_range[0])/5) + 1), # 每5度一个点
                    min(10, int((altitude_range[1] - altitude_range[0])/100) + 1) # 每100km一个点
                ])
                # 确保 steps 元素为整数
                steps = [int(s) for s in steps] 
                logger.info(f"网格搜索步数: {steps}")
                optimizer_instance = AlgorithmClass(
                    objective_func=objective_function,
                    bounds=bounds,
                    steps=steps
                )
            elif algorithm == 'pso':
                optimizer_instance = AlgorithmClass(
                    objective_func=objective_function,
                    bounds=bounds,
                    num_particles=algorithm_params.get('num_particles', 30),
                    max_iter=max_iterations,
                    w=algorithm_params.get('w', 0.7),
                    c1=algorithm_params.get('c1', 1.4),
                    c2=algorithm_params.get('c2', 1.4)
                )
            elif algorithm == 'genetic':
                 optimizer_instance = AlgorithmClass(
                    objective_func=objective_function,
                    bounds=bounds,
                    population_size=algorithm_params.get('population_size', 50),
                    max_generations=max_iterations,
                    crossover_rate=algorithm_params.get('crossover_rate', 0.8),
                    mutation_rate=algorithm_params.get('mutation_rate', 0.2)
                )
            elif algorithm == 'mopso':
                # 使用多目标函数而不是加权评估函数
                optimizer_instance = AlgorithmClass(
                    objective_funcs=multi_objective_function,  # 使用多目标函数
                    bounds=bounds,
                    num_particles=algorithm_params.get('num_particles', 40),
                    max_iter=max_iterations,
                    w=algorithm_params.get('w', 0.7),
                    c1=algorithm_params.get('c1', 1.4),
                    c2=algorithm_params.get('c2', 1.4),
                    weight_method=algorithm_params.get('weight_method', 'weighted_sum'),
                    archive_size=algorithm_params.get('archive_size', 100)
                )

            if optimizer_instance:
                logger.info("设置进度回调...")
                # 注意：这里的 _update_progress 会检查 self.is_cancelled
                optimizer_instance.set_progress_callback(self._update_progress) 
                
                logger.info("执行优化...")
                # 执行优化
                if algorithm == 'mopso':
                    # 对于MOPSO，传递权重向量
                    weight_array = np.array([weights.get('coverage', 0.5), 
                                           weights.get('collision_risk', 0.3), 
                                           weights.get('cost', 0.2)])
                    result = optimizer_instance.optimize(weights=weight_array)
                else:
                    # 其他算法使用标准调用
                    result = optimizer_instance.optimize()
                
                logger.info("优化执行完成")
            else:
                 logger.error("未能初始化优化器实例")


        except InterruptedError:
            logger.warning("优化过程被用户取消")
            return None
        except ImportError as e:
             logger.error(f"导入优化算法时出错: {e}")
             return None
        except Exception as e:
            logger.error(f"优化过程中发生错误: {e}", exc_info=True) # 记录详细堆栈
            return None
        finally:
            # 清理资源等（如果需要）
            pass

        end_time = time.time()
        logger.info(f"优化总耗时: {end_time - start_time:.2f} 秒")

        # 处理和格式化结果
        if algorithm == 'mopso' and result and result.get('pareto_front'):
            # 如果是MOPSO，使用特殊处理
            formatted_result = self._format_mopso_result(result, weights)
        else:
            # 其他算法的标准处理
            if result is None or result.get('best_score', -float('inf')) <= -1: # 检查是否找到有效解
                logger.warning("优化未找到有效配置或得分过低")
                return None
            
            formatted_result = self._format_optimization_result(result, algorithm, weights)
        
        # 添加历史记录以供绘图
        if algorithm == 'grid':
             # 提取 grid search 的历史分数
            history_scores = [item[1] for item in result.get('history', []) if item[1] > -1]
            formatted_result['history'] = {
                 'iterations': list(range(1, len(history_scores) + 1)),
                 'best_scores': np.maximum.accumulate(history_scores) if history_scores else [],
                 'mean_scores': [] # Grid search doesn't have mean score per "iteration"
            }
        elif algorithm in ['pso', 'genetic', 'mopso']:
             # PSO/GA/MOPSO 的 optimize 方法应该直接返回包含 best_scores 的字典
            iterations = list(range(1, len(result.get('best_scores', [])) + 1))
            formatted_result['history'] = {
                 'iterations': iterations,
                 'best_scores': result.get('best_scores', []),
                 # 尝试计算平均分 (如果算法 history 支持)
                 'mean_scores': self._calculate_mean_scores(result.get('history', [])), 
            }
        else:
            formatted_result['history'] = {'iterations': [], 'best_scores': [], 'mean_scores': []}

        logger.info(f"优化成功完成，最佳得分: {formatted_result['best_score']:.4f}")
        logger.debug(f"最佳配置详情: {formatted_result['best_config']}")
        
        # 尝试在后台保存可视化图（可选，主要用于调试）
        if optimizer_instance and hasattr(optimizer_instance, 'visualize_optimization'):
            try:
                fig = optimizer_instance.visualize_optimization()
                save_path = f'{algorithm}_optimization_process.png'
                plt.savefig(save_path)
                plt.close(fig)
                logger.info(f"优化过程图已保存至 {save_path}")
            except Exception as e:
                logger.error(f"{algorithm} 可视化保存失败: {e}")

        return formatted_result

    def _calculate_mean_scores(self, history):
        """从PSO或GA的历史记录中计算平均得分"""
        mean_scores = []
        if not history:
            return mean_scores
        
        try:
            for state in history:
                scores = state.get('scores') # PSO 'scores', GA 'scores'
                if scores is not None and len(scores) > 0:
                     # 过滤掉无效分数 (例如 < 0)
                     valid_scores = scores[scores > -1]
                     if len(valid_scores) > 0:
                         mean_scores.append(np.mean(valid_scores))
                     else:
                         mean_scores.append(0) # 或者 None 或 np.nan
                else:
                     # 如果某次迭代没有分数信息，可以添加一个占位符
                     mean_scores.append(0) # 或者 None 或 np.nan
        except Exception as e:
            logger.warning(f"计算平均分时出错: {e}")
            return [] # 返回空列表表示失败
            
        return mean_scores


    def _format_optimization_result(self, result, algorithm, weights):
        """格式化优化结果为统一格式，并计算最终指标"""
        best_position = result.get('best_position')
        best_score = result.get('best_score', -float('inf'))

        if best_position is None or best_score <= -1:
             logger.warning("格式化结果时发现无效的最佳位置或得分")
             return {'best_score': best_score, 'best_config': None, 'history': {}}

        # 从 best_position 解析参数
        sats = int(round(best_position[0]))
        planes = int(round(best_position[1]))
        inc = best_position[2]
        alt = best_position[3]
        
        # 基本验证
        if planes <= 0 or sats <= 0 or sats % planes != 0:
             logger.warning(f"格式化结果时发现无效参数: sats={sats}, planes={planes}")
             return {'best_score': best_score, 'best_config': None, 'history': {}}

        relative_spacing = 1
        if planes > 1:
            # 确保 relative_spacing 为正整数
            relative_spacing = max(1, int(round(sats / (2 * planes)))) 
            # Walker 定义 T/P/F, F=relative_spacing
            # T = sats, P = planes
            # F (Phasing Parameter) ranges from 0 to P-1 normally
            # Here relative_spacing seems different? Let's stick to the original formula for now
            # Needs clarification on what 'relative_spacing' means here. Assuming it's the 'F' parameter related.
            # Original formula: relative_spacing = sats // (2 * planes) # This might lead to 0
            # Let's use a common definition or ensure it's at least 1 if planes > 1
            # Revisit this calculation based on the exact definition used in ConstellationDesigner

        best_config = {
            'total_satellites': sats,
            'num_planes': planes,
            'inclination': inc,
            'altitude': alt,
            'relative_spacing': relative_spacing, # 使用计算出的相对间距
            'eccentricity': 0.0 # 假设为圆轨道
        }
        
        final_metrics = {}
        constellation_states = None
        # 计算最终的详细指标
        try:
            logger.debug(f"为最佳配置计算最终指标: {best_config}")
            # 使用 designe r创建最终星座状态
            constellation_states = self.designer.create_walker_constellation(
                total_satellites=best_config['total_satellites'],
                num_planes=best_config['num_planes'],
                relative_spacing=best_config['relative_spacing'], # 使用配置中的间距
                inclination=np.radians(best_config['inclination']), # 角度转弧度
                altitude=best_config['altitude'] * 1000 # 公里转米
            )
            
            # 评估最终星座
            final_metrics = self.designer.evaluate_constellation(constellation_states, weights)
            logger.debug(f"最终指标: {final_metrics}")
            
        except Exception as e:
            logger.error(f"计算最终指标时出错: {e}", exc_info=True)
            # 即使指标计算失败，仍然返回找到的最佳参数配置
        
        # 组合最终结果
        formatted_result = {
            'best_score': final_metrics.get('overall_score', best_score), # 使用评估得出的最终分数
            'best_config': best_config,
            'final_metrics': final_metrics,
            'constellation_states': constellation_states, # 返回星座状态，用于UI可视化
            'algorithm': algorithm
            # history 会在 optimize_walker_constellation 中添加
        }
        
        return formatted_result

    def _evaluate_configuration(self, params, total_satellites_range,
                              num_planes_range, inclination_range, altitude_range, weights):
        """
        评估单个星座配置的目标函数。
        params: 优化算法传递的参数列表/数组 [sats, planes, inc, alt]
        """
        try:
            sats_raw, planes_raw, inc, alt = params
            
            # --- 参数验证和处理 ---
            # 使用向下取整确保参数有效性
            sats = int(round(sats_raw))
            planes = int(round(planes_raw))

            # 1. 基本范围检查并修正
            sats = max(total_satellites_range[0], min(sats, total_satellites_range[1]))
            planes = max(num_planes_range[0], min(planes, num_planes_range[1]))
            inc = max(inclination_range[0], min(inc, inclination_range[1]))
            alt = max(altitude_range[0], min(alt, altitude_range[1]))

            # 2. 确保平面数和卫星数大于0
            if planes <= 0:
                planes = num_planes_range[0]
            if sats <= 0:
                sats = total_satellites_range[0]

            # 3. 关键修改: 自动调整卫星数以满足Walker星座约束
            # 确保卫星数能被平面数整除
            sats_per_plane = sats // planes
            if sats_per_plane == 0:
                sats_per_plane = 1
            sats = sats_per_plane * planes  # 调整后的卫星总数
            
            # 检查调整后的参数是否仍在范围内
            if not (total_satellites_range[0] <= sats <= total_satellites_range[1]):
                logger.debug(f"调整后卫星数 {sats} 超出范围 {total_satellites_range}")
                # 尝试找到最接近的有效值
                if sats < total_satellites_range[0]:
                    planes = max(1, total_satellites_range[0] // planes)
                    sats = planes * planes
                elif sats > total_satellites_range[1]:
                    planes = min(planes, total_satellites_range[1] // max(1, sats_per_plane))
                    sats = planes * sats_per_plane
            
            # 计算相对间距 (同 _format_optimization_result)
            relative_spacing = 1
            if planes > 1:
                 relative_spacing = max(1, int(round(sats / (2 * planes))))

            # --- 创建和评估星座 ---
            logger.debug(f"评估配置: Sats={sats}, Planes={planes}, Inc={inc:.2f}, Alt={alt:.1f}, Spacing={relative_spacing}")

            # 调用designer创建星座状态
            constellation_states = self.designer.create_walker_constellation(
                total_satellites=sats, 
                num_planes=planes, 
                relative_spacing=relative_spacing, 
                inclination=np.radians(inc), # 角度转弧度
                altitude=alt * 1000 # 公里转米
            )
            
            # 评估星座指标
            metrics = self.designer.evaluate_constellation(constellation_states, weights)
            
            # 确保返回一个有效的数值分数
            score = metrics.get('overall_score', -1.0)
            if score is None or np.isnan(score):
                 logger.warning(f"评估返回无效分数 for config: {params}. Metrics: {metrics}")
                 return -1.0

            logger.debug(f"配置 [sats={sats},planes={planes},inc={inc:.1f},alt={alt:.1f}] 得分: {score:.4f}")
            return float(score) # 确保返回浮点数

        except Exception as e:
            logger.error(f"评估配置 {params} 时出错: {e}", exc_info=True)
            return -1.0 # 返回负分表示评估失败
            
    # 移除 _grid_search_optimization, _pso_optimization, _genetic_optimization 内部实现
    # 因为优化逻辑在 optimization_algorithms.py 中

    # 移除 _process_grid_result 方法，逻辑合并到 _format_optimization_result
    
    # 移除 _basic_grid_search 方法，不再需要

    # 其他辅助方法 (如多进程/线程并行评估) 可以根据需要添加
    # ...

    def _format_mopso_result(self, result, weights):
        """格式化MOPSO优化结果"""
        if not result or 'pareto_front' not in result or not result['pareto_front']:
            logger.warning("MOPSO结果中没有帕累托前沿")
            return {'best_score': -1.0, 'best_config': None, 'history': {}}
        
        # 获取帕累托前沿解
        pareto_front = result['pareto_front']
        
        # 如果使用加权和方法，已经选择了最佳解
        if result['method'] == 'weighted_sum' and result['best_position'] is not None:
            best_position = result['best_position']
            best_score = result['best_score']
        else:
            # 否则，根据权重从帕累托前沿中选择最佳解
            best_position = None
            best_weighted_score = -float('inf')
            
            # 创建权重向量
            weight_array = np.array([
                weights.get('coverage', 0.5),
                weights.get('collision_risk', 0.3),
                weights.get('cost', 0.2)
            ])
            
            # 遍历帕累托前沿，找出加权得分最高的解
            for position, scores in pareto_front:
                if len(scores) != len(weight_array):
                    continue  # 跳过维度不匹配的情况
                    
                weighted_score = np.sum(scores * weight_array)
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_position = position
            
            best_score = best_weighted_score
        
        if best_position is None:
            logger.warning("未能从帕累托前沿中选择最佳解")
            return {'best_score': -1.0, 'best_config': None, 'history': {}}
        
        # 解析最佳位置
        sats = int(round(best_position[0]))
        planes = int(round(best_position[1]))
        inc = best_position[2]
        alt = best_position[3]
        
        # 确保参数有效
        sats_per_plane = max(1, sats // planes)
        sats = sats_per_plane * planes  # 调整后的卫星总数
        
        # 计算相对间距
        relative_spacing = 1
        if planes > 1:
            relative_spacing = max(1, int(round(sats / (2 * planes))))
        
        # 创建最佳配置字典
        best_config = {
            'total_satellites': sats,
            'num_planes': planes,
            'inclination': inc,
            'altitude': alt,
            'relative_spacing': relative_spacing,
            'eccentricity': 0.0
        }
        
        # 为最佳配置计算详细指标
        final_metrics = {}
        constellation_states = None
        try:
            # 创建星座状态
            constellation_states = self.designer.create_walker_constellation(
                total_satellites=best_config['total_satellites'],
                num_planes=best_config['num_planes'],
                relative_spacing=best_config['relative_spacing'],
                inclination=np.radians(best_config['inclination']),  # 角度转弧度
                altitude=best_config['altitude'] * 1000  # 公里转米
            )
            
            # 评估星座
            final_metrics = self.designer.evaluate_constellation(constellation_states, weights)
        except Exception as e:
            logger.error(f"计算MOPSO最终指标时出错: {e}")
        
        # 组合结果
        formatted_result = {
            'best_score': final_metrics.get('overall_score', best_score),
            'best_config': best_config,
            'final_metrics': final_metrics,
            'constellation_states': constellation_states,
            'algorithm': 'mopso',
            'pareto_front': pareto_front  # 保留帕累托前沿信息
        }
        
        return formatted_result

# Example usage (for testing)
if __name__ == '__main__':
    # 需要一个 Propagator 和 Designer 实例来测试
    # from main import OrbitPropagator, ConstellationDesigner 
    try:
        # 这是一个简化的测试，实际使用需要正确的 Propagator 和 Designer
        class MockPropagator:
            pass
        class MockDesigner:
            def create_walker_constellation(self, total_satellites, num_planes, relative_spacing, inclination, altitude, eccentricity=0.0):
                # 返回模拟的状态 (例如，数量符合即可)
                logger.info(f"Mock Create: T={total_satellites}, P={num_planes}, F={relative_spacing}, i={np.degrees(inclination):.1f}, alt={altitude/1000:.1f}")
                if total_satellites <= 0 or num_planes <=0: raise ValueError("Invalid sats/planes")
                return np.random.rand(total_satellites, 6) 
            
            def evaluate_constellation(self, constellation_states, weights):
                 # 返回模拟的指标和分数
                 num_sats = len(constellation_states)
                 coverage = np.random.rand() * (num_sats / 50.0) # 简单模拟覆盖率
                 collision = np.random.rand() * (1.0 - num_sats / 100.0) # 简单模拟碰撞风险
                 score = (weights.get('coverage', 0.5) * coverage + 
                          weights.get('collision_risk', 0.3) * (1-collision) + # 假设风险越小越好
                          weights.get('cost', 0.2) * (1.0 - num_sats / 100.0) # 简单成本模型
                         )
                 logger.info(f"Mock Evaluate (sats={num_sats}): Cov={coverage:.3f}, Col={collision:.3f}, Score={score:.3f}")
                 return {'coverage': coverage, 'collision_risk': collision, 'overall_score': score}

        mock_propagator = MockPropagator()
        mock_designer = MockDesigner()
        
        optimizer = ConstellationOptimizer(mock_propagator, mock_designer)

        # 设置一个简单的进度回调
        def progress_callback(value, message):
            print(f"Progress: {value:.1f}% - {message}")
        optimizer.set_progress_callback(progress_callback)

        print("--- 测试网格搜索 ---")
        result_grid = optimizer.optimize_walker_constellation(
             algorithm='grid',
             total_satellites_range=(10, 20), # 缩小范围以加快测试
             num_planes_range=(2, 4),
             inclination_range=(50, 60),
             altitude_range=(600, 700),
             max_iterations=5, # 对grid无效，但保留参数
             # algorithm_params={'steps': [3, 2, 2, 2]} # 指定更少的步数
        )
        if result_grid:
            print("Grid Search Best Score:", result_grid['best_score'])
            print("Grid Search Best Config:", result_grid['best_config'])
        else:
             print("Grid Search failed or found no valid solution.")

        print("\n--- 测试粒子群优化 ---")
        result_pso = optimizer.optimize_walker_constellation(
             algorithm='pso',
             total_satellites_range=(10, 30), 
             num_planes_range=(2, 5),
             inclination_range=(45, 90),
             altitude_range=(500, 1000),
             max_iterations=10, # 减少迭代次数
             algorithm_params={'num_particles': 10} # 减少粒子数
        )
        if result_pso:
             print("PSO Best Score:", result_pso['best_score'])
             print("PSO Best Config:", result_pso['best_config'])
             # print("PSO History:", result_pso['history'])
        else:
             print("PSO failed or found no valid solution.")

        print("\n--- 测试遗传算法 ---")
        result_ga = optimizer.optimize_walker_constellation(
            algorithm='genetic',
            total_satellites_range=(10, 30), 
            num_planes_range=(2, 5),
            inclination_range=(45, 90),
            altitude_range=(500, 1000),
            max_iterations=10, # 减少代数
            algorithm_params={'population_size': 15} # 减少种群大小
        )
        if result_ga:
             print("GA Best Score:", result_ga['best_score'])
             print("GA Best Config:", result_ga['best_config'])
             # print("GA History:", result_ga['history'])
        else:
              print("GA failed or found no valid solution.")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        traceback.print_exc()

