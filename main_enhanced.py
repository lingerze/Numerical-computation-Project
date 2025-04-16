import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import font as tkfont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
import pandas as pd
import time
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from multiprocessing import Pool
import matplotlib.font_manager as fm

# 设置matplotlib中文字体
try:
    # 检查系统中的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    print("警告: 中文字体设置失败，可能会导致中文显示不正确")

# 导入原有模块
import main as original_module

# 导入新的功能模块
try:
    from numerical_methods import NumericalIntegrators, ErrorAnalysis
    numerical_methods_available = True
    print("成功导入数值方法模块")
except ImportError as e:
    numerical_methods_available = False
    print(f"警告: 数值方法模块导入失败: {e}，部分功能将不可用")

try:
    from visualization_enhancements import create_earth_plot, plot_orbit_with_evolution, InteractivePlotWindow
    visualization_available = True
    print("成功导入可视化增强模块")
except ImportError as e:
    visualization_available = False
    print(f"警告: 可视化增强模块导入失败: {e}，部分功能将不可用")

try:
    from perturbation_models import GravityField, AtmosphericModel, SolarRadiationPressure, LunarSolarGravity
    perturbation_models_available = True
    print("成功导入高级摄动模型")
except ImportError as e:
    perturbation_models_available = False
    print(f"警告: 高级摄动模型导入失败: {e}，将使用基本模型")

try:
    from ui_extensions import (AdvancedAnalysisWindow, MethodComparisonWindow, 
                              ConstellationOptimizationWindow, ErrorAnalysisWindow, add_menubar)
    ui_extensions_available = True
    print("成功导入UI扩展模块")
except ImportError as e:
    ui_extensions_available = False
    print(f"警告: UI扩展模块导入失败: {e}，部分功能将不可用")


# 增强版碰撞分析器类
class EnhancedCollisionAnalyzer:
    """增强版碰撞分析器,支持高效的碰撞概率计算和自适应规避"""
    
    def __init__(self, propagator, logger=None):
        self.propagator = propagator
        self.logger = logger
    
    def log(self, message):
        """记录日志"""
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def calculate_minimum_distance(self, state1, state2, t_span=(0, 86400), dt=60):
        """计算两颗卫星之间的最小距离"""
        t_range = np.arange(t_span[0], t_span[1], dt)
        min_distance = float('inf')
        min_distance_time = 0
        
        for t in t_range:
            # 简化计算,直接使用线性传播
            pos1 = state1[:3] + state1[3:] * t
            pos2 = state2[:3] + state2[3:] * t
            distance = np.linalg.norm(pos1 - pos2)
            
            if distance < min_distance:
                min_distance = distance
                min_distance_time = t
        
        return min_distance, min_distance_time
    
    def calculate_collision_probability_analytical(self, state1, state2, cov1, cov2, safe_distance=1000.0):
        """使用解析方法计算碰撞概率"""
        # 获取当前位置
        pos1 = state1[:3]
        pos2 = state2[:3]
        
        # 计算相对位置
        rel_pos = pos1 - pos2
        
        # 计算相对位置协方差矩阵
        rel_cov = cov1[:3, :3] + cov2[:3, :3]
        
        # 计算马氏距离
        try:
            mahalanobis_dist = np.sqrt(rel_pos.T @ np.linalg.inv(rel_cov) @ rel_pos)
        except np.linalg.LinAlgError:
            # 协方差矩阵可能是奇异的,使用伪逆
            rel_cov_pinv = np.linalg.pinv(rel_cov)
            mahalanobis_dist = np.sqrt(rel_pos.T @ rel_cov_pinv @ rel_pos)
        
        # 计算碰撞概率
        p_collision = 1 - np.exp(-0.5 * (safe_distance / mahalanobis_dist)**2)
        
        return min(p_collision, 1.0)
    
    def calculate_collision_probability_clustered(self, constellation_states, cov_matrices=None, 
                                                safe_distance=1000.0, cluster_size=5):
        """使用簇分割法计算碰撞概率"""
        num_satellites = len(constellation_states)
        
        # 默认协方差矩阵
        if cov_matrices is None:
            cov_matrices = [np.eye(6) * 100 for _ in range(num_satellites)]
            
        # 将卫星划分为多个簇
        num_clusters = (num_satellites + cluster_size - 1) // cluster_size
        clusters = []
        cluster_covs = []
        
        for i in range(0, num_satellites, cluster_size):
            end_idx = min(i + cluster_size, num_satellites)
            clusters.append(constellation_states[i:end_idx])
            cluster_covs.append(cov_matrices[i:end_idx])
        
        total_risk = 0.0
        close_approaches = 0
        collision_pairs = []
        
        # 计算簇间碰撞风险
        for i in range(num_clusters):
            for j in range(i+1, num_clusters):
                for idx1, state1 in enumerate(clusters[i]):
                    for idx2, state2 in enumerate(clusters[j]):
                        # 计算距离
                        distance = np.linalg.norm(state1[:3] - state2[:3])
                        
                        # 如果距离小于安全距离,计算详细的碰撞概率
                        if distance < safe_distance * 5:  # 预筛选
                            cov1 = cluster_covs[i][idx1]
                            cov2 = cluster_covs[j][idx2]
                            
                            probability = self.calculate_collision_probability_analytical(
                                state1, state2, cov1, cov2, safe_distance)
                            
                            if probability > 0.001:  # 仅记录有意义的碰撞风险
                                total_risk += probability
                                close_approaches += 1
                                collision_pairs.append(((i*cluster_size + idx1, j*cluster_size + idx2), probability))
        
        # 计算平均风险
        total_pairs = num_satellites * (num_satellites - 1) // 2
        avg_risk = total_risk / total_pairs if total_pairs > 0 else 0
        
        return avg_risk, close_approaches, collision_pairs
    
    def design_avoidance_maneuver(self, state, threat_state, delta_v_max=1.0):
        """设计碰撞规避机动"""
        # 获取相对位置和速度
        rel_pos = state[:3] - threat_state[:3]
        rel_vel = state[3:] - threat_state[3:]
        
        # 计算当前最小距离和发生时间
        min_dist, t_min = self.calculate_minimum_distance(state, threat_state)
        
        # 创建规避方向
        # 选择垂直于相对位置和速度的方向
        if np.linalg.norm(rel_pos) > 0 and np.linalg.norm(rel_vel) > 0:
            cross_prod = np.cross(rel_pos, rel_vel)
            if np.linalg.norm(cross_prod) > 0:
                avoidance_dir = cross_prod / np.linalg.norm(cross_prod)
            else:
                # 如果叉积接近零,选择与相对位置垂直的任意方向
                avoidance_dir = np.array([rel_pos[1], -rel_pos[0], 0])
                if np.linalg.norm(avoidance_dir) > 0:
                    avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
                else:
                    avoidance_dir = np.array([1, 0, 0])  # 默认方向
        else:
            avoidance_dir = np.array([1, 0, 0])  # 默认方向
        
        # 计算规避速度增量
        delta_v = avoidance_dir * delta_v_max
        
        # 验证规避效果
        new_state = state.copy()
        new_state[3:] += delta_v
        
        new_min_dist, _ = self.calculate_minimum_distance(new_state, threat_state)
        
        # 如果规避后距离反而减小,尝试相反方向
        if new_min_dist < min_dist:
            delta_v = -delta_v
            new_state[3:] = state[3:] + delta_v
            new_min_dist, _ = self.calculate_minimum_distance(new_state, threat_state)
        
        return delta_v, min_dist, new_min_dist

    def optimize_constellation(self, constellation_states, designer, cov_matrices=None, 
                             max_maneuver_dv=0.1, max_iterations=10, coverage_weight=0.5):
        """优化星座构型以降低碰撞风险并保持覆盖性能"""
        num_satellites = len(constellation_states)
        
        # 默认协方差矩阵
        if cov_matrices is None:
            cov_matrices = [np.eye(6) * 100 for _ in range(num_satellites)]
        
        # 计算初始碰撞风险
        avg_risk, close_approaches, collision_pairs = self.calculate_collision_probability_clustered(
            constellation_states, cov_matrices)
        
        # 计算初始覆盖率
        initial_coverage = designer.calculate_coverage(constellation_states)
        
        self.log(f"初始碰撞风险: {avg_risk:.6f}, 潜在碰撞对数: {close_approaches}")
        self.log(f"初始覆盖率: {initial_coverage*100:.2f}%")
        
        # 保存原始星座状态
        original_states = constellation_states.copy()
        
        # 迭代优化
        for iteration in range(max_iterations):
            if len(collision_pairs) == 0 or avg_risk < 0.001:
                self.log("碰撞风险已降至可接受水平,优化结束")
                break
            
            # 按碰撞概率排序,处理风险最高的对
            collision_pairs.sort(key=lambda x: x[1], reverse=True)
            highest_risk_pair, highest_risk = collision_pairs[0]
            
            sat_i, sat_j = highest_risk_pair
            self.log(f"优化迭代 {iteration+1}: 处理卫星对 ({sat_i}, {sat_j}), 碰撞风险 {highest_risk:.6f}")
            
            # 为两颗卫星设计规避机动
            delta_v_i, dist_before, dist_after_i = self.design_avoidance_maneuver(
                constellation_states[sat_i], constellation_states[sat_j], max_maneuver_dv/2)
            
            delta_v_j, _, dist_after_j = self.design_avoidance_maneuver(
                constellation_states[sat_j], constellation_states[sat_i], max_maneuver_dv/2)
            
            # 应用规避机动
            new_states = constellation_states.copy()
            new_states[sat_i][3:] += delta_v_i
            new_states[sat_j][3:] += delta_v_j
            
            # 计算新的覆盖率
            new_coverage = designer.calculate_coverage(new_states)
            coverage_change = (new_coverage - initial_coverage) / initial_coverage
            
            # 评估机动的影响
            if coverage_change < -coverage_weight:
                self.log(f"放弃机动: 覆盖率下降过多 ({coverage_change*100:.2f}%)")
                
                # 从碰撞对列表中移除该对
                collision_pairs.pop(0)
                continue
            
            # 更新星座状态
            constellation_states = new_states
            
            # 更新协方差矩阵
            cov_matrices[sat_i][3:, 3:] += np.outer(delta_v_i, delta_v_i) * 0.1
            cov_matrices[sat_j][3:, 3:] += np.outer(delta_v_j, delta_v_j) * 0.1
            
            # 重新计算碰撞风险
            avg_risk, close_approaches, collision_pairs = self.calculate_collision_probability_clustered(
                constellation_states, cov_matrices)
            
            self.log(f"规避后碰撞风险: {avg_risk:.6f}, 潜在碰撞对数: {close_approaches}")
            self.log(f"规避后覆盖率: {new_coverage*100:.2f}% (变化: {coverage_change*100:+.2f}%)")
            self.log(f"星间最小距离从 {dist_before:.2f}m 增加到 {max(dist_after_i, dist_after_j):.2f}m")
        
        # 计算总的燃料消耗
        total_dv = 0
        for i in range(num_satellites):
            dv = np.linalg.norm(constellation_states[i][3:] - original_states[i][3:])
            total_dv += dv
        
        self.log(f"优化完成! 总速度增量: {total_dv:.4f} m/s")
        
        return constellation_states, cov_matrices, avg_risk, new_coverage


class EnhancedConstellationApp(original_module.ConstellationSimulationApp):
    """增强版星座仿真应用程序"""
    
    def __init__(self, root):
        # 调用原始初始化方法
        super().__init__(root)
        
        # 设置窗口标题
        self.root.title("增强版近地星座构型演化仿真系统")
        
        # 添加额外的高级模型
        self.init_advanced_models()
        
        # 添加额外的控制区域
        self.add_advanced_controls()
        
        # 添加菜单栏
        if ui_extensions_available:
            self.init_menubar()
        
        # 初始化分析结果存储
        self.analysis_results = {}
        
        # 创建数据存储目录
        os.makedirs("results", exist_ok=True)
        
        # 初始化增强版碰撞分析器
        self.collision_analyzer = EnhancedCollisionAnalyzer(self.propagator, self.log)
    
    def init_advanced_models(self):
        """初始化高级物理模型"""
        if perturbation_models_available:
            self.gravity_field = GravityField(max_degree=4)
            self.atmospheric_model = AtmosphericModel()
            self.srp_model = SolarRadiationPressure()
            self.third_body = LunarSolarGravity()
        
        # 扩展数值积分方法列表
        self.integration_methods = [
            ("RK45 (自适应步长)", "RK45"), 
            ("经典四阶龙格库塔", "RK4"),
            ("速度Verlet", "Verlet"),
            ("辛欧拉", "Symplectic"),
            ("欧拉法", "Euler")
        ]
    
    def add_advanced_controls(self):
        """添加高级控制选项"""
        # 创建高级选项区域
        advanced_frame = ttk.LabelFrame(self.control_frame, text="高级选项")
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 数值方法选择
        ttk.Label(advanced_frame, text="数值方法:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.method_var = tk.StringVar(value="RK45")
        method_combo = ttk.Combobox(advanced_frame, textvariable=self.method_var, 
                               values=[method[1] for method in self.integration_methods])
        method_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 高级摄动选项
        if perturbation_models_available:
            # 高阶引力场
            self.high_order_gravity_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(advanced_frame, text="高阶引力场", variable=self.high_order_gravity_var).grid(
                row=1, column=0, sticky=tk.W, padx=5, pady=5)
            
            # 第三体摄动
            self.third_body_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(advanced_frame, text="月日摄动", variable=self.third_body_var).grid(
                row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 错误分析选项
        self.error_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="保守量误差分析", variable=self.error_analysis_var).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 方法比较按钮
        if numerical_methods_available:
            self.compare_btn = ttk.Button(advanced_frame, text="方法比较", command=self.compare_numerical_methods)
            self.compare_btn.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 碰撞规避选项
        self.collision_avoidance_frame = ttk.LabelFrame(advanced_frame, text="碰撞规避选项")
        self.collision_avoidance_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # 启用碰撞规避
        self.enable_avoidance_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.collision_avoidance_frame, text="启用自适应碰撞规避", 
                       variable=self.enable_avoidance_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 最大机动增量
        ttk.Label(self.collision_avoidance_frame, text="最大机动增量 (m/s):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_dv_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.collision_avoidance_frame, textvariable=self.max_dv_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 覆盖率权重
        ttk.Label(self.collision_avoidance_frame, text="覆盖率权重:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.coverage_weight_var = tk.DoubleVar(value=0.5)
        ttk.Entry(self.collision_avoidance_frame, textvariable=self.coverage_weight_var, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
    
    def init_menubar(self):
        """初始化菜单栏"""
        callbacks = {
            "save_constellation": self.save_constellation,
            "load_constellation": self.load_constellation,
            "export_data": self.export_data,
            "orbit_evolution": self.show_orbit_evolution,
            "compare_methods": self.compare_numerical_methods,
            "perturbation_analysis": self.show_perturbation_analysis,
            "collision_analysis": self.show_collision_analysis,
            "coverage_analysis": self.show_coverage_analysis,
            "optimize_constellation": self.optimize_constellation,
            "error_analysis": self.show_error_analysis,
            "settings": self.show_settings,
            "show_help": self.show_help,
            "show_about": self.show_about
        }
        
        self.menubar = add_menubar(self.root, callbacks)
    
    def run_simulation(self):
        """运行轨道传播仿真（覆盖原方法）"""
        if self.constellation_states is None:
            messagebox.showwarning("警告", "请先创建星座!")
            return
        
        self.log("开始轨道传播仿真...")
        
        # 获取仿真参数
        sim_duration = self.sim_duration_var.get() * self.propagator.DAY_SEC  # 转换为秒
        time_step = self.time_step_var.get()
        
        # 获取物理参数
        am_ratio = self.am_ratio_var.get()
        cd = self.cd_var.get()
        cr = self.cr_var.get()
        
        # 获取摄动开关
        enable_j2 = self.j2_var.get()
        enable_drag = self.drag_var.get()
        enable_srp = self.srp_var.get()
        
        # 获取数值方法
        method = self.method_var.get()
        
        # 时间范围
        t_span = (0, sim_duration)
        
        # 执行碰撞规避优化(如果启用)
        if self.enable_avoidance_var.get():
            self.log("执行自适应碰撞规避优化...")
            max_dv = self.max_dv_var.get()
            coverage_weight = self.coverage_weight_var.get()
            
            # 为每个卫星创建一个协方差矩阵
            cov_matrices = [np.eye(6) * 100 for _ in range(len(self.constellation_states))]
            
            # 执行优化
            optimized_states, _, risk, coverage = self.collision_analyzer.optimize_constellation(
                self.constellation_states.copy(), self.designer, cov_matrices, 
                max_maneuver_dv=max_dv, coverage_weight=coverage_weight)
            
            # 更新星座状态
            self.constellation_states = optimized_states
            self.log(f"优化后碰撞风险: {risk:.6f}")
            self.log(f"优化后覆盖率: {coverage*100:.2f}%")
            
            # 重新绘制星座
            self.plot_constellation(self.constellation_states)
        
        # 存储传播结果
        self.propagated_states = []
        self.propagation_times = []
        
        # 存储误差分析数据（如果启用）
        if self.error_analysis_var.get() and numerical_methods_available:
            self.energy_errors = []
            self.momentum_errors = []
        
        # 进度更新
        total_sats = len(self.constellation_states)
        
        # 记录开始时间
        start_time = time.time()
        
        # 确定步数
        if method != "RK45":  # 非自适应方法需要指定步数
            steps = int(sim_duration / time_step)
        else:
            steps = None
        
        for i, initial_state in enumerate(self.constellation_states):
            # 更新进度
            progress = (i / total_sats) * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()
            
            # 传播轨道
            times, states = self.propagator.propagate_orbit(
                initial_state, t_span, am_ratio, cd, cr,
                method=method, steps=steps,
                enable_j2=enable_j2, enable_drag=enable_drag, enable_srp=enable_srp
            )
            
            # 存储结果
            self.propagated_states.append(states)
            self.propagation_times.append(times)
            
            # 误差分析
            if self.error_analysis_var.get() and numerical_methods_available:
                energy_error = ErrorAnalysis.energy_conservation_error(states, self.propagator.MU)
                momentum_error = ErrorAnalysis.angular_momentum_error(states)
                self.energy_errors.append(energy_error)
                self.momentum_errors.append(momentum_error)
            
            self.log(f"卫星 {i+1}/{total_sats} 轨道传播完成")
        
        # 完成进度
        self.progress_var.set(100)
        
        # 记录总计算时间
        total_time = time.time() - start_time
        self.log(f"仿真完成! 总时长: {sim_duration/self.propagator.DAY_SEC:.1f} 天")
        self.log(f"计算耗时: {total_time:.2f} 秒")
        
        # 分析轨道演化
        self.analyze_orbit_evolution()
    
    def optimize_constellation(self):
        """打开星座优化窗口"""
        try:
            self.log("正在加载星座优化模块...")
            
            # 导入UI扩展模块
            try:
                from ui_extensions import ConstellationOptimizationWindow
            except ImportError as e:
                raise ImportError(f"无法导入ConstellationOptimizationWindow: {str(e)}\n请确保ui_extensions.py文件存在并包含此类。")
                
            # 导入星座设计器模块
            try:
                from main import ConstellationDesigner
            except ImportError as e:
                raise ImportError(f"无法导入ConstellationDesigner: {str(e)}\n请确保main.py文件存在并包含此类。")
                
            # 检查是否导入了constellation_optimization模块
            try:
                import constellation_optimization
                self.log("成功导入constellation_optimization模块")
            except ImportError as e:
                raise ImportError(f"无法导入constellation_optimization模块: {str(e)}\n请确保constellation_optimization.py文件存在。")
            
            # 检查是否存在星座设计器
            if not hasattr(self, 'designer'):
                self.log("创建星座设计器...")
                self.designer = ConstellationDesigner(self.propagator)
                
            # 创建星座优化窗口
            self.log("创建星座优化窗口...")
            opt_window = ConstellationOptimizationWindow(self.root, self.designer)
            
            # 记录日志
            self.log("星座优化窗口已启动")
            
        except ImportError as e:
            # 处理导入错误
            error_msg = f"导入星座优化模块失败: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("导入错误", error_msg)
        except Exception as e:
            error_msg = f"打开星座优化窗口时出错：{str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)
            import traceback
            traceback.print_exc()
            self.log("详细错误信息已打印到控制台")
    
    def compare_numerical_methods(self):
        """数值方法比较窗口"""
        try:
            # 导入UI扩展模块
            from ui_extensions import MethodComparisonWindow
            
            # 创建方法比较窗口
            MethodComparisonWindow(self.root, "数值方法比较", self.propagator)
        except Exception as e:
            messagebox.showerror("错误", f"打开数值方法比较窗口时出错：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def show_orbit_evolution(self):
        """显示轨道演化分析"""
        if self.propagated_states is None:
            messagebox.showwarning("警告", "请先运行仿真!")
            return
        
        if not ui_extensions_available:
            messagebox.showwarning("警告", "UI扩展模块未加载!")
            return
        
        # 创建分析窗口
        analysis_window = AdvancedAnalysisWindow(self.root, "轨道演化分析")
        
        # 选择第一颗卫星进行分析
        states = self.propagated_states[0]
        times = self.propagation_times[0]
        
        # 转换为开普勒轨道根数
        kepler_history = np.array([self.propagator.cartesian_to_kepler(state) for state in states])
        time_days = times / self.propagator.DAY_SEC
        
        # 在分析窗口中创建图表
        analysis_window.create_orbit_evolution_plot(kepler_history, time_days)
    
    def show_perturbation_analysis(self):
        """分析不同摄动力对轨道的影响"""
        # 检查是否有初始状态
        if not hasattr(self, 'initial_state') or self.initial_state is None:
            tk.messagebox.showwarning("警告", "没有初始状态数据，请先设置轨道参数")
            return
        
        # 创建进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("摄动分析进度")
        progress_window.geometry("400x150")
        
        # 添加进度标签和进度条
        ttk.Label(progress_window, text="正在计算不同摄动模型下的轨道演化...").pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        status_var = tk.StringVar(value="准备中...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)
        
        # 更新进度的函数
        def update_progress(value, message):
            progress_var.set(value)
            status_var.set(message)
            progress_window.update()
        
        # 获取当前参数
        t_span = (0, float(self.parameter_vars['duration'].get()) * 86400)  # 转换为秒
        steps = int(self.parameter_vars['steps'].get())
        
        # 进行不同摄动条件下的轨道传播
        try:
            # 纯二体问题传播 (无摄动)
            update_progress(10, "计算二体模型轨道...")
            t_ref, ref_states = self.propagator.propagate_orbit(
                self.initial_state, t_span, steps=steps,
                enable_j2=False, enable_drag=False, enable_srp=False
            )
            
            # 只考虑J2摄动
            update_progress(30, "计算J2摄动轨道...")
            t_j2, j2_states = self.propagator.propagate_orbit(
                self.initial_state, t_span, steps=steps,
                enable_j2=True, enable_drag=False, enable_srp=False
            )
            
            # 只考虑大气阻力
            update_progress(50, "计算大气阻力轨道...")
            t_drag, drag_states = self.propagator.propagate_orbit(
                self.initial_state, t_span, steps=steps,
                enable_j2=False, enable_drag=True, enable_srp=False
            )
            
            # 只考虑太阳辐射压
            update_progress(70, "计算太阳辐射压轨道...")
            t_srp, srp_states = self.propagator.propagate_orbit(
                self.initial_state, t_span, steps=steps,
                enable_j2=False, enable_drag=False, enable_srp=True
            )
            
            # 所有摄动一起考虑
            update_progress(90, "计算完全摄动轨道...")
            t_all, all_states = self.propagator.propagate_orbit(
                self.initial_state, t_span, steps=steps,
                enable_j2=True, enable_drag=True, enable_srp=True
            )
            
            update_progress(100, "计算完成，准备显示结果...")
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 导入高级可视化模块
            from advanced_visualization import AdvancedVisualizationWindow, create_interactive_visualization
            
            # 创建高级可视化窗口
            viz_window = create_interactive_visualization(self.root, "轨道摄动分析")
            
            # 设置数据
            viz_window.set_data('states', all_states)  # 默认全摄动状态
            viz_window.set_data('times', t_all)
            viz_window.set_data('ref_states', ref_states)
            viz_window.set_data('j2_states', j2_states)
            viz_window.set_data('drag_states', drag_states)
            viz_window.set_data('srp_states', srp_states)
            
            # 显示摄动分析图
            viz_window.viz_type_var.set("摄动分析")
            viz_window.plot_perturbation_analysis()
            
            # 记录日志
            self.log("完成摄动分析，显示结果")
            
        except Exception as e:
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示错误信息
            tk.messagebox.showerror("错误", f"摄动分析过程中出错:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
            # 记录日志
            self.log(f"摄动分析出错: {str(e)}")
    
    def show_collision_analysis(self):
        """增强版碰撞风险分析"""
        if self.constellation_states is None:
            messagebox.showwarning("警告", "请先创建星座!")
            return
        
        # 创建高级分析窗口
        if ui_extensions_available:
            analysis_window = AdvancedAnalysisWindow(self.root, "碰撞风险分析")
            
            # 计算碰撞风险
            self.log("计算碰撞风险...")
            cov_matrices = [np.eye(6) * 100 for _ in range(len(self.constellation_states))]
            
            avg_risk, close_approaches, collision_pairs = self.collision_analyzer.calculate_collision_probability_clustered(
                self.constellation_states, cov_matrices)
            
            self.log(f"星座碰撞风险: {avg_risk:.6f}")
            self.log(f"潜在碰撞对数: {close_approaches}")
            
            # 添加风险数据到窗口
            if close_approaches > 0:
                # 排序碰撞对
                collision_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # 创建数据框
                risk_data = pd.DataFrame({
                    "卫星i": [pair[0][0] for pair, risk in collision_pairs],
                    "卫星j": [pair[0][1] for pair, risk in collision_pairs],
                    "碰撞概率": [risk for pair, risk in collision_pairs],
                    "最小距离(km)": [np.linalg.norm(self.constellation_states[pair[0][0]][:3] - 
                                           self.constellation_states[pair[0][1]][:3])/1000 
                                 for pair, risk in collision_pairs]
                })
                
                # 在窗口中显示数据
                analysis_window.create_collision_risk_table(risk_data)
                
                # 绘制碰撞风险热图
                risk_matrix = np.zeros((len(self.constellation_states), len(self.constellation_states)))
                for (i, j), risk in collision_pairs:
                    risk_matrix[i, j] = risk
                    risk_matrix[j, i] = risk
                
                analysis_window.create_collision_risk_heatmap(risk_matrix)
                
                # 显示规避建议
                if avg_risk > 0.01:
                    # 获取最高风险的对
                    high_risk_pair, high_risk = collision_pairs[0]
                    sat_i, sat_j = high_risk_pair
                    
                    # 设计规避机动
                    delta_v, dist_before, dist_after = self.collision_analyzer.design_avoidance_maneuver(
                        self.constellation_states[sat_i], self.constellation_states[sat_j], 0.1)
                    
                    # 显示规避建议
                    analysis_window.display_avoidance_recommendation(
                        sat_i, sat_j, high_risk, dist_before, dist_after, delta_v)
            else:
                analysis_window.display_message("未检测到潜在碰撞风险")
        else:
            # 如果UI扩展不可用，使用简单的消息框
            cov_matrices = [np.eye(6) * 100 for _ in range(len(self.constellation_states))]
            avg_risk, close_approaches, _ = self.collision_analyzer.calculate_collision_probability_clustered(
                self.constellation_states, cov_matrices)
            messagebox.showinfo("碰撞风险分析", f"星座碰撞风险: {avg_risk:.6f}\n潜在碰撞对数: {close_approaches}")
    
    def show_coverage_analysis(self):
        """显示覆盖率分析"""
        messagebox.showinfo("功能提示", "覆盖率分析功能正在开发中")
    
    def show_error_analysis(self):
        """执行误差分析并显示结果"""
        # 检查是否有初始状态
        if not hasattr(self, 'initial_state') or self.initial_state is None:
            tk.messagebox.showwarning("警告", "没有初始状态数据，请先设置轨道参数")
            return
        
        # 创建误差分析窗口
        error_window = tk.Toplevel(self.root)
        error_window.title("误差分析配置")
        error_window.geometry("600x400")
        
        # 添加参数设置框架
        param_frame = ttk.LabelFrame(error_window, text="分析参数")
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 仿真时间
        ttk.Label(param_frame, text="仿真时间(天):").grid(row=0, column=0, padx=5, pady=5)
        duration_var = tk.DoubleVar(value=7.0)
        ttk.Entry(param_frame, textvariable=duration_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 要比较的方法
        ttk.Label(param_frame, text="比较方法:").grid(row=1, column=0, padx=5, pady=5)
        
        # 方法选择框架
        methods_frame = ttk.Frame(param_frame)
        methods_frame.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        # 方法复选框变量
        rk45_var = tk.BooleanVar(value=True)
        rk4_var = tk.BooleanVar(value=True)
        verlet_var = tk.BooleanVar(value=True)
        symplectic_var = tk.BooleanVar(value=True)
        euler_var = tk.BooleanVar(value=True)
        
        # 添加复选框
        ttk.Checkbutton(methods_frame, text="RK45", variable=rk45_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="RK4", variable=rk4_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="Verlet", variable=verlet_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="Symplectic Euler", variable=symplectic_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="欧拉法", variable=euler_var).pack(side=tk.LEFT, padx=5)
        
        # 步长设置
        ttk.Label(param_frame, text="步长(秒):").grid(row=2, column=0, padx=5, pady=5)
        step_var = tk.DoubleVar(value=60.0)
        ttk.Entry(param_frame, textvariable=step_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 摄动设置
        ttk.Label(param_frame, text="启用摄动:").grid(row=3, column=0, padx=5, pady=5)
        
        # 摄动复选框框架
        pert_frame = ttk.Frame(param_frame)
        pert_frame.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        # 摄动复选框变量
        j2_var = tk.BooleanVar(value=True)
        drag_var = tk.BooleanVar(value=False)
        srp_var = tk.BooleanVar(value=False)
        
        # 添加复选框
        ttk.Checkbutton(pert_frame, text="J2", variable=j2_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(pert_frame, text="大气阻力", variable=drag_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(pert_frame, text="太阳辐射压", variable=srp_var).pack(side=tk.LEFT, padx=5)
        
        # 分析类型
        ttk.Label(param_frame, text="分析类型:").grid(row=4, column=0, padx=5, pady=5)
        
        # 分析类型选择框架
        analysis_frame = ttk.Frame(param_frame)
        analysis_frame.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        # 分析类型复选框变量
        accuracy_var = tk.BooleanVar(value=True)
        energy_var = tk.BooleanVar(value=True)
        momentum_var = tk.BooleanVar(value=True)
        
        # 添加复选框
        ttk.Checkbutton(analysis_frame, text="精度分析", variable=accuracy_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(analysis_frame, text="能量误差", variable=energy_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(analysis_frame, text="角动量误差", variable=momentum_var).pack(side=tk.LEFT, padx=5)
        
        # 添加结果文本区域
        result_frame = ttk.LabelFrame(error_window, text="分析结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_text = tk.Text(result_frame, wrap=tk.WORD, width=80, height=10)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.configure(yscrollcommand=scrollbar.set)
        
        # 添加运行按钮和进度条
        control_frame = ttk.Frame(error_window)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(control_frame, variable=progress_var, maximum=100, length=400)
        progress_bar.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 定义运行函数
        def run_error_analysis():
            # 获取参数
            t_span = (0, duration_var.get() * 86400)  # 转换为秒
            step_size = step_var.get()
            
            # 确定要比较的方法
            methods_to_compare = []
            if rk45_var.get():
                methods_to_compare.append(("RK45", None, None))  # 特殊处理
            if rk4_var.get():
                from numerical_methods import NumericalIntegrators
                steps = int((t_span[1] - t_span[0]) / step_size)
                methods_to_compare.append(("RK4", NumericalIntegrators.rk4_method, steps))
            if verlet_var.get():
                from numerical_methods import NumericalIntegrators
                steps = int((t_span[1] - t_span[0]) / step_size)
                methods_to_compare.append(("Verlet", NumericalIntegrators.verlet_method, steps))
            if symplectic_var.get():
                from numerical_methods import NumericalIntegrators
                steps = int((t_span[1] - t_span[0]) / step_size)
                methods_to_compare.append(("Symplectic Euler", NumericalIntegrators.symplectic_euler, steps))
            if euler_var.get():
                from numerical_methods import NumericalIntegrators
                steps = int((t_span[1] - t_span[0]) / step_size)
                methods_to_compare.append(("Euler", NumericalIntegrators.euler_method, steps))
            
            if not methods_to_compare:
                tk.messagebox.showwarning("警告", "请至少选择一种数值方法进行比较")
                return
            
            # 清空结果文本
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "开始运行误差分析...\n")
            error_window.update()
            
            try:
                # 准备参数
                enable_j2 = j2_var.get()
                enable_drag = drag_var.get()
                enable_srp = srp_var.get()
                
                # 更新进度条
                progress_var.set(10)
                error_window.update()
                
                # 定义微分方程函数
                def func(t, y):
                    return self.propagator.orbit_derivatives(
                        t, y, enable_j2=enable_j2, enable_drag=enable_drag, enable_srp=enable_srp
                    )
                
                # 导入误差分析模块
                from numerical_methods import ErrorAnalysis
                
                # 进行误差分析
                result_text.insert(tk.END, "计算参考解...\n")
                error_window.update()
                
                # 更新进度条
                progress_var.set(30)
                error_window.update()
                
                # 特殊处理RK45方法
                import time
                from scipy.integrate import solve_ivp
                
                results = {}
                
                # 计算参考解 (使用更高精度的RK45)
                start_time = time.time()
                sol = solve_ivp(func, t_span, self.initial_state, method='RK45', 
                             rtol=1e-12, atol=1e-12, dense_output=True)
                ref_time = time.time() - start_time
                
                # 存储参考解
                results["reference"] = {
                    "name": "RK45 高精度参考解",
                    "time": ref_time,
                    "solution": sol.y.T,
                    "time_points": sol.t,
                    "error": 0.0
                }
                
                # 对每个方法进行计算
                for i, (name, method_func, steps) in enumerate(methods_to_compare):
                    # 更新进度条
                    progress_var.set(30 + 60 * (i+1) / len(methods_to_compare))
                    result_text.insert(tk.END, f"计算方法 {name}...\n")
                    error_window.update()
                    
                    if name == "RK45":
                        # 使用solve_ivp
                        start_time = time.time()
                        sol = solve_ivp(func, t_span, self.initial_state, method='RK45', 
                                      rtol=1e-8, atol=1e-8, dense_output=True)
                        elapsed = time.time() - start_time
                        
                        # 计算相对误差
                        error = ErrorAnalysis.relative_error(results["reference"]["solution"], sol.y.T)
                        
                        results[name] = {
                            "name": name,
                            "time": elapsed,
                            "solution": sol.y.T,
                            "time_points": sol.t,
                            "error": error
                        }
                    else:
                        # 使用其他方法
                        start_time = time.time()
                        t, y = method_func(func, t_span, self.initial_state, steps)
                        elapsed = time.time() - start_time
                        
                        # 计算相对误差
                        error = ErrorAnalysis.relative_error(results["reference"]["solution"], y)
                        
                        results[name] = {
                            "name": name,
                            "time": elapsed,
                            "solution": y,
                            "time_points": t,
                            "error": error
                        }
                
                # 更新进度条
                progress_var.set(95)
                result_text.insert(tk.END, "分析完成，显示结果...\n\n")
                error_window.update()
                
                # 显示分析结果
                result_text.insert(tk.END, "========== 精度与性能比较 ==========\n\n")
                result_text.insert(tk.END, f"{'方法':<15} {'计算时间(秒)':<15} {'相对误差':<15}\n")
                result_text.insert(tk.END, "-" * 45 + "\n")
                
                for name, data in results.items():
                    if name != "reference":
                        result_text.insert(tk.END, f"{name:<15} {data['time']:<15.6f} {data['error']:<15.6e}\n")
                
                # 如果需要能量误差分析
                if energy_var.get():
                    result_text.insert(tk.END, "\n========== 能量守恒误差 ==========\n\n")
                    for name, data in results.items():
                        if name != "reference":
                            # 计算能量误差
                            energy_error = ErrorAnalysis.energy_conservation_error(
                                data['solution'], 3.986004418e14
                            )
                            max_energy_error = np.max(np.abs(energy_error))
                            mean_energy_error = np.mean(np.abs(energy_error))
                            
                            result_text.insert(tk.END, f"{name} 能量误差:\n")
                            result_text.insert(tk.END, f"  最大误差: {max_energy_error:.6e}\n")
                            result_text.insert(tk.END, f"  平均误差: {mean_energy_error:.6e}\n\n")
                
                # 如果需要角动量误差分析
                if momentum_var.get():
                    result_text.insert(tk.END, "\n========== 角动量守恒误差 ==========\n\n")
                    for name, data in results.items():
                        if name != "reference":
                            # 计算角动量误差
                            momentum_error = ErrorAnalysis.angular_momentum_error(data['solution'])
                            max_momentum_error = np.max(np.abs(momentum_error))
                            mean_momentum_error = np.mean(np.abs(momentum_error))
                            
                            result_text.insert(tk.END, f"{name} 角动量误差:\n")
                            result_text.insert(tk.END, f"  最大误差: {max_momentum_error:.6e}\n")
                            result_text.insert(tk.END, f"  平均误差: {mean_momentum_error:.6e}\n\n")
                
                # 更新进度条
                progress_var.set(100)
                error_window.update()
                
                # 使用高级可视化模块显示结果
                from advanced_visualization import create_interactive_visualization
                
                # 创建可视化窗口
                viz_window = create_interactive_visualization(self.root, "误差分析可视化")
                
                # 设置数据
                viz_window.set_data('error_results', results)
                
                # 显示误差分析图
                viz_window.viz_type_var.set("误差分析")
                viz_window.plot_error_analysis()
                
                # 记录日志
                self.log("完成误差分析，显示结果")
                
            except Exception as e:
                # 显示错误
                result_text.insert(tk.END, f"\n错误: {str(e)}\n")
                import traceback
                traceback.print_exc()
                # 记录日志
                self.log(f"误差分析出错: {str(e)}")
        
        # 添加运行按钮
        run_button = ttk.Button(control_frame, text="运行分析", command=run_error_analysis)
        run_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 记录日志
        self.log("打开误差分析窗口")
    
    def save_constellation(self):
        """保存当前星座配置"""
        if self.constellation_states is None:
            messagebox.showwarning("警告", "没有星座可保存!")
            return
        
        # 保存星座状态到文件
        try:
            np.save("results/constellation_states.npy", self.constellation_states)
            
            # 保存参数
            params = {
                "total_satellites": self.total_satellites_var.get(),
                "num_planes": self.num_planes_var.get(),
                "phase_factor": self.phase_factor_var.get(),
                "altitude": self.altitude_var.get(),
                "inclination": self.inclination_var.get(),
                "eccentricity": self.eccentricity_var.get(),
                "am_ratio": self.am_ratio_var.get(),
                "cd": self.cd_var.get(),
                "cr": self.cr_var.get()
            }
            np.save("results/constellation_params.npy", params)
            
            self.log("星座配置已保存到results目录")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存星座时出错: {str(e)}")
    
    def load_constellation(self):
        """加载保存的星座配置"""
        try:
            self.constellation_states = np.load("results/constellation_states.npy")
            params = np.load("results/constellation_params.npy", allow_pickle=True).item()
            
            # 设置参数值
            self.total_satellites_var.set(params["total_satellites"])
            self.num_planes_var.set(params["num_planes"])
            self.phase_factor_var.set(params["phase_factor"])
            self.altitude_var.set(params["altitude"])
            self.inclination_var.set(params["inclination"])
            self.eccentricity_var.set(params["eccentricity"])
            self.am_ratio_var.set(params["am_ratio"])
            self.cd_var.set(params["cd"])
            self.cr_var.set(params["cr"])
            
            self.log("星座配置已加载")
            
            # 绘制星座
            self.plot_constellation(self.constellation_states)
        except Exception as e:
            messagebox.showerror("加载错误", f"加载星座时出错: {str(e)}")
    
    def export_data(self):
        """导出仿真数据"""
        if self.propagated_states is None:
            messagebox.showwarning("警告", "没有数据可导出!")
            return
        
        try:
            # 导出第一颗卫星的数据作为示例
            states = self.propagated_states[0]
            times = self.propagation_times[0]
            
            # 创建数据框
            data = {
                "Time(s)": times,
                "X(m)": states[:, 0],
                "Y(m)": states[:, 1],
                "Z(m)": states[:, 2],
                "VX(m/s)": states[:, 3],
                "VY(m/s)": states[:, 4],
                "VZ(m/s)": states[:, 5]
            }
            
            # 添加轨道根数
            kepler_elements = np.array([self.propagator.cartesian_to_kepler(state) for state in states])
            data["SemiMajorAxis(m)"] = kepler_elements[:, 0]
            data["Eccentricity"] = kepler_elements[:, 1]
            data["Inclination(rad)"] = kepler_elements[:, 2]
            data["RAAN(rad)"] = kepler_elements[:, 3]
            data["ArgOfPerigee(rad)"] = kepler_elements[:, 4]
            data["MeanAnomaly(rad)"] = kepler_elements[:, 5]
            
            # 创建数据框并导出
            df = pd.DataFrame(data)
            df.to_csv("results/satellite_data.csv", index=False)
            
            # 如果有误差分析数据，也导出
            if hasattr(self, 'energy_errors') and self.energy_errors:
                error_data = {
                    "Time(s)": times,
                    "EnergyError": self.energy_errors[0],
                    "MomentumError": self.momentum_errors[0]
                }
                error_df = pd.DataFrame(error_data)
                error_df.to_csv("results/error_analysis.csv", index=False)
            
            self.log("数据已导出到results目录")
        except Exception as e:
            messagebox.showerror("导出错误", f"导出数据时出错: {str(e)}")
    
    def show_settings(self):
        """显示设置对话框"""
        messagebox.showinfo("功能提示", "设置功能正在开发中")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        近地星座构型演化仿真系统使用指南:
        
        1. 创建星座: 设置卫星数量、轨道面数、轨道参数后点击"创建星座"按钮
        2. 运行仿真: 设置仿真参数后点击"运行仿真"按钮
        3. 分析功能: 使用"分析"菜单下的各种工具分析仿真结果
        4. 高级功能: 使用"工具"菜单下的功能进行高级分析和优化
        
        详细使用指南请参阅项目文档。
        """
        messagebox.showinfo("使用指南", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
        近地星座构型演化仿真系统 v2.0
        
        本系统用于模拟和分析近地卫星星座的轨道演化特性，
        包含多种摄动力模型、数值计算方法和分析工具。
        
        作者: 基于原始程序改进
        """
        messagebox.showinfo("关于", about_text)


def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置UI字体支持中文，增大字体大小
    try:
        # 中文字体设置
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei", size=12)  # 增大字体
        
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family="Microsoft YaHei", size=12)  # 增大字体
        
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family="Microsoft YaHei", size=12)  # 增大字体
        
        # 设置全局字体大小
        root.option_add("*Font", default_font)
    except Exception as e:
        print(f"警告: 字体设置失败 - {e}")
    
    # 启动应用程序
    app = EnhancedConstellationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 