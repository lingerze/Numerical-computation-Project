"""
近地星座构型演化仿真系统 (Enhanced version)

本程序提供近地卫星星座的构型设计、轨道演化仿真和性能分析功能。
主要功能包括：
1. Walker星座设计与参数配置
2. 高精度轨道传播（含J2、大气阻力和太阳辐射压摄动）
3. 星座覆盖率、碰撞风险分析
4. 轨道演化可视化与3D动画
5. 星座优化功能（根据覆盖率、碰撞风险等目标自动搜索最优参数）
6. 多种数值积分方法对比

作者:Hongjin Lin、Zilin LI

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
import datetime
import time
import pandas as pd
import seaborn as sns

# 配置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12                       # 设置字体大小

try:
    from numerical_methods import NumericalIntegrators, ErrorAnalysis
    numerical_methods_available = True
except ImportError:
    numerical_methods_available = False

class OrbitPropagator:
    """轨道传播器，用于模拟卫星轨道演化"""
    
    def __init__(self):
        # 常数
        self.MU = 3.986004418e14  # 地球引力常数 (m^3/s^2)
        self.RE = 6378.137e3      # 地球半径 (m)
        self.J2 = 1.08263e-3      # J2 系数
        self.SOLAR_P = 4.56e-6    # 太阳辐射压力 (N/m^2)
        self.DAY_SEC = 86400      # 一天的秒数
        self.earth_rotation_rate = 7.2921159e-5  # 地球自转角速度 (rad/s)
        # 时间测量变量
        self.last_propagation_time = 0
        
    def kepler_to_cartesian(self, kepler_elements):
        """开普勒轨道要素转笛卡尔坐标"""
        a, e, i, RAAN, w, M = kepler_elements
        
        # 偏近点角计算（牛顿法求解开普勒方程）
        E = M  # 初始猜测
        for _ in range(10):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        
        # 真近点角
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        
        # 轨道平面上的位置
        r = a * (1 - e * np.cos(E))
        x_orbit = r * np.cos(nu)
        y_orbit = r * np.sin(nu)
        
        # 旋转矩阵
        R3_RAAN = np.array([
            [np.cos(RAAN), -np.sin(RAAN), 0],
            [np.sin(RAAN), np.cos(RAAN), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w), np.cos(w), 0],
            [0, 0, 1]
        ])
        
        # 合成旋转矩阵
        R = R3_RAAN @ R1_i @ R3_w
        
        # 位置向量
        r_vec = R @ np.array([x_orbit, y_orbit, 0])
        
        # 速度计算
        n = np.sqrt(self.MU / a**3)  # 平均运动
        p = a * (1 - e**2)
        
        vx_orbit = -n * a**2 / r * np.sin(E)
        vy_orbit = n * a**2 / r * np.sqrt(1 - e**2) * np.cos(E)
        
        v_vec = R @ np.array([vx_orbit, vy_orbit, 0])
        
        return np.concatenate((r_vec, v_vec))
    
    def cartesian_to_kepler(self, state_vector):
        """笛卡尔坐标转开普勒轨道要素"""
        r_vec = state_vector[:3]
        v_vec = state_vector[3:]
        
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # 角动量矢量
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        
        # 节点线
        n_vec = np.cross(np.array([0, 0, 1]), h_vec)
        n = np.linalg.norm(n_vec)
        
        # 偏心率矢量
        e_vec = np.cross(v_vec, h_vec) / self.MU - r_vec / r
        e = np.linalg.norm(e_vec)
        
        # 半长轴
        a = h**2 / (self.MU * (1 - e**2))
        
        # 轨道倾角
        i = np.arccos(h_vec[2] / h)
        
        # 升交点赤经
        if n > 1e-10:
            RAAN = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                RAAN = 2 * np.pi - RAAN
        else:
            RAAN = 0
        
        # 近地点幅角
        if n > 1e-10:
            w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = np.arccos(e_vec[0] / e)
            if e_vec[1] < 0:
                w = 2 * np.pi - w
        
        # 真近点角
        nu = np.arccos(np.dot(e_vec, r_vec) / (e * r))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
        
        # 偏近点角
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        
        # 平近点角
        M = E - e * np.sin(E)
        
        return np.array([a, e, i, RAAN, w, M])
    
    def J2_acceleration(self, r_vec):
        """计算J2摄动加速度"""
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)
        
        # J2扰动
        factor = 1.5 * self.J2 * self.MU * self.RE**2 / r**5
        
        ax = factor * x * (5 * z**2 / r**2 - 1)
        ay = factor * y * (5 * z**2 / r**2 - 1)
        az = factor * z * (5 * z**2 / r**2 - 3)
        
        return np.array([ax, ay, az])
    
    def atmospheric_density(self, h):
        """简化的大气密度模型，输入高度(km)，返回密度(kg/m^3)"""
        # 基于NRLMSISE-00模型的简化版本
        h_km = h / 1000.0  # 转换为km
        
        # 高度分段的参考高度和密度
        h_ref = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])
        rho_ref = np.array([2.7e-10, 1.8e-11, 2.1e-12, 3.1e-13, 6.2e-14, 
                           1.6e-14, 5.2e-15, 2.2e-15, 1.0e-15])
        
        # 改进插值方法，使用numpy的插值函数而不是循环
        if h_km <= h_ref[0]:
            return rho_ref[0]
        elif h_km >= h_ref[-1]:
            return rho_ref[-1]
        else:
            # 对数线性插值
            log_rho_ref = np.log(rho_ref)
            log_rho = np.interp(h_km, h_ref, log_rho_ref)
            return np.exp(log_rho)
    
    def drag_acceleration(self, r_vec, v_vec, A_m_ratio, cd=2.2):
        """计算大气阻力加速度"""
        r = np.linalg.norm(r_vec)
        h = r - self.RE  # 高度
        
        # 计算大气密度
        rho = self.atmospheric_density(h)
        
        # 计算地球自转速度
        omega_earth = np.array([0, 0, self.earth_rotation_rate])
        v_atm = np.cross(omega_earth, r_vec)  # 大气速度
        v_rel = v_vec - v_atm  # 相对速度
        v_rel_mag = np.linalg.norm(v_rel)
        
        # 阻力加速度
        a_drag = -0.5 * rho * v_rel_mag * cd * A_m_ratio * v_rel
        
        return a_drag
    
    def srp_acceleration(self, r_vec, A_m_ratio, cr=1.5, t=0):
        """计算太阳辐射压力加速度，添加时间参数来改进太阳方向模型"""
        # 改进的太阳方向模型，考虑地球绕太阳公转
        earth_orbital_rate = 2 * np.pi / (365.25 * self.DAY_SEC)  # 地球公转角速度 (rad/s)
        sun_direction = np.array([
            np.cos(earth_orbital_rate * t),
            np.sin(earth_orbital_rate * t),
            0.0
        ])
        sun_direction = sun_direction / np.linalg.norm(sun_direction)
        
        # 检查地球阴影
        r = np.linalg.norm(r_vec)
        cos_angle = np.dot(r_vec, sun_direction) / r
        
        # 使用圆柱阴影模型
        in_shadow = cos_angle < 0 and np.sqrt(r**2 - cos_angle**2 * r**2) < self.RE
        
        # 如果卫星在阴影中，没有辐射压力
        if in_shadow:
            return np.zeros(3)
        
        # 辐射压力加速度
        a_srp = -self.SOLAR_P * cr * A_m_ratio * sun_direction
        
        return a_srp
    
    def orbit_derivatives(self, t, state, A_m_ratio=0.01, cd=2.2, cr=1.5, 
                          enable_j2=True, enable_drag=True, enable_srp=True):
        """轨道动力学微分方程，增加了摄动开关"""
        r_vec = state[:3]
        v_vec = state[3:]
        
        r = np.linalg.norm(r_vec)
        
        # 二体运动
        a_gravity = -self.MU * r_vec / r**3
        
        # 总加速度初始化为二体项
        a_total = a_gravity
        
        # J2摄动
        if enable_j2:
            a_J2 = self.J2_acceleration(r_vec)
            a_total += a_J2
        
        # 大气阻力
        if enable_drag:
            a_drag = self.drag_acceleration(r_vec, v_vec, A_m_ratio, cd)
            a_total += a_drag
        
        # 太阳辐射压力
        if enable_srp:
            a_srp = self.srp_acceleration(r_vec, A_m_ratio, cr, t)
            a_total += a_srp
        
        return np.concatenate((v_vec, a_total))
    
    def propagate_orbit(self, initial_state, t_span, A_m_ratio=0.01, cd=2.2, cr=1.5,
                         method='RK45', steps=1000, rtol=1e-8, atol=1e-8,
                         enable_j2=True, enable_drag=True, enable_srp=True):
        """传播卫星轨道，支持多种数值方法"""
        
        start_time = time.time()
        
        # 创建函数参数闭包
        def func(t, y):
            return self.orbit_derivatives(t, y, A_m_ratio, cd, cr, enable_j2, enable_drag, enable_srp)
        
        # 根据选择的方法进行轨道传播
        if numerical_methods_available:
            if method == 'RK45':
                times, states, nfev = NumericalIntegrators.adaptive_rk45(
                    func, t_span, initial_state, rtol=rtol, atol=atol
                )
            elif method == 'RK4':
                times, states = NumericalIntegrators.rk4_method(
                    func, t_span, initial_state, steps
                )
            elif method == 'Euler':
                times, states = NumericalIntegrators.euler_method(
                    func, t_span, initial_state, steps
                )
            elif method == 'Verlet':
                times, states = NumericalIntegrators.verlet_method(
                    func, t_span, initial_state, steps
                )
            elif method == 'Symplectic':
                times, states = NumericalIntegrators.symplectic_euler(
                    func, t_span, initial_state, steps
                )
            else:
                # 默认回退到SciPy的solve_ivp
                sol = solve_ivp(
                    func,
                    t_span,
                    initial_state,
                    method='RK45',
                    rtol=rtol,
                    atol=atol
                )
                times = sol.t
                states = sol.y.T
        else:
            # 如果没有导入数值方法模块，使用SciPy
            sol = solve_ivp(
                func,
                t_span,
                initial_state,
                method='RK45',
                rtol=rtol,
                atol=atol
            )
            times = sol.t
            states = sol.y.T
        
        # 记录计算时间
        self.last_propagation_time = time.time() - start_time
        
        return times, states
    
    def compare_integration_methods(self, initial_state, t_span, methods=None, 
                                    A_m_ratio=0.01, cd=2.2, cr=1.5,
                                    enable_j2=True, enable_drag=True, enable_srp=True):
        """比较不同积分方法的精度和性能"""
        if not numerical_methods_available:
            return None, "数值方法模块不可用，请确保numerical_methods.py文件存在"
        
        if methods is None:
            methods = [
                ("RK45", NumericalIntegrators.adaptive_rk45, None),
                ("RK4", NumericalIntegrators.rk4_method, 1000),
                ("Verlet", NumericalIntegrators.verlet_method, 1000),
                ("Symplectic", NumericalIntegrators.symplectic_euler, 1000),
                ("Euler", NumericalIntegrators.euler_method, 1000)
            ]
        
        # 创建函数参数闭包
        def func(t, y):
            return self.orbit_derivatives(t, y, A_m_ratio, cd, cr, enable_j2, enable_drag, enable_srp)
        
        # 比较方法
        results = ErrorAnalysis.compare_methods(
            func, t_span, initial_state, 
            [(name, method, steps) for name, method, steps in methods if steps is not None],
            extra_params={"adaptive_params": {"rtol": 1e-10, "atol": 1e-13}}
        )
        
        # 生成比较报告
        report = "积分方法比较报告:\n"
        report += "-" * 60 + "\n"
        report += f"{'方法':<15} {'计算时间(秒)':<15} {'相对误差':<15} {'步数':<10}\n"
        report += "-" * 60 + "\n"
        
        for name, data in results.items():
            if name != "reference":
                report += f"{name:<15} {data['time']:<15.6f} {data['error']:<15.6e} {data.get('steps', 'N/A'):<10}\n"
        
        report += f"参考方法: {results['reference']['name']}, 计算时间: {results['reference']['time']:.6f}秒\n"
        report += "-" * 60 + "\n"
        
        # 分析轨道保守量
        report += "\n轨道保守量分析:\n"
        report += "-" * 60 + "\n"
        
        for name, data in results.items():
            if name != "reference":
                # 能量守恒误差
                energy_error = ErrorAnalysis.energy_conservation_error(data['solution'], self.MU)
                # 角动量守恒误差
                momentum_error = ErrorAnalysis.angular_momentum_error(data['solution'])
                
                report += f"{name} 方法:\n"
                report += f"  能量最大相对误差: {np.max(np.abs(energy_error)):.6e}\n"
                report += f"  角动量最大相对误差: {np.max(np.abs(momentum_error)):.6e}\n"
        
        return results, report
    
    def mean_orbital_elements(self, state_history):
        """计算平均轨道根数"""
        kepler_elements = []
        for state in state_history:
            kepler_elements.append(self.cartesian_to_kepler(state))
        
        return np.mean(kepler_elements, axis=0)
    
    def calculate_orbit_stability(self, state_history):
        """计算轨道稳定性指标"""
        # 提取开普勒要素历史
        kepler_history = np.array([self.cartesian_to_kepler(state) for state in state_history])
        
        # 计算各要素的标准差
        std_elements = np.std(kepler_history, axis=0)
        
        # 计算相对变化
        mean_elements = np.mean(kepler_history, axis=0)
        rel_variation = std_elements / np.abs(mean_elements)
        
        # 轨道要素名称
        element_names = ['Semi-major axis', 'Eccentricity', 'Inclination', 
                         'RAAN', 'Argument of Perigee', 'Mean Anomaly']
        
        # 创建结果字典
        stability = {element_names[i]: {
            'mean': mean_elements[i],
            'std': std_elements[i],
            'rel_var': rel_variation[i]
        } for i in range(6)}
        
        return stability


class ConstellationDesigner:
    """星座设计与优化"""
    
    def __init__(self, propagator):
        self.propagator = propagator
    
    def create_walker_constellation(self, total_satellites, num_planes, relative_spacing, 
                                   inclination, altitude, eccentricity=0.0):
        """创建Walker星座配置"""
        satellites_per_plane = total_satellites // num_planes
        
        # 计算轨道参数
        a = altitude + self.propagator.RE  # 半长轴
        
        # 初始化卫星状态
        constellation_states = []
        
        for p in range(num_planes):
            # 每个轨道面的升交点赤经
            RAAN = 2 * np.pi * p / num_planes
            
            for s in range(satellites_per_plane):
                # 计算卫星在轨道上的相位角
                M = 2 * np.pi * s / satellites_per_plane
                
                # 应用相位差
                phase_diff = 2 * np.pi * relative_spacing * p / total_satellites
                M = (M + phase_diff) % (2 * np.pi)
                
                # 轨道根数
                kepler_elements = np.array([a, eccentricity, inclination, RAAN, 0.0, M])
                
                # 转换为笛卡尔坐标
                state = self.propagator.kepler_to_cartesian(kepler_elements)
                constellation_states.append(state)
        
        return np.array(constellation_states)
    
    def calculate_coverage(self, constellation_states, elevation_mask=10.0):
        """计算星座的覆盖率"""
        # 这是一个简化的覆盖率计算，实际应用中需要更复杂的模型
        earth_radius = self.propagator.RE
        
        # 创建全球网格
        lat_grid = np.linspace(-90, 90, 18)  # 10度一格
        lon_grid = np.linspace(-180, 180, 36)  # 10度一格
        
        total_points = len(lat_grid) * len(lon_grid)
        covered_points = 0
        
        for lat in lat_grid:
            for lon in lon_grid:
                # 转换网格点到ECEF坐标
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                
                x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = earth_radius * np.sin(lat_rad)
                
                ground_point = np.array([x, y, z])
                
                # 检查任一卫星是否覆盖该点
                for state in constellation_states:
                    sat_pos = state[:3]
                    
                    # 计算卫星到地面点的方向向量
                    sat_to_ground = ground_point - sat_pos
                    distance = np.linalg.norm(sat_to_ground)
                    
                    # 计算高度角
                    cos_elevation = np.dot(ground_point, sat_to_ground) / (np.linalg.norm(ground_point) * distance)
                    elevation = 90 - np.degrees(np.arccos(cos_elevation))
                    
                    if elevation >= elevation_mask:
                        covered_points += 1
                        break
        
        coverage_ratio = covered_points / total_points
        return coverage_ratio
    
    def calculate_collision_risk(self, constellation_states, safe_distance=1000.0):
        """计算星座内碰撞风险指标"""
        num_satellites = len(constellation_states)
        total_pairs = num_satellites * (num_satellites - 1) // 2
        
        risk_sum = 0.0
        close_approaches = 0
        
        for i in range(num_satellites):
            for j in range(i+1, num_satellites):
                pos_i = constellation_states[i][:3]
                pos_j = constellation_states[j][:3]
                
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < safe_distance:
                    close_approaches += 1
                
                # 风险度量：安全距离/实际距离（越大风险越高）
                if distance > 0:
                    risk = safe_distance / distance
                    risk_sum += min(risk, 1.0)  # 限制最大风险为1
        
        avg_risk = risk_sum / total_pairs if total_pairs > 0 else 0
        return avg_risk, close_approaches
    
    def evaluate_constellation(self, constellation_states, weights=None):
        """综合评估星座性能"""
        if weights is None:
            weights = {
                'coverage': 0.4,
                'collision_risk': 0.3,
                'energy': 0.3
            }
        
        # 计算覆盖率
        coverage = self.calculate_coverage(constellation_states)
        
        # 计算碰撞风险
        risk, close_approaches = self.calculate_collision_risk(constellation_states)
        
        # 计算能量效率（简化模型：轨道高度越低越好）
        avg_altitude = 0
        for state in constellation_states:
            r = np.linalg.norm(state[:3])
            altitude = r - self.propagator.RE
            avg_altitude += altitude
        
        avg_altitude /= len(constellation_states)
        norm_altitude = min(1.0, 2000e3 / avg_altitude)  # 归一化，假设2000km是最高高度
        
        # 计算综合评分
        score = (weights['coverage'] * coverage + 
                weights['energy'] * norm_altitude + 
                weights['collision_risk'] * (1 - risk))
        
        return score, {
            'coverage': coverage,
            'collision_risk': risk,
            'normalized_altitude': norm_altitude
        }
    
    def optimize_constellation(self, objective='coverage', total_satellites=None, num_planes=None, 
                              inclination=None, altitude=None, eccentricity=0.0, 
                              weights=None, max_iterations=10):
        """
        优化星座设计参数
        
        参数:
        - objective: 优化目标，可选 'coverage', 'collision_risk', 'multi_objective'
        - total_satellites: 可选的卫星总数范围，如(12, 48)
        - num_planes: 可选的轨道面数范围，如(3, 8)
        - inclination: 可选的倾角范围(弧度)，如(0.7, 1.5)
        - altitude: 可选的高度范围(米)，如(500e3, 1200e3)
        - eccentricity: 偏心率
        - weights: 多目标优化的权重
        - max_iterations: 最大优化迭代次数
        
        返回:
        - best_params: 最优参数
        - best_score: 最优得分
        - optimization_history: 优化历史
        """
        import numpy as np
        
        # 设置默认参数范围
        if total_satellites is None:
            total_satellites = (12, 48)
        if num_planes is None:
            num_planes = (3, 8)
        if inclination is None:
            inclination = (np.radians(30), np.radians(90))
        if altitude is None:
            altitude = (500e3, 1200e3)
        
        # 检查参数范围是否合法
        if total_satellites[0] > total_satellites[1]:
            self.log("警告: 卫星数范围错误，将交换最小值和最大值")
            total_satellites = (total_satellites[1], total_satellites[0])
        
        if num_planes[0] > num_planes[1]:
            self.log("警告: 轨道面数范围错误，将交换最小值和最大值")
            num_planes = (num_planes[1], num_planes[0])
        
        if inclination[0] > inclination[1]:
            self.log("警告: 倾角范围错误，将交换最小值和最大值")
            inclination = (inclination[1], inclination[0])
        
        if altitude[0] > altitude[1]:
            self.log("警告: 高度范围错误，将交换最小值和最大值")
            altitude = (altitude[1], altitude[0])
        
        # 设置权重
        if weights is None:
            weights = {'coverage': 0.6, 'collision_risk': 0.3, 'energy': 0.1}
        else:
            # 验证权重的完整性
            required_weights = ['coverage', 'collision_risk', 'energy']
            for w in required_weights:
                if w not in weights:
                    self.log(f"警告: 权重'{w}'未指定，使用默认值")
                    if w == 'coverage':
                        weights[w] = 0.6
                    elif w == 'collision_risk':
                        weights[w] = 0.3
                    else:
                        weights[w] = 0.1
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight != 1.0 and total_weight > 0:
                self.log(f"警告: 权重总和不为1.0，将自动归一化")
                for w in weights:
                    weights[w] /= total_weight
        
        best_score = 0
        best_params = None
        optimization_history = []
        
        # 初始参数采样
        total_sat_samples = np.linspace(total_satellites[0], total_satellites[1], 5, dtype=int)
        num_planes_samples = np.linspace(num_planes[0], num_planes[1], 4, dtype=int)
        inc_samples = np.linspace(inclination[0], inclination[1], 4)
        alt_samples = np.linspace(altitude[0], altitude[1], 4)
        
        # 随机组合参数进行初始搜索
        np.random.seed(42)  # 确保可重复性
        param_combinations = []
        
        # 确保总卫星数能被轨道面数整除
        for total_sats in total_sat_samples:
            for num_plane in num_planes_samples:
                # 调整卫星数使其能被轨道面数整除
                adjusted_total_sats = total_sats
                if total_sats % num_plane != 0:
                    # 向上取整到能被轨道面数整除的数
                    adjusted_total_sats = ((total_sats + num_plane - 1) // num_plane) * num_plane
                    if adjusted_total_sats > total_satellites[1]:
                        # 如果超出范围，则向下取整
                        adjusted_total_sats = (total_sats // num_plane) * num_plane
                    
                    if adjusted_total_sats == 0:
                        # 确保至少有num_plane个卫星
                        adjusted_total_sats = num_plane
                
                # 确保每个轨道面至少有2颗卫星
                if adjusted_total_sats // num_plane >= 2:
                    for inc in inc_samples:
                        for alt in alt_samples:
                            # 随机相位因子
                            rel_spacing = np.random.randint(0, min(num_plane, 3))
                            param_combinations.append((adjusted_total_sats, num_plane, rel_spacing, inc, alt))
        
        if not param_combinations:
            self.log("错误: 无法找到满足条件的参数组合，请调整参数范围")
            self.log(f"当前参数范围: 卫星数={total_satellites}, 轨道面数={num_planes}")
            self.log("建议: 增大卫星数上限或减小轨道面数下限")
            return None, 0, []
        
        self.log(f"生成了{len(param_combinations)}种参数组合进行优化搜索")
        
        # 评估每个参数组合
        for i, (total_sats, num_plane, rel_spacing, inc, alt) in enumerate(param_combinations):
            if i >= max_iterations:
                break
                
            try:
                # 创建星座
                self.log(f"评估参数组合 {i+1}/{min(len(param_combinations), max_iterations)}: "
                         f"卫星数={total_sats}, 轨道面={num_plane}, 倾角={np.degrees(inc):.1f}°, "
                         f"高度={alt/1000:.1f}km")
                
                constellation_states = self.create_walker_constellation(
                    total_sats, num_plane, rel_spacing, inc, alt, eccentricity)
                
                # 根据目标进行评估
                if objective == 'coverage':
                    score = self.calculate_coverage(constellation_states)
                elif objective == 'collision_risk':
                    risk, _ = self.calculate_collision_risk(constellation_states)
                    score = 1 - risk  # 转换为越高越好
                else:  # 多目标
                    score, metrics = self.evaluate_constellation(constellation_states, weights)
                
                # 记录结果
                params = {
                    'total_satellites': total_sats,
                    'num_planes': num_plane,
                    'relative_spacing': rel_spacing,
                    'inclination': inc,
                    'altitude': alt,
                    'eccentricity': eccentricity,
                    'score': score
                }
                
                optimization_history.append(params)
                
                # 更新最优解
                if score > best_score:
                    best_score = score
                    best_params = params
                    self.log(f"找到更好的解，得分={score:.4f}")
                else:
                    self.log(f"当前得分={score:.4f}")
            except Exception as e:
                self.log(f"评估参数组合时出错: {str(e)}")
                continue
        
        if best_params is None:
            self.log("警告: 未找到任何可行解!")
            return None, 0, optimization_history
            
        self.log(f"优化完成! 最优得分: {best_score:.4f}")
        self.log(f"最优参数: 卫星数={best_params['total_satellites']}, "
                f"轨道面={best_params['num_planes']}, "
                f"倾角={np.degrees(best_params['inclination']):.1f}°, "
                f"高度={best_params['altitude']/1000:.1f}km")
        
        return best_params, best_score, optimization_history
    
    def log(self, message):
        """日志输出，可由外部重定向"""
        print(f"[星座设计器] {message}")


class CollisionAnalyzer:
    """碰撞分析与规避策略"""
    
    def __init__(self, propagator):
        self.propagator = propagator
    
    def calculate_minimum_distance(self, state1, state2, t_span=(0, 86400), dt=60):
        """计算两颗卫星在一段时间内的最小距离"""
        times = np.arange(t_span[0], t_span[1], dt)
        
        min_distance = float('inf')
        min_time = None
        
        for t in times:
            # 传播两颗卫星的轨道
            _, states1 = self.propagator.propagate_orbit(state1, (0, t))
            _, states2 = self.propagator.propagate_orbit(state2, (0, t))
            
            pos1 = states1[-1][:3]
            pos2 = states2[-1][:3]
            
            distance = np.linalg.norm(pos1 - pos2)
            
            if distance < min_distance:
                min_distance = distance
                min_time = t
        
        return min_distance, min_time
    
    def calculate_collision_probability(self, state1, state2, cov1, cov2, safe_distance=1000.0, samples=1000):
        """使用蒙特卡洛法计算碰撞概率"""
        # 获取当前位置
        pos1 = state1[:3]
        pos2 = state2[:3]
        
        # 计算相对位置
        rel_pos = pos1 - pos2
        
        # 创建相对位置协方差矩阵
        rel_cov = cov1[:3, :3] + cov2[:3, :3]
        
        # 生成多元正态分布样本
        np.random.seed(42)  # 固定随机数种子以确保可重复性
        samples = np.random.multivariate_normal(rel_pos, rel_cov, samples)
        
        # 计算每个样本的距离
        distances = np.linalg.norm(samples, axis=1)
        
        # 计算碰撞概率
        collision_count = np.sum(distances < safe_distance)
        collision_probability = collision_count / len(distances)
        
        return collision_probability
    
    def design_avoidance_maneuver(self, state, threat_state, delta_v_max=1.0):
        """设计碰撞规避机动"""
        r = state[:3]
        v = state[3:]
        
        # 计算当前轨道面法向量
        h = np.cross(r, v)
        h_unit = h / np.linalg.norm(h)
        
        # 计算切向方向
        t_unit = np.cross(h_unit, r) / np.linalg.norm(r)
        
        # 计算径向方向
        r_unit = r / np.linalg.norm(r)
        
        # 默认使用法向机动（改变轨道面）
        delta_v = delta_v_max * h_unit
        
        # 检查机动后的效果
        new_v = v + delta_v
        new_state = np.concatenate((r, new_v))
        
        min_dist_before, _ = self.calculate_minimum_distance(state, threat_state)
        min_dist_after, _ = self.calculate_minimum_distance(new_state, threat_state)
        
        # 如果法向机动效果不好，尝试切向机动
        if min_dist_after < min_dist_before * 1.5:
            delta_v = delta_v_max * t_unit
            new_v = v + delta_v
            new_state = np.concatenate((r, new_v))
            min_dist_after, _ = self.calculate_minimum_distance(new_state, threat_state)
        
        # 如果切向机动效果还是不好，尝试径向机动
        if min_dist_after < min_dist_before * 1.5:
            delta_v = delta_v_max * r_unit
            new_v = v + delta_v
            new_state = np.concatenate((r, new_v))
            min_dist_after, _ = self.calculate_minimum_distance(new_state, threat_state)
        
        return delta_v, min_dist_before, min_dist_after


class ConstellationSimulationApp:
    """星座仿真应用程序"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("近地星座构型演化仿真系统")
        self.root.geometry("1200x800")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 创建参数设置区域
        self.create_parameter_widgets()
        
        # 创建右侧可视化区域
        self.visualization_frame = ttk.LabelFrame(self.main_frame, text="轨道可视化")
        self.visualization_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 设置图表
        self.setup_plot()
        
        # 初始化轨道传播器和星座设计器
        self.propagator = OrbitPropagator()
        self.designer = ConstellationDesigner(self.propagator)
        self.analyzer = CollisionAnalyzer(self.propagator)
        
        # 存储星座数据
        self.constellation_states = None
        self.propagated_states = None
        self.animation = None
        self.is_animating = False
        
        # 初始状态
        self.satellites = []
        self.earth = None
        self.trajectory_lines = []
        
        # 初始化日志区域
        self.create_log_area()
        
        # 创建进度条
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.control_frame, variable=self.progress_var, length=200)
        self.progress.pack(pady=10, padx=10)
        
    def create_parameter_widgets(self):
        """创建参数设置控件"""
        # 星座参数区域
        const_frame = ttk.LabelFrame(self.control_frame, text="星座参数")
        const_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 卫星数量
        ttk.Label(const_frame, text="卫星总数:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.total_satellites_var = tk.IntVar(value=12)
        ttk.Spinbox(const_frame, from_=6, to=60, textvariable=self.total_satellites_var, width=5).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 轨道面数量
        ttk.Label(const_frame, text="轨道面数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_planes_var = tk.IntVar(value=3)
        ttk.Spinbox(const_frame, from_=1, to=20, textvariable=self.num_planes_var, width=5).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 相位因子
        ttk.Label(const_frame, text="相位因子:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.phase_factor_var = tk.IntVar(value=1)
        ttk.Spinbox(const_frame, from_=0, to=20, textvariable=self.phase_factor_var, width=5).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 轨道参数区域
        orbit_frame = ttk.LabelFrame(self.control_frame, text="轨道参数")
        orbit_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 轨道高度
        ttk.Label(orbit_frame, text="轨道高度 (km):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.altitude_var = tk.DoubleVar(value=600)
        ttk.Spinbox(orbit_frame, from_=200, to=2000, textvariable=self.altitude_var, width=5).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 轨道倾角
        ttk.Label(orbit_frame, text="轨道倾角 (度):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.inclination_var = tk.DoubleVar(value=45)
        ttk.Spinbox(orbit_frame, from_=0, to=90, textvariable=self.inclination_var, width=5).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 偏心率
        ttk.Label(orbit_frame, text="偏心率:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.eccentricity_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(orbit_frame, from_=0, to=0.3, increment=0.01, textvariable=self.eccentricity_var, width=5).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 卫星物理参数区域
        sat_frame = ttk.LabelFrame(self.control_frame, text="卫星物理参数")
        sat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 面积质量比
        ttk.Label(sat_frame, text="面积质量比 (m²/kg):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.am_ratio_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(sat_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.am_ratio_var, width=5).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 阻力系数
        ttk.Label(sat_frame, text="阻力系数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cd_var = tk.DoubleVar(value=2.2)
        ttk.Spinbox(sat_frame, from_=1.5, to=3.0, increment=0.1, textvariable=self.cd_var, width=5).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 反射系数
        ttk.Label(sat_frame, text="反射系数:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.cr_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(sat_frame, from_=1.0, to=2.0, increment=0.1, textvariable=self.cr_var, width=5).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 仿真参数区域
        sim_frame = ttk.LabelFrame(self.control_frame, text="仿真参数")
        sim_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 仿真时长
        ttk.Label(sim_frame, text="仿真时长 (天):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sim_duration_var = tk.DoubleVar(value=7)
        ttk.Spinbox(sim_frame, from_=1, to=30, textvariable=self.sim_duration_var, width=5).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 时间步长
        ttk.Label(sim_frame, text="步长 (秒):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.time_step_var = tk.DoubleVar(value=600)
        ttk.Spinbox(sim_frame, from_=10, to=3600, textvariable=self.time_step_var, width=5).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 设置摄动选项
        pert_frame = ttk.LabelFrame(self.control_frame, text="摄动选项")
        pert_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # J2摄动
        self.j2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="J2摄动", variable=self.j2_var).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 大气阻力
        self.drag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="大气阻力", variable=self.drag_var).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 太阳辐射压
        self.srp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="太阳辐射压", variable=self.srp_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 按钮区域
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建星座按钮
        self.create_btn = ttk.Button(button_frame, text="创建星座", command=self.create_constellation)
        self.create_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # 运行仿真按钮
        self.run_btn = ttk.Button(button_frame, text="运行仿真", command=self.run_simulation)
        self.run_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 动画按钮
        self.animate_btn = ttk.Button(button_frame, text="播放动画", command=self.toggle_animation)
        self.animate_btn.grid(row=1, column=0, padx=5, pady=5)
        
        # 分析按钮
        self.analyze_btn = ttk.Button(button_frame, text="分析结果", command=self.analyze_results)
        self.analyze_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # 优化星座按钮
        self.optimize_btn = ttk.Button(button_frame, text="星座优化", command=self.open_optimization_window)
        self.optimize_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    def create_log_area(self):
        """创建日志区域"""
        log_frame = ttk.LabelFrame(self.control_frame, text="仿真日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, width=30, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def log(self, message):
        """向日志区域添加消息"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def setup_plot(self):
        """设置3D图表"""
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.set_title('近地星座轨道可视化')
        
        # 设置坐标轴范围
        limit = 10000  # km
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
        
        # 保持纵横比
        self.ax.set_box_aspect([1, 1, 1])
        
        # 添加网格
        self.ax.grid(True)
    
    def create_constellation(self):
        """创建星座"""
        self.log("正在创建星座...")
        
        # 获取参数
        total_satellites = self.total_satellites_var.get()
        num_planes = self.num_planes_var.get()
        relative_spacing = self.phase_factor_var.get()
        altitude = self.altitude_var.get() * 1000  # 转换为米
        inclination = np.radians(self.inclination_var.get())
        eccentricity = self.eccentricity_var.get()
        
        # 创建星座
        self.constellation_states = self.designer.create_walker_constellation(
            total_satellites, num_planes, relative_spacing, inclination, altitude, eccentricity)
        
        self.log(f"成功创建包含{total_satellites}颗卫星的Walker星座")
        self.log(f"轨道面数量: {num_planes}")
        self.log(f"轨道高度: {altitude/1000} km")
        self.log(f"轨道倾角: {np.degrees(inclination)}°")
        
        # 计算覆盖率
        coverage = self.designer.calculate_coverage(self.constellation_states)
        self.log(f"星座覆盖率: {coverage*100:.2f}%")
        
        # 计算碰撞风险
        risk, close_approaches = self.designer.calculate_collision_risk(self.constellation_states)
        self.log(f"碰撞风险指数: {risk:.4f}")
        self.log(f"近距离接近次数: {close_approaches}")
        
        # 绘制初始星座
        self.plot_constellation(self.constellation_states)
    
    def run_simulation(self):
        """运行轨道传播仿真"""
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
        
        # 创建时间数组
        t_span = (0, sim_duration)
        
        # 存储传播结果
        self.propagated_states = []
        
        # 进度更新
        total_sats = len(self.constellation_states)
        
        for i, initial_state in enumerate(self.constellation_states):
            # 更新进度
            progress = (i / total_sats) * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()
            
            # 传播轨道
            times, states = self.propagator.propagate_orbit(initial_state, t_span, am_ratio, cd, cr)
            
            # 存储结果
            self.propagated_states.append(states)
            
            self.log(f"卫星 {i+1}/{total_sats} 轨道传播完成")
        
        # 完成进度
        self.progress_var.set(100)
        
        self.log(f"仿真完成! 总时长: {sim_duration/self.propagator.DAY_SEC:.1f} 天")
        
        # 分析轨道演化
        self.analyze_orbit_evolution()
    
    def analyze_orbit_evolution(self):
        """分析轨道演化结果"""
        if self.propagated_states is None or len(self.propagated_states) == 0:
            return
        
        self.log("分析轨道演化...")
        
        # 分析轨道高度变化
        initial_heights = []
        final_heights = []
        
        for states in self.propagated_states:
            # 初始轨道高度
            r_init = np.linalg.norm(states[0][:3])
            h_init = r_init - self.propagator.RE
            initial_heights.append(h_init)
            
            # 最终轨道高度
            r_final = np.linalg.norm(states[-1][:3])
            h_final = r_final - self.propagator.RE
            final_heights.append(h_final)
        
        avg_initial_height = np.mean(initial_heights) / 1000  # 转换为km
        avg_final_height = np.mean(final_heights) / 1000  # 转换为km
        max_height_decay = (np.max(initial_heights) - np.min(final_heights)) / 1000  # 转换为km
        
        self.log(f"平均初始高度: {avg_initial_height:.2f} km")
        self.log(f"平均最终高度: {avg_final_height:.2f} km")
        self.log(f"最大高度衰减: {max_height_decay:.2f} km")
        
        # 分析轨道面演化
        initial_inc = []
        final_inc = []
        initial_raan = []
        final_raan = []
        
        for states in self.propagated_states:
            # 计算初始轨道要素
            kepler_init = self.propagator.cartesian_to_kepler(states[0])
            initial_inc.append(kepler_init[2])
            initial_raan.append(kepler_init[3])
            
            # 计算最终轨道要素
            kepler_final = self.propagator.cartesian_to_kepler(states[-1])
            final_inc.append(kepler_final[2])
            final_raan.append(kepler_final[3])
        
        avg_inc_change = np.mean(np.abs(np.array(final_inc) - np.array(initial_inc))) * 180 / np.pi  # 转换为度
        avg_raan_change = np.mean(np.abs(np.array(final_raan) - np.array(initial_raan))) * 180 / np.pi  # 转换为度
        
        self.log(f"平均倾角变化: {avg_inc_change:.4f}°")
        self.log(f"平均升交点变化: {avg_raan_change:.4f}°")
        
        # 分析碰撞风险演化
        initial_states = np.array([states[0] for states in self.propagated_states])
        final_states = np.array([states[-1] for states in self.propagated_states])
        
        initial_risk, initial_close = self.designer.calculate_collision_risk(initial_states)
        final_risk, final_close = self.designer.calculate_collision_risk(final_states)
        
        self.log(f"初始碰撞风险: {initial_risk:.4f}")
        self.log(f"最终碰撞风险: {final_risk:.4f}")
        self.log(f"初始近距离接近: {initial_close}")
        self.log(f"最终近距离接近: {final_close}")
    
    def toggle_animation(self):
        """切换动画播放状态"""
        if self.propagated_states is None or len(self.propagated_states) == 0:
            messagebox.showwarning("警告", "请先运行仿真!")
            return
        
        if self.is_animating:
            # 停止动画
            if self.animation is not None:
                self.animation.event_source.stop()
            self.is_animating = False
            self.animate_btn.config(text="播放动画")
            self.log("动画已暂停")
        else:
            # 开始动画
            self.animate_constellation()
            self.is_animating = True
            self.animate_btn.config(text="暂停动画")
            self.log("动画播放中...")
    
    def animate_constellation(self):
        """创建星座运动动画"""
        # 初始化图形对象
        self.clear_plot()
        
        # 绘制地球
        self.plot_earth()
        
        # 设置卫星和轨迹
        self.satellites = []
        self.trajectory_lines = []
        
        colors = plt.cm.jet(np.linspace(0, 1, len(self.propagated_states)))
        
        for i, states in enumerate(self.propagated_states):
            # 绘制初始位置
            sat, = self.ax.plot([], [], [], 'o', markersize=5, color=colors[i])
            self.satellites.append(sat)
            
            # 绘制轨迹
            line, = self.ax.plot([], [], [], '-', linewidth=1, alpha=0.5, color=colors[i])
            self.trajectory_lines.append(line)
        
        # 设置动画更新函数
        def update(frame):
            # 计算当前帧对应的时间点
            total_frames = min(100, len(self.propagated_states[0]))
            stride = len(self.propagated_states[0]) // total_frames
            idx = frame * stride
            
            if idx >= len(self.propagated_states[0]):
                idx = len(self.propagated_states[0]) - 1
            
            # 更新每颗卫星的位置
            for i, states in enumerate(self.propagated_states):
                if idx < len(states):
                    # 获取当前位置
                    pos = states[idx][:3] / 1000  # 转换为km
                    self.satellites[i].set_data([pos[0]], [pos[1]])
                    self.satellites[i].set_3d_properties([pos[2]])
                    
                    # 更新轨迹
                    traj = states[:idx+1:stride] / 1000  # 降采样以提高性能
                    if len(traj) > 0:
                        self.trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                        self.trajectory_lines[i].set_3d_properties(traj[:, 2])
            
            return self.satellites + self.trajectory_lines
        
        # 创建动画
        self.animation = FuncAnimation(
            self.fig, update, frames=range(100), 
            interval=100, blit=True, repeat=True)
        
        # 刷新画布
        self.canvas.draw()
    
    def plot_constellation(self, states):
        """绘制星座"""
        self.clear_plot()
        
        # 绘制地球
        self.plot_earth()
        
        # 使用彩色集表示不同轨道面
        colors = plt.cm.jet(np.linspace(0, 1, len(states)))
        
        # 绘制卫星
        for i, state in enumerate(states):
            position = state[:3] / 1000  # 转换为km
            self.ax.scatter(position[0], position[1], position[2], color=colors[i], s=30)
            
            # 计算开普勒轨道要素
            kepler = self.propagator.cartesian_to_kepler(state)
            a, e, inc, RAAN, w, M = kepler
            
            # 绘制轨道
            theta = np.linspace(0, 2*np.pi, 100)
            
            # 计算轨道面上的点
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x_orbit = r * np.cos(theta)
            y_orbit = r * np.sin(theta)
            z_orbit = np.zeros_like(theta)
            
            # 旋转到正确的轨道平面
            # 旋转矩阵
            R3_RAAN = np.array([
                [np.cos(RAAN), -np.sin(RAAN), 0],
                [np.sin(RAAN), np.cos(RAAN), 0],
                [0, 0, 1]
            ])
            
            R1_i = np.array([
                [1, 0, 0],
                [0, np.cos(inc), -np.sin(inc)],
                [0, np.sin(inc), np.cos(inc)]
            ])
            
            R3_w = np.array([
                [np.cos(w), -np.sin(w), 0],
                [np.sin(w), np.cos(w), 0],
                [0, 0, 1]
            ])
            
            # 合成旋转矩阵
            R = R3_RAAN @ R1_i @ R3_w
            
            # 绘制轨道
            orbit_points = np.vstack((x_orbit, y_orbit, z_orbit)).T
            rotated_points = np.array([R @ point for point in orbit_points])
            
            self.ax.plot(rotated_points[:, 0] / 1000, rotated_points[:, 1] / 1000, rotated_points[:, 2] / 1000, 
                        color=colors[i], alpha=0.3)
        
        # 调整视角
        self.ax.view_init(elev=30, azim=45)
        
        # 刷新画布
        self.canvas.draw()
        
        self.log(f"已绘制包含{len(states)}颗卫星的星座")
    
    def plot_earth(self):
        """绘制地球"""
        # 地球半径（km）
        r = self.propagator.RE / 1000
        
        # 创建球体网格
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 绘制地球
        self.earth = self.ax.plot_surface(
            x, y, z, color='lightblue', alpha=0.3, edgecolor='none')
        
        # 添加赤道
        theta = np.linspace(0, 2 * np.pi, 100)
        x_eq = r * np.cos(theta)
        y_eq = r * np.sin(theta)
        z_eq = np.zeros_like(theta)
        self.ax.plot(x_eq, y_eq, z_eq, 'b--', alpha=0.5, linewidth=1)
    
    def clear_plot(self):
        """清空图表"""
        self.ax.clear()
        self.setup_plot()
    
    def analyze_results(self):
        """分析仿真结果"""
        if self.propagated_states is None or len(self.propagated_states) == 0:
            messagebox.showwarning("警告", "请先运行仿真!")
            return
        
        self.log("进行深入分析...")
        
        # 创建分析窗口
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("星座演化分析结果")
        analysis_window.geometry("800x600")
        
        # 创建分析标签页
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 高度变化分析标签页
        height_frame = ttk.Frame(notebook)
        notebook.add(height_frame, text="轨道高度分析")
        self.create_height_analysis(height_frame)
        
        # 轨道面变化分析标签页
        plane_frame = ttk.Frame(notebook)
        notebook.add(plane_frame, text="轨道面分析")
        self.create_plane_analysis(plane_frame)
        
        # 碰撞风险分析标签页
        collision_frame = ttk.Frame(notebook)
        notebook.add(collision_frame, text="碰撞风险分析")
        self.create_collision_analysis(collision_frame)
        
        # 覆盖率分析标签页
        coverage_frame = ttk.Frame(notebook)
        notebook.add(coverage_frame, text="覆盖率分析")
        self.create_coverage_analysis(coverage_frame)
        
        # 优化建议标签页
        optimization_frame = ttk.Frame(notebook)
        notebook.add(optimization_frame, text="星座优化建议")
        self.create_optimization_suggestions(optimization_frame)
    
    def create_height_analysis(self, parent):
        """创建轨道高度分析图表"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 分析轨道高度变化
        time_steps = len(self.propagated_states[0])
        sim_duration = self.sim_duration_var.get()
        times = np.linspace(0, sim_duration, time_steps)
        
        for i, states in enumerate(self.propagated_states):
            heights = []
            for state in states:
                r = np.linalg.norm(state[:3])
                h = (r - self.propagator.RE) / 1000  # 转换为km
                heights.append(h)
            
            ax.plot(times, heights, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('轨道高度 (km)')
        ax.set_title('星座轨道高度随时间变化')
        ax.grid(True)
        
        # 添加统计信息
        all_heights = []
        for states in self.propagated_states:
            heights = []
            for state in states:
                r = np.linalg.norm(state[:3])
                h = (r - self.propagator.RE) / 1000  # 转换为km
                heights.append(h)
            all_heights.append(heights)
        
        all_heights = np.array(all_heights)
        
        # 计算各时间点的统计数据
        mean_heights = np.mean(all_heights, axis=0)
        std_heights = np.std(all_heights, axis=0)
        
        # 添加均值线
        ax.plot(times, mean_heights, 'r-', linewidth=2, label='均值')
        ax.fill_between(times, mean_heights - std_heights, mean_heights + std_heights, 
                       color='r', alpha=0.2, label='±1倍标准差')
        
        ax.legend()
        
        # 添加高度衰减信息
        initial_height = mean_heights[0]
        final_height = mean_heights[-1]
        height_decay = initial_height - final_height
        
        text_info = (f"初始平均高度: {initial_height:.2f} km\n"
                    f"最终平均高度: {final_height:.2f} km\n"
                    f"平均高度衰减: {height_decay:.2f} km\n"
                    f"衰减率: {height_decay/sim_duration:.2f} km/天")
        
        ax.annotate(text_info, xy=(0.02, 0.02), xycoords='axes fraction', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        canvas.draw()
    
    def create_plane_analysis(self, parent):
        """创建轨道面变化分析图表"""
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 分析轨道倾角和升交点变化
        time_steps = len(self.propagated_states[0])
        sim_duration = self.sim_duration_var.get()
        times = np.linspace(0, sim_duration, time_steps)
        
        all_inc = []
        all_raan = []
        
        for i, states in enumerate(self.propagated_states):
            incs = []
            raans = []
            
            for state in states:
                kepler = self.propagator.cartesian_to_kepler(state)
                inc = kepler[2] * 180 / np.pi  # 转换为度
                raan = kepler[3] * 180 / np.pi  # 转换为度
                
                incs.append(inc)
                raans.append(raan)
            
            all_inc.append(incs)
            all_raan.append(raans)
            
            color = plt.cm.jet(i / len(self.propagated_states))
            ax1.plot(times, incs, color=color, alpha=0.7, linewidth=1)
            ax2.plot(times, raans, color=color, alpha=0.7, linewidth=1)
        
        all_inc = np.array(all_inc)
        all_raan = np.array(all_raan)
        
        # 计算统计数据
        mean_inc = np.mean(all_inc, axis=0)
        std_inc = np.std(all_inc, axis=0)
        mean_raan = np.mean(all_raan, axis=0)
        std_raan = np.std(all_raan, axis=0)
        
        # 绘制平均值和标准差
        ax1.plot(times, mean_inc, 'r-', linewidth=2)
        ax1.fill_between(times, mean_inc - std_inc, mean_inc + std_inc, 
                        color='r', alpha=0.2)
        
        ax2.plot(times, mean_raan, 'r-', linewidth=2)
        ax2.fill_between(times, mean_raan - std_raan, mean_raan + std_raan,
                         color='r', alpha=0.2)
        
        # 设置标签
        ax1.set_xlabel('时间 (天)')
        ax1.set_ylabel('倾角 (度)')
        ax1.set_title('轨道倾角变化')
        ax1.grid(True)
        
        ax2.set_xlabel('时间 (天)')
        ax2.set_ylabel('升交点赤经 (度)')
        ax2.set_title('升交点赤经变化')
        ax2.grid(True)
        
        # 在第三张图中绘制升交点漂移率分布
        raan_drifts = []
        for raans in all_raan:
            # 计算每天的平均漂移率
            if len(raans) >= 2:
                total_drift = raans[-1] - raans[0]
                days = times[-1] - times[0]
                if days > 0:
                    drift_rate = total_drift / days
                    raan_drifts.append(drift_rate)
        
        ax3.hist(raan_drifts, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('升交点漂移率 (度/天)')
        ax3.set_ylabel('卫星数量')
        ax3.set_title('升交点漂移率分布')
        ax3.grid(True)
        
        # 添加统计信息
        if raan_drifts:
            mean_drift = np.mean(raan_drifts)
            std_drift = np.std(raan_drifts)
            
            text_info = (f"平均漂移率: {mean_drift:.4f} 度/天\n"
                        f"标准差: {std_drift:.4f} 度/天\n"
                        f"J2摄动理论预测值: {self.calculate_j2_raan_drift():.4f} 度/天")
            
            ax3.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction', 
                       va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        canvas.draw()
    
    def calculate_j2_raan_drift(self):
        """计算J2摄动导致的升交点漂移理论值（度/天）"""
        # 获取轨道参数
        altitude = self.altitude_var.get() * 1000  # 转换为米
        inclination = np.radians(self.inclination_var.get())
        eccentricity = self.eccentricity_var.get()
        
        a = altitude + self.propagator.RE
        
        # J2摄动导致的升交点漂移
        n = np.sqrt(self.propagator.MU / a**3)  # 平均运动（弧度/秒）
        
        drift_rate = -3 * self.propagator.J2 * self.propagator.RE**2 * n * np.cos(inclination) / (2 * a**2 * (1 - eccentricity**2)**2)
        
        # 转换为度/天
        drift_rate_deg_day = drift_rate * 180 / np.pi * self.propagator.DAY_SEC
        
        return drift_rate_deg_day
    
    def create_collision_analysis(self, parent):
        """创建碰撞风险分析图表"""
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 分析轨道演化过程中的碰撞风险
        time_steps = len(self.propagated_states[0])
        sim_duration = self.sim_duration_var.get()
        
        # 选择关键时间点进行分析
        sample_points = min(10, time_steps)
        sample_indices = np.linspace(0, time_steps-1, sample_points, dtype=int)
        sample_times = np.linspace(0, sim_duration, sample_points)
        
        risks = []
        close_approaches = []
        
        self.log("计算碰撞风险演化...")
        
        for idx in sample_indices:
            # 提取特定时间点的所有卫星状态
            states_at_time = np.array([states[idx] for states in self.propagated_states])
            
            # 计算碰撞风险和近距离接近
            risk, close = self.designer.calculate_collision_risk(states_at_time)
            
            risks.append(risk)
            close_approaches.append(close)
        
        # 绘制风险变化
        ax1.plot(sample_times, risks, 'ro-', linewidth=2)
        ax1.set_xlabel('时间 (天)')
        ax1.set_ylabel('风险指数')
        ax1.set_title('碰撞风险演化')
        ax1.grid(True)
        
        # 绘制近距离接近次数
        ax2.plot(sample_times, close_approaches, 'bo-', linewidth=2)
        ax2.set_xlabel('时间 (天)')
        ax2.set_ylabel('接近次数')
        ax2.set_title('近距离接近次数演化')
        ax2.grid(True)
        
        # 计算最小距离矩阵
        self.log("计算最小距离矩阵...")
        
        # 使用最终状态计算最小距离
        final_states = np.array([states[-1] for states in self.propagated_states])
        num_satellites = len(final_states)
        
        min_distance_matrix = np.zeros((num_satellites, num_satellites))
        
        for i in range(num_satellites):
            for j in range(i+1, num_satellites):
                dist = np.linalg.norm(final_states[i][:3] - final_states[j][:3]) / 1000  # 转换为km
                min_distance_matrix[i, j] = dist
                min_distance_matrix[j, i] = dist
        
        # 绘制热图
        im = ax3.imshow(min_distance_matrix, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax3, label='最小距离 (km)')
        
        ax3.set_xlabel('卫星索引')
        ax3.set_ylabel('卫星索引')
        ax3.set_title('卫星间最小距离矩阵')
        
        # 添加风险分析
        initial_risk = risks[0]
        final_risk = risks[-1]
        risk_change = final_risk - initial_risk
        
        text_info = (f"初始风险指数: {initial_risk:.4f}\n"
                    f"最终风险指数: {final_risk:.4f}\n"
                    f"风险变化: {risk_change:+.4f}\n"
                    f"最小卫星间距: {np.min(min_distance_matrix[min_distance_matrix > 0]):.2f} km")
        
        ax1.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction', 
                   va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        canvas.draw()
    
    def create_coverage_analysis(self, parent):
        """创建覆盖率分析图表"""
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], projection='3d')
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 分析覆盖率演化
        time_steps = len(self.propagated_states[0])
        sim_duration = self.sim_duration_var.get()
        
        # 选择关键时间点进行分析
        sample_points = min(10, time_steps)
        sample_indices = np.linspace(0, time_steps-1, sample_points, dtype=int)
        sample_times = np.linspace(0, sim_duration, sample_points)
        
        coverages = []
        
        self.log("计算覆盖率演化...")
        
        for idx in sample_indices:
            # 提取特定时间点的所有卫星状态
            states_at_time = np.array([states[idx] for states in self.propagated_states])
            
            # 计算覆盖率
            coverage = self.designer.calculate_coverage(states_at_time)
            coverages.append(coverage)
        
        # 绘制覆盖率变化
        ax1.plot(sample_times, coverages, 'go-', linewidth=2)
        ax1.set_xlabel('时间 (天)')
        ax1.set_ylabel('覆盖率')
        ax1.set_title('星座覆盖率演化')
        ax1.grid(True)
        
        # 添加覆盖率变化信息
        initial_coverage = coverages[0]
        final_coverage = coverages[-1]
        coverage_change = final_coverage - initial_coverage
        
        text_info = (f"初始覆盖率: {initial_coverage*100:.2f}%\n"
                    f"最终覆盖率: {final_coverage*100:.2f}%\n"
                    f"覆盖率变化: {coverage_change*100:+.2f}%")
        
        ax1.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction', 
                   va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 绘制最终星座的3D覆盖图
        # 获取最终状态
        final_states = np.array([states[-1] for states in self.propagated_states])
        
        # 绘制地球
        r = self.propagator.RE / 1000  # 转换为km
        
        # 创建球体网格
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 绘制地球
        ax2.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')
        
        # 创建全球网格
        lat_grid = np.linspace(-90, 90, 18)  # 10度一格
        lon_grid = np.linspace(-180, 180, 36)  # 10度一格
        
        # 计算每个网格点的覆盖情况
        coverage_map = np.zeros((len(lat_grid), len(lon_grid)))
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                # 转换网格点到ECEF坐标
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                
                x = r * np.cos(lat_rad) * np.cos(lon_rad)
                y = r * np.cos(lat_rad) * np.sin(lon_rad)
                z = r * np.sin(lat_rad)
                
                ground_point = np.array([x, y, z])
                
                # 检查任一卫星是否覆盖该点
                is_covered = False
                elevation_mask = 10.0  # 度
                
                for state in final_states:
                    sat_pos = state[:3] / 1000  # 转换为km
                    
                    # 计算卫星到地面点的方向向量
                    sat_to_ground = ground_point - sat_pos
                    distance = np.linalg.norm(sat_to_ground)
                    
                    # 计算高度角
                    cos_elevation = np.dot(ground_point, sat_to_ground) / (np.linalg.norm(ground_point) * distance)
                    elevation = 90 - np.degrees(np.arccos(cos_elevation))
                    
                    if elevation >= elevation_mask:
                        is_covered = True
                        break
                
                coverage_map[i, j] = 1 if is_covered else 0
        
        # 绘制覆盖点
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                if coverage_map[i, j] > 0:
                    lat_rad = np.radians(lat)
                    lon_rad = np.radians(lon)
                    
                    x = (r + 0.1) * np.cos(lat_rad) * np.cos(lon_rad)
                    y = (r + 0.1) * np.cos(lat_rad) * np.sin(lon_rad)
                    z = (r + 0.1) * np.sin(lat_rad)
                    
                    ax2.scatter(x, y, z, c='green', s=10, alpha=0.7)
        
        # 绘制卫星
        for state in final_states:
            pos = state[:3] / 1000  # 转换为km
            ax2.scatter(pos[0], pos[1], pos[2], c='red', s=30)
        
        ax2.set_title('最终星座覆盖图')
        ax2.set_axis_off()
        ax2.view_init(elev=30, azim=45)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_optimization_suggestions(self, parent):
        """创建星座优化建议界面"""
        # 创建控制区域
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 优化目标
        ttk.Label(control_frame, text="优化目标:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        objective_var = tk.StringVar(value="multi_objective")
        objective_combo = ttk.Combobox(control_frame, textvariable=objective_var, 
                                       values=["coverage", "collision_risk", "multi_objective"])
        objective_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 最大迭代次数
        ttk.Label(control_frame, text="最大迭代次数:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        max_iter_var = tk.IntVar(value=10)
        max_iter_spin = ttk.Spinbox(control_frame, from_=5, to=20, textvariable=max_iter_var, width=5)
        max_iter_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 权重设置
        weights_frame = ttk.LabelFrame(parent, text="优化权重")
        weights_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 覆盖率权重
        ttk.Label(weights_frame, text="覆盖率权重:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        coverage_weight_var = tk.DoubleVar(value=0.6)
        coverage_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                          textvariable=coverage_weight_var, width=5)
        coverage_weight_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 碰撞风险权重
        ttk.Label(weights_frame, text="碰撞风险权重:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        collision_weight_var = tk.DoubleVar(value=0.3)
        collision_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                          textvariable=collision_weight_var, width=5)
        collision_weight_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 能量效率权重
        ttk.Label(weights_frame, text="能量效率权重:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        energy_weight_var = tk.DoubleVar(value=0.1)
        energy_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                        textvariable=energy_weight_var, width=5)
        energy_weight_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 参数范围设置
        params_frame = ttk.LabelFrame(parent, text="参数范围")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 卫星数量范围
        ttk.Label(params_frame, text="卫星数量范围:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        sat_min_var = tk.IntVar(value=12)
        sat_min_spin = ttk.Spinbox(params_frame, from_=6, to=30, textvariable=sat_min_var, width=5)
        sat_min_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        sat_max_var = tk.IntVar(value=48)
        sat_max_spin = ttk.Spinbox(params_frame, from_=12, to=60, textvariable=sat_max_var, width=5)
        sat_max_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道面数量范围
        ttk.Label(params_frame, text="轨道面数量范围:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        plane_min_var = tk.IntVar(value=3)
        plane_min_spin = ttk.Spinbox(params_frame, from_=2, to=8, textvariable=plane_min_var, width=5)
        plane_min_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        plane_max_var = tk.IntVar(value=8)
        plane_max_spin = ttk.Spinbox(params_frame, from_=3, to=12, textvariable=plane_max_var, width=5)
        plane_max_spin.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道倾角范围
        ttk.Label(params_frame, text="轨道倾角范围(°):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        inc_min_var = tk.DoubleVar(value=30)
        inc_min_spin = ttk.Spinbox(params_frame, from_=0, to=60, increment=5,
                                 textvariable=inc_min_var, width=5)
        inc_min_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        inc_max_var = tk.DoubleVar(value=90)
        inc_max_spin = ttk.Spinbox(params_frame, from_=45, to=90, increment=5,
                                 textvariable=inc_max_var, width=5)
        inc_max_spin.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道高度范围
        ttk.Label(params_frame, text="轨道高度范围(km):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        alt_min_var = tk.DoubleVar(value=500)
        alt_min_spin = ttk.Spinbox(params_frame, from_=300, to=800, increment=50,
                                 textvariable=alt_min_var, width=5)
        alt_min_spin.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        
        alt_max_var = tk.DoubleVar(value=1200)
        alt_max_spin = ttk.Spinbox(params_frame, from_=600, to=2000, increment=50,
                                 textvariable=alt_max_var, width=5)
        alt_max_spin.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(parent, text="优化结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 结果文本框
        result_text = tk.Text(result_frame, wrap=tk.WORD, width=80, height=10)
        result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_text, command=result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.config(yscrollcommand=scrollbar.set)
        
        # 运行优化按钮
        run_button = ttk.Button(parent, text="运行优化", 
                              command=lambda: self.run_optimization(
                                  objective_var.get(), 
                                  (sat_min_var.get(), sat_max_var.get()),
                                  (plane_min_var.get(), plane_max_var.get()),
                                  (np.radians(inc_min_var.get()), np.radians(inc_max_var.get())),
                                  (alt_min_var.get() * 1000, alt_max_var.get() * 1000),
                                  {
                                      'coverage': coverage_weight_var.get(),
                                      'collision_risk': collision_weight_var.get(),
                                      'energy': energy_weight_var.get()
                                  },
                                  max_iter_var.get(),
                                  result_text
                              ))
        run_button.pack(padx=10, pady=10)
        
        # 应用最优配置按钮
        apply_button = ttk.Button(parent, text="应用最优配置", 
                                command=lambda: self.apply_optimal_configuration(result_text))
        apply_button.pack(padx=10, pady=10)
    
    def run_optimization(self, objective, satellites_range, planes_range, inclination_range, 
                       altitude_range, weights, max_iterations, result_text):
        """运行星座优化"""
        self.log(f"开始星座优化计算 (目标: {objective})")
        
        # 更新结果文本框
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "星座优化计算中...\n")
        self.root.update_idletasks()
        
        # 记录原来的终端输出函数
        original_log = self.designer.log
        
        # 重定向输出到文本框
        def text_log(message):
            result_text.insert(tk.END, message + "\n")
            result_text.see(tk.END)
            self.root.update_idletasks()
            
        self.designer.log = text_log
        
        try:
            # 运行优化
            best_params, best_score, history = self.designer.optimize_constellation(
                objective=objective,
                total_satellites=satellites_range,
                num_planes=planes_range,
                inclination=inclination_range,
                altitude=altitude_range,
                weights=weights,
                max_iterations=max_iterations
            )
            
            # 显示优化结果
            result_text.insert(tk.END, "\n========= 优化结果 =========\n")
            result_text.insert(tk.END, f"最优得分: {best_score:.4f}\n\n")
            result_text.insert(tk.END, "最优参数配置:\n")
            result_text.insert(tk.END, f"卫星总数: {best_params['total_satellites']}\n")
            result_text.insert(tk.END, f"轨道面数: {best_params['num_planes']}\n")
            result_text.insert(tk.END, f"相位因子: {best_params['relative_spacing']}\n")
            result_text.insert(tk.END, f"轨道倾角: {np.degrees(best_params['inclination']):.2f}°\n")
            result_text.insert(tk.END, f"轨道高度: {best_params['altitude']/1000:.2f} km\n")
            
            # 存储优化结果
            self.optimization_result = best_params
            
            self.log(f"星座优化完成，最优得分: {best_score:.4f}")
        except Exception as e:
            result_text.insert(tk.END, f"优化过程中出错: {str(e)}\n")
            self.log(f"优化过程中出错: {str(e)}")
        finally:
            # 恢复原来的输出函数
            self.designer.log = original_log
    
    def apply_optimal_configuration(self, result_text):
        """应用最优星座配置"""
        if not hasattr(self, 'optimization_result') or self.optimization_result is None:
            result_text.insert(tk.END, "没有可用的优化结果!\n")
            messagebox.showwarning("警告", "没有可用的优化结果，请先运行优化!")
            return
        
        # 更新界面参数
        params = self.optimization_result
        
        try:
            self.total_satellites_var.set(params['total_satellites'])
            self.num_planes_var.set(params['num_planes'])
            self.phase_factor_var.set(params['relative_spacing'])
            self.inclination_var.set(np.degrees(params['inclination']))
            self.altitude_var.set(params['altitude'] / 1000)
            self.eccentricity_var.set(params['eccentricity'])
            
            result_text.insert(tk.END, "已应用最优配置到主界面参数!\n")
            result_text.see(tk.END)
            self.log("已应用最优星座配置")
            
            # 询问用户是否自动创建星座
            if messagebox.askyesno("确认", "是否立即使用最优参数创建星座?"):
                # 自动创建星座
                self.create_constellation()
        except Exception as e:
            error_msg = f"应用配置时出错: {str(e)}"
            result_text.insert(tk.END, error_msg + "\n")
            result_text.see(tk.END)
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def open_optimization_window(self):
        """打开星座优化窗口"""
        # 创建优化窗口
        opt_window = tk.Toplevel(self.root)
        opt_window.title("星座优化")
        opt_window.geometry("800x700")
        
        # 创建控制区域
        control_frame = ttk.Frame(opt_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 优化目标
        ttk.Label(control_frame, text="优化目标:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        objective_var = tk.StringVar(value="multi_objective")
        objective_combo = ttk.Combobox(control_frame, textvariable=objective_var, 
                                      values=["coverage", "collision_risk", "multi_objective"])
        objective_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 最大迭代次数
        ttk.Label(control_frame, text="最大迭代次数:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        max_iter_var = tk.IntVar(value=10)
        max_iter_spin = ttk.Spinbox(control_frame, from_=5, to=20, textvariable=max_iter_var, width=5)
        max_iter_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 权重设置
        weights_frame = ttk.LabelFrame(opt_window, text="优化权重")
        weights_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 覆盖率权重
        ttk.Label(weights_frame, text="覆盖率权重:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        coverage_weight_var = tk.DoubleVar(value=0.6)
        coverage_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                          textvariable=coverage_weight_var, width=5)
        coverage_weight_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 碰撞风险权重
        ttk.Label(weights_frame, text="碰撞风险权重:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        collision_weight_var = tk.DoubleVar(value=0.3)
        collision_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                           textvariable=collision_weight_var, width=5)
        collision_weight_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 能量效率权重
        ttk.Label(weights_frame, text="能量效率权重:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        energy_weight_var = tk.DoubleVar(value=0.1)
        energy_weight_spin = ttk.Spinbox(weights_frame, from_=0.1, to=1.0, increment=0.1,
                                        textvariable=energy_weight_var, width=5)
        energy_weight_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 参数范围设置
        params_frame = ttk.LabelFrame(opt_window, text="参数范围")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 卫星数量范围
        ttk.Label(params_frame, text="卫星数量范围:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        sat_min_var = tk.IntVar(value=12)
        sat_min_spin = ttk.Spinbox(params_frame, from_=6, to=30, textvariable=sat_min_var, width=5)
        sat_min_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        sat_max_var = tk.IntVar(value=48)
        sat_max_spin = ttk.Spinbox(params_frame, from_=12, to=60, textvariable=sat_max_var, width=5)
        sat_max_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道面数量范围
        ttk.Label(params_frame, text="轨道面数量范围:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        plane_min_var = tk.IntVar(value=3)
        plane_min_spin = ttk.Spinbox(params_frame, from_=2, to=8, textvariable=plane_min_var, width=5)
        plane_min_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        plane_max_var = tk.IntVar(value=8)
        plane_max_spin = ttk.Spinbox(params_frame, from_=3, to=12, textvariable=plane_max_var, width=5)
        plane_max_spin.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道倾角范围
        ttk.Label(params_frame, text="轨道倾角范围(°):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        inc_min_var = tk.DoubleVar(value=30)
        inc_min_spin = ttk.Spinbox(params_frame, from_=0, to=60, increment=5,
                                 textvariable=inc_min_var, width=5)
        inc_min_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        inc_max_var = tk.DoubleVar(value=90)
        inc_max_spin = ttk.Spinbox(params_frame, from_=45, to=90, increment=5,
                                 textvariable=inc_max_var, width=5)
        inc_max_spin.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 轨道高度范围
        ttk.Label(params_frame, text="轨道高度范围(km):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        alt_min_var = tk.DoubleVar(value=500)
        alt_min_spin = ttk.Spinbox(params_frame, from_=300, to=800, increment=50,
                                 textvariable=alt_min_var, width=5)
        alt_min_spin.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(params_frame, text="至").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        
        alt_max_var = tk.DoubleVar(value=1200)
        alt_max_spin = ttk.Spinbox(params_frame, from_=600, to=2000, increment=50,
                                 textvariable=alt_max_var, width=5)
        alt_max_spin.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(opt_window, text="优化结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 结果文本框
        result_text = tk.Text(result_frame, wrap=tk.WORD, width=80, height=20)
        result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_text, command=result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.config(yscrollcommand=scrollbar.set)
        
        # 优化描述
        description = """
星座优化功能说明:

该工具将在指定参数范围内搜索最优星座配置。支持以下优化目标:
- coverage: 最大化全球覆盖率
- collision_risk: 最小化卫星间碰撞风险
- multi_objective: 综合考虑覆盖率、碰撞风险和能量效率的多目标优化

优化过程将生成多种星座参数组合并评估其性能，最终推荐最优配置。
        """
        result_text.insert(tk.END, description)
        
        # 按钮区域
        button_frame = ttk.Frame(opt_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 运行优化按钮
        run_button = ttk.Button(button_frame, text="运行优化", 
                              command=lambda: self.run_optimization(
                                  objective_var.get(), 
                                  (sat_min_var.get(), sat_max_var.get()),
                                  (plane_min_var.get(), plane_max_var.get()),
                                  (np.radians(inc_min_var.get()), np.radians(inc_max_var.get())),
                                  (alt_min_var.get() * 1000, alt_max_var.get() * 1000),
                                  {
                                      'coverage': coverage_weight_var.get(),
                                      'collision_risk': collision_weight_var.get(),
                                      'energy': energy_weight_var.get()
                                  },
                                  max_iter_var.get(),
                                  result_text
                              ))
        run_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 应用最优配置按钮
        apply_button = ttk.Button(button_frame, text="应用最优配置", 
                                command=lambda: self.apply_optimal_configuration(result_text))
        apply_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=opt_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 设置模态窗口
        opt_window.transient(self.root)
        opt_window.grab_set()
        self.root.wait_window(opt_window)


if __name__ == "__main__":
    # 检查必要的模块是否已导入
    import sys
    missing_modules = []
    
    try:
        from numerical_methods import NumericalIntegrators, ErrorAnalysis
        numerical_methods_available = True
    except ImportError:
        numerical_methods_available = False
        missing_modules.append("numerical_methods")
    
    try:
        import perturbation_models
        perturbation_models_available = True
    except ImportError:
        perturbation_models_available = False
        missing_modules.append("perturbation_models")
    
    try:
        import visualization_enhancements
        visualization_available = True
    except ImportError:
        visualization_available = False
        missing_modules.append("visualization_enhancements")
    
    try:
        import ui_extensions
        ui_extensions_available = True
    except ImportError:
        ui_extensions_available = False
        missing_modules.append("ui_extensions")
    
    # 如果缺少模块，显示警告
    if missing_modules:
        print("警告: 以下模块未找到，部分功能可能无法使用:")
        for module in missing_modules:
            print(f"  - {module}")
        print("程序将以基本功能继续运行")
        print("-" * 50)
    
    # 启动主应用程序
    try:
        root = tk.Tk()
        app = ConstellationSimulationApp(root)
        root.mainloop()
    except Exception as e:
        # 显示错误对话框
        import traceback
        error_message = f"运行时错误:\n{str(e)}\n\n{traceback.format_exc()}"
        
        if 'root' not in locals() or root is None:
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
        
        error_window = tk.Toplevel(root)
        error_window.title("程序错误")
        error_window.geometry("600x400")
        
        # 错误信息显示
        error_frame = ttk.Frame(error_window)
        error_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        error_label = ttk.Label(error_frame, text="程序运行中发生错误:", font=('SimHei', 12))
        error_label.pack(pady=5)
        
        # 错误文本框
        error_text = tk.Text(error_frame, wrap=tk.WORD, width=70, height=15)
        error_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        error_text.insert(tk.END, error_message)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(error_text, command=error_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        error_text.config(yscrollcommand=scrollbar.set)
        
        # 关闭按钮
        close_button = ttk.Button(error_window, text="关闭", command=root.destroy)
        close_button.pack(pady=10)
        
        error_window.protocol("WM_DELETE_WINDOW", root.destroy)
        
        root.mainloop()
