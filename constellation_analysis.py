"""
星座分析模块，提供高效的星座性能评估和分析工具
支持多线程并行计算和高级分析算法
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import time
import threading
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

class ConstellationAnalyzer:
    """星座分析器，提供高效的星座性能评估和分析功能"""
    
    def __init__(self, propagator=None, max_workers=None):
        """
        初始化星座分析器
        
        参数:
        - propagator: 轨道传播器对象（可选）
        - max_workers: 最大并行工作线程数（默认为CPU核心数）
        """
        self.propagator = propagator
        # 设置最大工作线程数，默认为CPU核心数
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        # 创建状态锁
        self.lock = threading.Lock()
        # 地球参数
        self.earth_radius = 6378.137  # 地球半径（千米）
        self.mu = 3.986004418e14      # 地球引力常数（m^3/s^2）
        
    def analyze_coverage(self, constellation_states, 
                        grid_size=36, elevation_mask=10.0, 
                        use_parallel=True, return_map=False):
        """
        分析星座覆盖性能
        
        参数:
        - constellation_states: 星座状态列表，每个元素是一个卫星的状态向量
        - grid_size: 全球网格分辨率（划分纬度和经度的格数）
        - elevation_mask: 最小仰角（度）
        - use_parallel: 是否使用并行计算
        - return_map: 是否返回覆盖率地图
        
        返回:
        - 覆盖分析结果字典
        """
        # 创建全球网格点
        lat_grid = np.linspace(-90, 90, grid_size)
        lon_grid = np.linspace(-180, 180, grid_size)
        
        # 转换为弧度
        lat_rad = np.radians(lat_grid)
        lon_rad = np.radians(lon_grid)
        
        # 最小仰角（弧度）
        elev_mask_rad = np.radians(elevation_mask)
        
        # 创建覆盖率矩阵
        coverage_matrix = np.zeros((grid_size, grid_size))
        
        # 定义单点覆盖计算函数
        def calculate_point_coverage(i, j):
            lat = lat_rad[i]
            lon = lon_rad[j]
            
            # 计算地面点的ECEF坐标
            x = self.earth_radius * np.cos(lat) * np.cos(lon)
            y = self.earth_radius * np.cos(lat) * np.sin(lon)
            z = self.earth_radius * np.sin(lat)
            ground_point = np.array([x, y, z])
            
            # 检查每个卫星对该点的覆盖
            for satellite_state in constellation_states:
                # 提取卫星位置（假设前三个元素是位置向量，单位是米）
                sat_pos = satellite_state[:3] / 1000.0  # 转换为千米
                
                # 计算卫星到地面点的向量
                sat_to_ground = ground_point - sat_pos
                sat_to_ground_norm = np.linalg.norm(sat_to_ground)
                
                # 计算地心到地面点的向量
                earth_to_ground = ground_point
                earth_to_ground_norm = np.linalg.norm(earth_to_ground)
                
                # 计算仰角
                cos_angle = np.dot(sat_to_ground, earth_to_ground) / (sat_to_ground_norm * earth_to_ground_norm)
                angle = np.arccos(min(1.0, max(-1.0, cos_angle)))  # 防止数值误差导致超出范围
                elevation = np.pi/2 - angle
                
                # 如果仰角大于最小仰角，认为该点被覆盖
                if elevation >= elev_mask_rad:
                    return 1.0
            
            # 如果没有卫星覆盖，返回0
            return 0.0
        
        # 并行计算所有网格点的覆盖情况
        if use_parallel:
            # 准备所有点的索引
            points = [(i, j) for i in range(grid_size) for j in range(grid_size)]
            
            # 并行计算
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(calculate_point_coverage, i, j) for i, j in points]
                
                # 获取结果
                for idx, future in enumerate(futures):
                    i, j = points[idx]
                    coverage_matrix[i, j] = future.result()
        else:
            # 串行计算
            for i in range(grid_size):
                for j in range(grid_size):
                    coverage_matrix[i, j] = calculate_point_coverage(i, j)
        
        # 计算全球覆盖率，考虑纬度权重（因为不同纬度的面积不同）
        total_weight = 0
        weighted_coverage = 0
        
        for i, lat in enumerate(lat_rad):
            # 纬度权重与cos(lat)成正比
            weight = np.cos(lat)
            total_weight += weight
            
            # 该纬度的平均覆盖率
            weighted_coverage += weight * np.mean(coverage_matrix[i, :])
        
        # 归一化得到全球加权覆盖率
        global_coverage = weighted_coverage / total_weight if total_weight > 0 else 0
        
        # 创建结果字典
        result = {
            'global_coverage': global_coverage,
            'min_elevation': elevation_mask,
            'coverage_by_latitude': {lat_grid[i]: np.mean(coverage_matrix[i, :]) for i in range(grid_size)},
            'coverage_by_longitude': {lon_grid[j]: np.mean(coverage_matrix[:, j]) for j in range(grid_size)}
        }
        
        # 如果需要，返回覆盖率地图
        if return_map:
            result['coverage_matrix'] = coverage_matrix
            result['lat_grid'] = lat_grid
            result['lon_grid'] = lon_grid
            
        return result
    
    def analyze_collision_risk(self, constellation_states, safe_distance=1.0, 
                              use_parallel=True, return_matrix=False):
        """
        分析星座碰撞风险
        
        参数:
        - constellation_states: 星座状态列表，每个元素是一个卫星的状态向量
        - safe_distance: 安全距离（千米）
        - use_parallel: 是否使用并行计算
        - return_matrix: 是否返回距离矩阵
        
        返回:
        - 碰撞风险分析结果字典
        """
        # 卫星数量
        n_satellites = len(constellation_states)
        
        if n_satellites < 2:
            return {
                'collision_risk': 0.0,
                'min_distance': float('inf'),
                'risky_pairs': []
            }
        
        # 创建距离矩阵
        distance_matrix = np.zeros((n_satellites, n_satellites))
        
        # 定义计算两颗卫星距离的函数
        def calculate_distance(i, j):
            if i == j:
                return float('inf')  # 自己到自己的距离设为无穷大
                
            # 提取卫星位置（假设前三个元素是位置向量，单位是米）
            pos_i = constellation_states[i][:3] / 1000.0  # 转换为千米
            pos_j = constellation_states[j][:3] / 1000.0
            
            # 计算距离
            distance = np.linalg.norm(pos_i - pos_j)
            
            return distance
        
        # 并行计算所有卫星对之间的距离
        if use_parallel:
            # 准备所有卫星对的索引
            pairs = [(i, j) for i in range(n_satellites) for j in range(i+1, n_satellites)]
            
            # 并行计算
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(calculate_distance, i, j) for i, j in pairs]
                
                # 获取结果
                for idx, future in enumerate(futures):
                    i, j = pairs[idx]
                    distance = future.result()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # 对称矩阵
        else:
            # 串行计算
            for i in range(n_satellites):
                for j in range(i+1, n_satellites):
                    distance = calculate_distance(i, j)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # 对称矩阵
        
        # 识别风险卫星对
        risky_pairs = []
        for i in range(n_satellites):
            for j in range(i+1, n_satellites):
                if distance_matrix[i, j] < safe_distance:
                    risky_pairs.append({
                        'satellite_i': i,
                        'satellite_j': j,
                        'distance': distance_matrix[i, j]
                    })
        
        # 计算最小距离
        min_distance = np.min(distance_matrix[distance_matrix > 0])
        
        # 计算总体碰撞风险（基于距离的指标）
        if len(risky_pairs) == 0:
            collision_risk = 0.0
        else:
            # 风险与1/距离成正比
            risk_values = [1.0 / max(0.001, pair['distance']) for pair in risky_pairs]
            collision_risk = np.sum(risk_values) / (n_satellites * (n_satellites - 1) / 2)
            
            # 归一化风险值到[0,1]区间
            collision_risk = min(1.0, collision_risk * safe_distance)
        
        # 创建结果字典
        result = {
            'collision_risk': collision_risk,
            'min_distance': min_distance,
            'risky_pairs': risky_pairs,
            'safe_distance': safe_distance,
            'n_satellites': n_satellites
        }
        
        # 如果需要，返回距离矩阵
        if return_matrix:
            result['distance_matrix'] = distance_matrix
            
        return result
    
    def analyze_revisit_time(self, constellation_states, locations, 
                            duration=86400, steps=1000, elevation_mask=10.0,
                            use_parallel=True):
        """
        分析特定位置的重访时间
        
        参数:
        - constellation_states: 星座初始状态列表
        - locations: 地面位置列表，每个元素是(纬度,经度)，单位：度
        - duration: 分析持续时间（秒）
        - steps: 时间步数
        - elevation_mask: 最小仰角（度）
        - use_parallel: 是否使用并行计算
        
        返回:
        - 重访时间分析结果字典
        """
        if self.propagator is None:
            raise ValueError("需要提供轨道传播器以分析重访时间")
            
        # 确保locations是列表
        if not isinstance(locations, list):
            locations = [locations]
            
        # 最小仰角（弧度）
        elev_mask_rad = np.radians(elevation_mask)
        
        # 时间步长
        dt = duration / steps
        
        # 时间序列
        time_points = np.linspace(0, duration, steps)
        
        # 转换位置到ECEF坐标
        ground_points = []
        for lat, lon in locations:
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = self.earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = self.earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = self.earth_radius * np.sin(lat_rad)
            ground_points.append(np.array([x, y, z]))
            
        # 创建结果容器
        visibility_history = np.zeros((len(locations), steps), dtype=bool)
        access_times = {i: [] for i in range(len(locations))}
        revisit_times = {i: [] for i in range(len(locations))}
        
        # 定义单点可见性计算函数
        def calculate_point_visibility(location_idx, time_idx):
            t = time_points[time_idx]
            ground_point = ground_points[location_idx]
            
            # 检查每个卫星对该点在该时刻的可见性
            for sat_idx, sat_state in enumerate(constellation_states):
                # 传播卫星轨道到当前时间点
                if time_idx == 0:
                    # 对于第一个时间点，使用初始状态
                    sat_pos = sat_state[:3] / 1000.0  # 转换为千米
                else:
                    # 对于后续时间点，传播轨道
                    t_span = (0, t)
                    result = self.propagator.propagate_orbit(sat_state, t_span)
                    sat_pos = result[1][-1, :3] / 1000.0  # 使用最后一个点的位置
                
                # 计算卫星到地面点的向量
                sat_to_ground = ground_point - sat_pos
                sat_to_ground_norm = np.linalg.norm(sat_to_ground)
                
                # 计算地心到地面点的向量
                earth_to_ground = ground_point
                earth_to_ground_norm = np.linalg.norm(earth_to_ground)
                
                # 计算仰角
                cos_angle = np.dot(sat_to_ground, earth_to_ground) / (sat_to_ground_norm * earth_to_ground_norm)
                angle = np.arccos(min(1.0, max(-1.0, cos_angle)))
                elevation = np.pi/2 - angle
                
                # 如果仰角大于最小仰角，认为该点被覆盖
                if elevation >= elev_mask_rad:
                    return True
            
            # 如果没有卫星覆盖，返回False
            return False
        
        # 并行计算所有点在所有时间的可见性
        if use_parallel:
            # 准备所有计算任务
            tasks = [(loc_idx, time_idx) for loc_idx in range(len(locations)) for time_idx in range(steps)]
            
            # 并行计算
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(calculate_point_visibility, loc_idx, time_idx) for loc_idx, time_idx in tasks]
                
                # 获取结果
                for idx, future in enumerate(futures):
                    loc_idx, time_idx = tasks[idx]
                    visibility_history[loc_idx, time_idx] = future.result()
        else:
            # 串行计算
            for loc_idx in range(len(locations)):
                for time_idx in range(steps):
                    visibility_history[loc_idx, time_idx] = calculate_point_visibility(loc_idx, time_idx)
        
        # 分析可见性历史，提取接入和重访时间
        for loc_idx in range(len(locations)):
            # 找出所有接入开始时间
            for time_idx in range(1, steps):
                if visibility_history[loc_idx, time_idx] and not visibility_history[loc_idx, time_idx-1]:
                    # 接入开始
                    access_times[loc_idx].append(time_points[time_idx])
            
            # 计算重访间隔
            if len(access_times[loc_idx]) >= 2:
                revisit_times[loc_idx] = [access_times[loc_idx][i] - access_times[loc_idx][i-1] 
                                        for i in range(1, len(access_times[loc_idx]))]
        
        # 创建结果字典
        result = {
            'locations': locations,
            'access_times': access_times,
            'revisit_times': revisit_times,
            'visibility_history': visibility_history,
            'time_points': time_points
        }
        
        # 计算统计指标
        avg_revisit = {}
        max_revisit = {}
        for loc_idx in range(len(locations)):
            if revisit_times[loc_idx]:
                avg_revisit[loc_idx] = np.mean(revisit_times[loc_idx])
                max_revisit[loc_idx] = np.max(revisit_times[loc_idx])
            else:
                avg_revisit[loc_idx] = None
                max_revisit[loc_idx] = None
        
        result['avg_revisit'] = avg_revisit
        result['max_revisit'] = max_revisit
        
        return result
    
    def analyze_constellation_geometry(self, constellation_states):
        """
        分析星座几何配置
        
        参数:
        - constellation_states: 星座状态列表
        
        返回:
        - 几何分析结果字典
        """
        # 卫星数量
        n_satellites = len(constellation_states)
        
        if n_satellites < 3:
            return {
                'volume': 0.0,
                'mean_altitude': 0.0,
                'altitude_std': 0.0,
                'eccentricity': 0.0
            }
        
        # 提取卫星位置
        positions = np.array([state[:3] / 1000.0 for state in constellation_states])  # 转换为千米
        
        # 计算平均高度和标准差
        altitudes = np.linalg.norm(positions, axis=1) - self.earth_radius
        mean_altitude = np.mean(altitudes)
        altitude_std = np.std(altitudes)
        
        try:
            # 尝试计算凸包体积作为空间分布指标
            hull = ConvexHull(positions)
            volume = hull.volume
        except:
            # 如果凸包计算失败，使用替代方法
            volume = np.max(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)) ** 3
        
        # 分析轨道平面分布
        # 提取速度向量
        velocities = np.array([state[3:6] for state in constellation_states])
        
        # 计算角动量向量（叉乘位置和速度）
        h_vectors = np.cross(positions, velocities)
        
        # 归一化角动量向量，得到轨道平面法向量
        h_norms = np.linalg.norm(h_vectors, axis=1)
        h_unit = h_vectors / h_norms[:, np.newaxis]
        
        # 计算轨道平面平均角度分离度
        avg_separation = 0.0
        count = 0
        for i in range(n_satellites):
            for j in range(i+1, n_satellites):
                dot_product = np.dot(h_unit[i], h_unit[j])
                angle = np.arccos(min(1.0, max(-1.0, dot_product)))
                avg_separation += angle
                count += 1
                
        avg_separation = avg_separation / count if count > 0 else 0.0
        
        # 计算平均轨道离心率
        e_vectors = []
        for i in range(n_satellites):
            r = positions[i]
            v = velocities[i]
            r_norm = np.linalg.norm(r)
            
            # 计算离心率向量
            h_cross_v = np.cross(h_vectors[i], v)
            e_vec = h_cross_v / self.mu - r / r_norm
            e_vectors.append(e_vec)
            
        e_norms = np.linalg.norm(e_vectors, axis=1)
        mean_eccentricity = np.mean(e_norms)
        
        # 创建结果字典
        result = {
            'n_satellites': n_satellites,
            'volume': volume,
            'mean_altitude': mean_altitude,
            'altitude_std': altitude_std,
            'mean_eccentricity': mean_eccentricity,
            'avg_plane_separation': np.degrees(avg_separation)
        }
        
        return result
    
    def visualization_3d(self, constellation_states, figsize=(10, 8), 
                         show_earth=True, show_orbits=False):
        """
        创建星座3D可视化
        
        参数:
        - constellation_states: 星座状态列表
        - figsize: 图形大小
        - show_earth: 是否显示地球
        - show_orbits: 是否显示完整轨道
        
        返回:
        - matplotlib图形对象
        """
        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取卫星位置
        positions = np.array([state[:3] / 1000.0 for state in constellation_states])  # 转换为千米
        
        # 计算轴界限
        max_radius = np.max(np.linalg.norm(positions, axis=1))
        limit = max_radius * 1.2  # 留出一些边距
        
        # 绘制地球
        if show_earth:
            # 创建球体
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = self.earth_radius * np.cos(u) * np.sin(v)
            y = self.earth_radius * np.sin(u) * np.sin(v)
            z = self.earth_radius * np.cos(v)
            ax.plot_surface(x, y, z, color='blue', alpha=0.2)
        
        # 绘制卫星
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  s=50, c='red', marker='o', label='卫星')
        
        # 如果需要，绘制完整轨道
        if show_orbits and self.propagator is not None:
            for state in constellation_states:
                # 传播一个完整轨道
                try:
                    # 估算轨道周期
                    r = np.linalg.norm(state[:3])
                    a = r / (1 - np.linalg.norm(state[3:6])**2 * r / self.mu)  # 半长轴
                    period = 2 * np.pi * np.sqrt(a**3 / self.mu)  # 轨道周期
                    
                    # 传播轨道
                    t_span = (0, period)
                    result = self.propagator.propagate_orbit(state, t_span, steps=100)
                    
                    # 提取轨道点
                    orbit_points = result[1][:, :3] / 1000.0  # 转换为千米
                    
                    # 绘制轨道
                    ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], 
                           'b-', alpha=0.3)
                except:
                    pass  # 如果轨道计算失败，跳过
        
        # 设置轴界限和标签
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        
        # 设置相等的坐标比例
        ax.set_box_aspect([1, 1, 1])
        
        # 添加标题和图例
        ax.set_title(f'星座3D可视化 ({len(constellation_states)}颗卫星)')
        ax.legend()
        
        return fig 