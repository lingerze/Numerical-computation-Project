
"""
星座设计器实现
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ConstellationDesigner:
    """Walker星座设计器"""
    
    def __init__(self):
        # 地球参数
        self.earth_radius = 6378.137  # km
        self.earth_mu = 398600.4418  # km^3/s^2
        
        # 评估参数
        self.coverage_threshold = 1000  # km
        self.collision_threshold = 50   # km
        
    def create_walker_constellation(self, total_sats, num_planes, 
                                  relative_spacing, inclination, altitude):
        """
        创建Walker星座
        
        参数:
        - total_sats: 卫星总数
        - num_planes: 轨道平面数
        - relative_spacing: 相对间距参数
        - inclination: 倾角 (弧度)
        - altitude: 高度 (km)
        
        返回:
        - 卫星初始状态数组 (N×6)
        """
        sats_per_plane = total_sats // num_planes
        orbit_radius = self.earth_radius + altitude
        
        # 初始化状态数组
        states = np.zeros((total_sats, 6))
        
        for p in range(num_planes):
            # 轨道平面升交点赤经
            raan = 2 * np.pi * p / num_planes
            
            for s in range(sats_per_plane):
                # 卫星在轨道平面内的相位角
                phase = 2 * np.pi * s / sats_per_plane
                if p > 0:
                    phase += 2 * np.pi * relative_spacing * p / total_sats
                
                # 计算轨道位置和速度
                pos, vel = self._calc_orbit_state(phase, inclination, raan, altitude)
                
                # 存储状态
                idx = p * sats_per_plane + s
                states[idx, :3] = pos
                states[idx, 3:] = vel
                
        return states
    
    def _calc_orbit_state(self, phase, inc, raan, altitude):
        """计算轨道位置和速度"""
        # 轨道半径
        r = self.earth_radius + altitude
        # 轨道速度
        v = np.sqrt(self.earth_mu / r)
        
        # 在轨道平面内的位置和速度
        x = r * np.cos(phase)
        y = r * np.sin(phase)
        z = 0
        vx = -v * np.sin(phase)
        vy = v * np.cos(phase)
        vz = 0
        
        # 应用倾角和升交点赤经变换
        pos, vel = self._transform_orbit([x,y,z], [vx,vy,vz], inc, raan)
        
        return pos, vel
    
    def _transform_orbit(self, pos, vel, inc, raan):
        """应用轨道倾角和升交点赤经变换"""
        # 倾角旋转 (绕x轴)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)]
        ])
        
        # 升交点赤经旋转 (绕z轴)
        Rz = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转
        R = Rz @ Rx
        
        # 旋转位置和速度
        new_pos = R @ np.array(pos)
        new_vel = R @ np.array(vel)
        
        return new_pos, new_vel
    
    def evaluate_constellation(self, states, weights=None):
        """
        评估星座性能
        
        参数:
        - states: 卫星状态数组
        - weights: 各指标权重
        
        返回:
        - 评估指标字典
        """
        if weights is None:
            weights = {
                'coverage': 0.5,
                'collision_risk': 0.3,
                'cost': 0.2
            }
            
        metrics = {
            'coverage': self._calc_coverage(states),
            'collision_risk': self._calc_collision_risk(states),
            'cost': self._calc_cost(states)
        }
        
        # 计算综合评分
        metrics['overall_score'] = (
            weights['coverage'] * metrics['coverage'] +
            weights['collision_risk'] * (1 - metrics['collision_risk']) +
            weights['cost'] * (1 - metrics['cost'])
        )
        
        return metrics
    
    def _calc_coverage(self, states):
        """计算覆盖率指标"""
        # 简化的覆盖率评估
        num_sats = len(states)
        avg_altitude = np.mean([np.linalg.norm(pos) for pos in states[:, :3]]) - self.earth_radius
        coverage_area = num_sats * (avg_altitude * np.tan(np.radians(30)))**2  # 假设30度仰角
        earth_area = 4 * np.pi * self.earth_radius**2
        return min(coverage_area / earth_area, 1.0)
    
    def _calc_collision_risk(self, states):
        """计算碰撞风险"""
        # 简化的碰撞风险评估
        positions = states[:, :3]
        distances = []
        
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
                
        if not distances:
            return 0.0
                
        min_dist = min(distances)
        if min_dist < self.collision_threshold:
            return 1.0
        elif min_dist > 2 * self.collision_threshold:
            return 0.0
        else:
            return 1 - (min_dist - self.collision_threshold) / self.collision_threshold
    
    def _calc_cost(self, states):
        """计算成本指标"""
        # 简化的成本评估 (卫星数量和高度)
        num_sats = len(states)
        avg_altitude = np.mean([np.linalg.norm(pos) for pos in states[:, :3]]) - self.earth_radius
        return min(num_sats / 100 + avg_altitude / 2000, 1.0)
    
    def visualize_constellation(self, states, figsize=(10, 8)):
        """
        可视化星座
        
        参数:
        - states: 卫星状态数组
        - figsize: 图表大小
        
        返回:
        - matplotlib图表对象
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地球
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.earth_radius * np.cos(u) * np.sin(v)
        y = self.earth_radius * np.sin(u) * np.sin(v)
        z = self.earth_radius * np.cos(v)
        ax.plot_surface(x, y, z, color='blue', alpha=0.2)
        
        # 绘制卫星
        positions = states[:, :3]
        ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
                  color='red', s=20, label='Satellites')
        
        # 设置图表
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Constellation Visualization')
        ax.legend()
        
        return fig
