"""
数值方法模块，提供各种高精度积分方法和误差分析工具
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp

class NumericalIntegrators:
    """提供多种数值积分方法"""
    
    @staticmethod
    def euler_method(func, t_span, y0, steps):
        """
        简单欧拉法
        
        参数:
        - func: 微分方程函数，形式为 func(t, y)
        - t_span: 时间范围 (t_start, t_end)
        - y0: 初始状态
        - steps: 时间步数
        
        返回:
        - t: 时间点数组
        - y: 状态历史数组
        """
        t = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        y = np.zeros((steps, len(y0)))
        y[0] = y0
        
        for i in range(1, steps):
            y[i] = y[i-1] + dt * func(t[i-1], y[i-1])
            
        return t, y
    
    @staticmethod
    def rk4_method(func, t_span, y0, steps):
        """
        经典四阶龙格-库塔方法
        
        参数:
        - func: 微分方程函数，形式为 func(t, y)
        - t_span: 时间范围 (t_start, t_end)
        - y0: 初始状态
        - steps: 时间步数
        
        返回:
        - t: 时间点数组
        - y: 状态历史数组
        """
        t = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        y = np.zeros((steps, len(y0)))
        y[0] = y0
        
        for i in range(1, steps):
            k1 = func(t[i-1], y[i-1])
            k2 = func(t[i-1] + dt/2, y[i-1] + dt/2 * k1)
            k3 = func(t[i-1] + dt/2, y[i-1] + dt/2 * k2)
            k4 = func(t[i-1] + dt, y[i-1] + dt * k3)
            
            y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
        return t, y
    
    @staticmethod
    def verlet_method(func, t_span, y0, steps):
        """
        速度Verlet积分法，适用于物理系统
        
        参数:
        - func: 微分方程函数，形式为 func(t, y)
        - t_span: 时间范围 (t_start, t_end)
        - y0: 初始状态，应当包含位置和速度
        - steps: 时间步数
        
        返回:
        - t: 时间点数组
        - y: 状态历史数组
        """
        t = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        n = len(y0)
        half_n = n // 2  # 假设状态向量前半是位置，后半是速度
        
        y = np.zeros((steps, n))
        y[0] = y0
        
        # 使用RK4计算第一步
        k1 = func(t[0], y[0])
        k2 = func(t[0] + dt/2, y[0] + dt/2 * k1)
        k3 = func(t[0] + dt/2, y[0] + dt/2 * k2)
        k4 = func(t[0] + dt, y[0] + dt * k3)
        y[1] = y[0] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # 对剩余步骤使用Verlet算法
        for i in range(2, steps):
            # 提取当前位置、速度和加速度
            pos_curr = y[i-1, :half_n]
            vel_curr = y[i-1, half_n:]
            acc_curr = func(t[i-1], y[i-1])[half_n:]  # 仅取加速度部分
            
            # 更新位置
            pos_next = pos_curr + vel_curr * dt + 0.5 * acc_curr * dt**2
            
            # 计算中点速度
            vel_mid = vel_curr + 0.5 * acc_curr * dt
            
            # 用更新后的位置计算下一个加速度
            state_tmp = np.concatenate((pos_next, vel_mid))
            acc_next = func(t[i], state_tmp)[half_n:]
            
            # 更新速度
            vel_next = vel_curr + 0.5 * (acc_curr + acc_next) * dt
            
            # 存储结果
            y[i, :half_n] = pos_next
            y[i, half_n:] = vel_next
            
        return t, y
    
    @staticmethod
    def symplectic_euler(func, t_span, y0, steps):
        """
        辛欧拉方法，适用于哈密顿系统
        
        参数:
        - func: 微分方程函数，形式为 func(t, y)
        - t_span: 时间范围 (t_start, t_end)
        - y0: 初始状态，应当包含位置和速度
        - steps: 时间步数
        
        返回:
        - t: 时间点数组
        - y: 状态历史数组
        """
        t = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        n = len(y0)
        half_n = n // 2  # 假设状态向量前半是位置，后半是速度
        
        y = np.zeros((steps, n))
        y[0] = y0
        
        for i in range(1, steps):
            # 当前状态
            pos_curr = y[i-1, :half_n]
            vel_curr = y[i-1, half_n:]
            
            # 计算加速度
            acc_curr = func(t[i-1], y[i-1])[half_n:]
            
            # 先更新速度
            vel_next = vel_curr + dt * acc_curr
            
            # 再用新速度更新位置
            pos_next = pos_curr + dt * vel_next
            
            # 存储结果
            y[i, :half_n] = pos_next
            y[i, half_n:] = vel_next
            
        return t, y


class ErrorAnalysis:
    """提供各种误差分析工具"""
    
    @staticmethod
    def relative_error(reference, approximation):
        """
        计算相对误差
        
        参数:
        - reference: 参考解
        - approximation: 近似解
        
        返回:
        - 相对误差的均方根
        """
        # 确保两个解的形状相等
        if reference.shape != approximation.shape:
            # 进行插值以匹配点数
            if len(reference) > len(approximation):
                indices = np.linspace(0, len(approximation)-1, len(reference)).astype(int)
                approximation_matched = approximation[indices]
                reference_matched = reference
            else:
                indices = np.linspace(0, len(reference)-1, len(approximation)).astype(int)
                reference_matched = reference[indices]
                approximation_matched = approximation
        else:
            reference_matched = reference
            approximation_matched = approximation
        
        # 计算欧几里德距离
        error = np.linalg.norm(reference_matched - approximation_matched, axis=1)
        reference_norm = np.linalg.norm(reference_matched, axis=1)
        
        # 避免除以零
        mask = reference_norm > 1e-10
        relative_error = np.zeros_like(error)
        relative_error[mask] = error[mask] / reference_norm[mask]
        
        # 返回均方根误差
        return np.sqrt(np.mean(relative_error**2))
    
    @staticmethod
    def energy_conservation_error(states, mu):
        """
        计算能量守恒误差
        
        参数:
        - states: 状态历史，每行形如 [x, y, z, vx, vy, vz]
        - mu: 引力常数
        
        返回:
        - 能量随时间的相对误差数组
        """
        # 计算初始能量
        r0 = np.linalg.norm(states[0, :3])
        v0 = np.linalg.norm(states[0, 3:])
        e0 = v0**2/2 - mu/r0  # 单位能量
        
        # 计算所有时间点的能量
        r = np.linalg.norm(states[:, :3], axis=1)
        v = np.linalg.norm(states[:, 3:], axis=1)
        e = v**2/2 - mu/r
        
        # 计算相对误差
        error = (e - e0) / abs(e0)
        
        return error
    
    @staticmethod
    def angular_momentum_error(states):
        """
        计算角动量守恒误差
        
        参数:
        - states: 状态历史，每行形如 [x, y, z, vx, vy, vz]
        
        返回:
        - 角动量随时间的相对误差数组
        """
        # 计算初始角动量
        r0 = states[0, :3]
        v0 = states[0, 3:]
        h0 = np.cross(r0, v0)
        h0_norm = np.linalg.norm(h0)
        
        # 计算所有时间点的角动量
        h = np.zeros((len(states), 3))
        for i in range(len(states)):
            h[i] = np.cross(states[i, :3], states[i, 3:])
        
        h_norm = np.linalg.norm(h, axis=1)
        
        # 计算相对误差
        error = (h_norm - h0_norm) / h0_norm
        
        return error

# 旧代码保留，确保兼容性
class OrbitPropagator:
    """简化的轨道传播器"""
    
    def __init__(self, model='2body'):
        """
        初始化轨道传播器
        
        参数:
        - model: 轨道模型 ('2body' 或 'j2')
        """
        self.model = model
        self.mu = 398600.4418  # 地球引力常数 (km^3/s^2)
        self.J2 = 1.08263e-3    # J2摄动系数
        self.Re = 6378.137      # 地球半径 (km)
        
    def propagate_orbit(self, initial_state, t_span, steps=1000, **kwargs):
        """
        传播轨道
        
        参数:
        - initial_state: 初始状态向量 [x,y,z,vx,vy,vz] (km, km/s)
        - t_span: 时间范围 (t_start, t_end) (秒)
        - steps: 时间步数
        
        返回:
        - 轨道状态历史 (N×6数组)
        """
        t = np.linspace(t_span[0], t_span[1], steps)
        
        if self.model == '2body':
            # 二体问题传播
            orbit_states = odeint(self._two_body_ode, initial_state, t, args=(self.mu,))
        elif self.model == 'j2':
            # J2摄动传播
            orbit_states = odeint(self._j2_ode, initial_state, t, args=(self.mu, self.J2, self.Re))
        else:
            raise ValueError(f"不支持的轨道模型: {self.model}")
            
        return orbit_states
    
    def _two_body_ode(self, state, t, mu):
        """二体问题运动方程"""
        x, y, z, vx, vy, vz = state
        r = np.array([x, y, z])
        r_norm = np.linalg.norm(r)
        
        # 加速度计算
        ax, ay, az = -mu * r / (r_norm ** 3)
        
        return [vx, vy, vz, ax, ay, az]
    
    def _j2_ode(self, state, t, mu, J2, Re):
        """J2摄动运动方程"""
        x, y, z, vx, vy, vz = state
        r = np.array([x, y, z])
        r_norm = np.linalg.norm(r)
        
        # 二体加速度
        a_2body = -mu * r / (r_norm ** 3)
        
        # J2摄动加速度
        k = 1.5 * J2 * mu * Re**2 / (r_norm**5)
        a_j2x = k * (x/r_norm * (5*z**2/r_norm**2 - 1))
        a_j2y = k * (y/r_norm * (5*z**2/r_norm**2 - 1))
        a_j2z = k * (z/r_norm * (5*z**2/r_norm**2 - 3))
        
        a_total = a_2body + np.array([a_j2x, a_j2y, a_j2z])
        
        return [vx, vy, vz, a_total[0], a_total[1], a_total[2]]
