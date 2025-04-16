"""
高级摄动力模型，包括扩展地球引力场模型、精确大气阻力模型和太阳辐射压模型
"""

import numpy as np
from scipy.special import lpmv
import datetime

class GravityField:
    """高阶地球引力场模型"""
    
    def __init__(self, max_degree=4):
        """
        初始化地球引力场模型
        
        参数:
        - max_degree: 引力场展开的最大阶数
        """
        self.max_degree = max_degree
        self.mu = 3.986004418e14  # 地球引力常数 (m^3/s^2)
        self.re = 6378.137e3      # 地球半径 (m)
        
        # EGM2008 引力场系数 (简化版，仅包含部分主要项)
        # 参考: https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
        self.C = np.zeros((max_degree+1, max_degree+1))
        self.S = np.zeros((max_degree+1, max_degree+1))
        
        # J2-J4 系数 (归一化后)
        self.C[2, 0] = -0.484165143790815e-03  # J2
        self.C[3, 0] = 0.957161207093473e-06   # J3
        self.C[4, 0] = 0.539965866638991e-06   # J4
        
        # 其他主要谐波系数
        self.C[2, 2] = 0.243938357328313e-05
        self.S[2, 2] = -0.140027370385934e-05
        self.C[3, 1] = 0.203107445066171e-05
        self.S[3, 1] = 0.248747042764235e-06
        self.C[3, 3] = 0.905660357698480e-07
        self.S[3, 3] = 0.218498343582072e-06
    
    def compute_acceleration(self, r_vec):
        """
        计算地球引力场加速度
        
        参数:
        - r_vec: 位置向量 (x, y, z) [m]
        
        返回:
        - 加速度向量 [m/s^2]
        """
        # 转换为球坐标
        r = np.linalg.norm(r_vec)
        if r == 0:
            return np.zeros(3)
            
        x, y, z = r_vec
        
        # 地心纬度和经度
        phi = np.arcsin(z / r)
        lambda_ = np.arctan2(y, x)
        
        # 计算勒让德多项式
        P = np.zeros((self.max_degree+1, self.max_degree+1))
        dP = np.zeros((self.max_degree+1, self.max_degree+1))
        
        sin_phi = np.sin(phi)
        
        for n in range(self.max_degree+1):
            for m in range(n+1):
                # 计算缔合勒让德函数
                P[n, m] = lpmv(m, n, sin_phi)
                
                # 计算导数
                if n > 0 and n > m:
                    if abs(sin_phi) != 1.0:  # 避免除以零
                        dP[n, m] = (n * sin_phi * P[n, m] - (n + m) * P[n-1, m]) / (1 - sin_phi**2)
                    else:
                        # 极点处的处理
                        dP[n, m] = 0.5 * n * (n + 1) * P[n, m]
        
        # 计算加速度
        a_r = 0.0  # 径向加速度
        a_phi = 0.0  # 纬度方向加速度
        a_lambda = 0.0  # 经度方向加速度
        
        for n in range(2, self.max_degree+1):
            for m in range(n+1):
                # 系数计算
                factor = (self.mu / r**2) * (self.re / r)**n * (n + 1)
                
                # 三角函数项
                cos_m_lambda = np.cos(m * lambda_)
                sin_m_lambda = np.sin(m * lambda_)
                
                # 引力势的导数
                V_r = factor * sum(P[n, m] * (self.C[n, m] * cos_m_lambda + self.S[n, m] * sin_m_lambda) for m in range(n+1))
                V_phi = factor * (r / (n + 1)) * sum(dP[n, m] * (self.C[n, m] * cos_m_lambda + self.S[n, m] * sin_m_lambda) for m in range(n+1))
                V_lambda = factor * (r / (n + 1)) * sum(m * P[n, m] * (self.S[n, m] * cos_m_lambda - self.C[n, m] * sin_m_lambda) for m in range(1, n+1))
                
                a_r -= V_r
                a_phi -= V_phi
                a_lambda -= V_lambda
        
        # 转换回笛卡尔坐标系
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_lambda = np.cos(lambda_)
        sin_lambda = np.sin(lambda_)
        
        a_x = a_r * cos_phi * cos_lambda - a_phi * sin_phi * cos_lambda - a_lambda * sin_lambda
        a_y = a_r * cos_phi * sin_lambda - a_phi * sin_phi * sin_lambda + a_lambda * cos_lambda
        a_z = a_r * sin_phi + a_phi * cos_phi
        
        return np.array([a_x, a_y, a_z])
    
    def J2_only_acceleration(self, r_vec):
        """
        仅计算J2摄动加速度（简化版，用于比较）
        
        参数:
        - r_vec: 位置向量 (x, y, z) [m]
        
        返回:
        - J2摄动加速度 [m/s^2]
        """
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)
        
        # J2扰动
        J2 = 1.08263e-3
        factor = 1.5 * J2 * self.mu * self.re**2 / r**5
        
        ax = factor * x * (5 * z**2 / r**2 - 1)
        ay = factor * y * (5 * z**2 / r**2 - 1)
        az = factor * z * (5 * z**2 / r**2 - 3)
        
        return np.array([ax, ay, az])

class AtmosphericModel:
    """高级大气密度模型"""
    
    def __init__(self):
        """初始化大气模型参数"""
        # NRLMSISE-00 模型的简化参考参数
        self.h_ref = np.array([
            0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 
            130, 140, 150, 180, 200, 250, 300, 350, 400, 450, 
            500, 600, 700, 800, 900, 1000
        ]) * 1000  # 转换为米
        
        self.rho_ref = np.array([
            1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 
            8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 
            8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 
            2.418e-11, 9.518e-12, 3.727e-12, 1.585e-12, 6.967e-13, 1.454e-13, 
            3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15
        ])
        
        # 外推参数
        self.scale_heights = np.zeros(len(self.h_ref) - 1)
        for i in range(len(self.h_ref) - 1):
            dh = self.h_ref[i+1] - self.h_ref[i]
            ratio = self.rho_ref[i+1] / self.rho_ref[i]
            if ratio > 0:
                self.scale_heights[i] = -dh / np.log(ratio)
            else:
                self.scale_heights[i] = dh  # 防止出现负数或零
    
    def density(self, h, geo_lat=0, geo_lon=0, day_of_year=None, solar_flux=150, ap=4):
        """
        计算给定高度的大气密度
        
        参数:
        - h: 高度 [m]
        - geo_lat: 地理纬度 [rad]
        - geo_lon: 地理经度 [rad]
        - day_of_year: 一年中的日期 (1-366)，None表示当前日期
        - solar_flux: F10.7太阳辐射流量指数 (默认150)
        - ap: 地磁活动指数 (默认4)
        
        返回:
        - 大气密度 [kg/m^3]
        """
        # 获取当前日期（如果未指定）
        if day_of_year is None:
            day_of_year = datetime.datetime.now().timetuple().tm_yday
        
        # 基本密度插值
        if h <= self.h_ref[0]:
            rho = self.rho_ref[0]
        elif h >= self.h_ref[-1]:
            # 外推密度
            h_diff = h - self.h_ref[-1]
            scale_h = self.scale_heights[-1]
            rho = self.rho_ref[-1] * np.exp(-h_diff / scale_h)
        else:
            # 找到插值区间
            idx = np.searchsorted(self.h_ref, h) - 1
            
            # 使用区间对应的标高插值
            h_diff = h - self.h_ref[idx]
            scale_h = self.scale_heights[idx]
            rho = self.rho_ref[idx] * np.exp(-h_diff / scale_h)
        
        # 应用地理位置变化因子 (昼夜变化和季节变化)
        # 计算当地太阳时
        solar_time = (geo_lon / (2 * np.pi) * 24 + datetime.datetime.now().hour) % 24
        
        # 计算太阳赤纬
        day_angle = 2 * np.pi * (day_of_year - 1) / 365
        solar_declination = 0.409 * np.sin(day_angle - 1.39)
        
        # 计算地方时角
        hour_angle = (solar_time - 12) * np.pi / 12
        
        # 计算太阳天顶角
        cos_zenith = np.sin(geo_lat) * np.sin(solar_declination) + \
                     np.cos(geo_lat) * np.cos(solar_declination) * np.cos(hour_angle)
        
        # 昼夜变化因子 (白天密度较高)
        day_night_factor = 1.0 + 0.3 * max(0, cos_zenith)
        
        # 季节变化因子 (夏季高于冬季)
        season_factor = 1.0 + 0.2 * np.sin(day_angle)
        
        # 太阳活动和地磁活动影响
        # F10.7的正常值范围约为70-250
        solar_activity_factor = 1.0 + 0.4 * (solar_flux - 150) / 100
        
        # Ap指数的正常值范围约为0-400
        geo_activity_factor = 1.0 + 0.2 * min(1.0, ap / 40)
        
        # 组合所有因子（高度超过150km时考虑太阳活动影响增强）
        if h > 150000:  # 150km
            solar_factor = solar_activity_factor * (1.0 + (h - 150000) / 450000)  # 高度越高影响越大
        else:
            solar_factor = 1.0
            
        final_density = rho * day_night_factor * season_factor * solar_factor * geo_activity_factor
        
        return final_density
    
    def compute_drag_acceleration(self, r_vec, v_vec, A_m_ratio, cd=2.2, earth_rotation_rate=7.2921159e-5):
        """
        计算大气阻力加速度
        
        参数:
        - r_vec: 位置向量 [m]
        - v_vec: 速度向量 [m/s]
        - A_m_ratio: 面积质量比 [m^2/kg]
        - cd: 阻力系数
        - earth_rotation_rate: 地球自转角速度 [rad/s]
        
        返回:
        - 阻力加速度向量 [m/s^2]
        """
        # 计算高度
        r = np.linalg.norm(r_vec)
        earth_radius = 6378.137e3
        h = r - earth_radius
        
        # 计算地心纬度和经度
        z = r_vec[2]
        geo_lat = np.arcsin(z / r)
        geo_lon = np.arctan2(r_vec[1], r_vec[0])
        
        # 计算大气密度
        rho = self.density(h, geo_lat, geo_lon)
        
        # 计算地球自转速度
        omega_earth = np.array([0, 0, earth_rotation_rate])
        v_atm = np.cross(omega_earth, r_vec)  # 大气速度
        v_rel = v_vec - v_atm  # 相对速度
        v_rel_mag = np.linalg.norm(v_rel)
        
        # 阻力加速度
        if v_rel_mag > 0:
            a_drag = -0.5 * rho * v_rel_mag * cd * A_m_ratio * v_rel
        else:
            a_drag = np.zeros(3)
        
        return a_drag

class SolarRadiationPressure:
    """高级太阳辐射压力模型"""
    
    def __init__(self):
        """初始化太阳辐射压力模型参数"""
        self.solar_flux = 1361.0  # 太阳辐射流量 [W/m^2]
        self.c = 299792458.0     # 光速 [m/s]
        self.AU = 149597870700.0  # 天文单位 [m]
        self.earth_radius = 6378137.0  # 地球半径 [m]
        
        # 地球公转参数
        self.earth_orbital_eccentricity = 0.0167
        self.earth_orbital_period = 365.256363004 * 86400.0  # 恒星年 [s]
        self.perihelion_date = datetime.datetime(2022, 1, 4)  # 近日点日期 (每年略有变化)
    
    def get_sun_direction(self, epoch=None):
        """
        计算太阳方向单位向量 (ECEF坐标系)
        
        参数:
        - epoch: 日期时间对象，默认为当前时间
        
        返回:
        - 太阳方向单位向量
        """
        if epoch is None:
            epoch = datetime.datetime.now()
        
        # 计算一年中的日期
        start_of_year = datetime.datetime(epoch.year, 1, 1)
        days = (epoch - start_of_year).total_seconds() / 86400.0
        
        # 计算地球在轨道上的位置
        days_since_perihelion = (epoch - self.perihelion_date).total_seconds() / 86400.0
        days_since_perihelion = days_since_perihelion % (self.earth_orbital_period / 86400.0)
        
        # 计算地球在轨道上的真近点角
        mean_anomaly = 2 * np.pi * days_since_perihelion / (self.earth_orbital_period / 86400.0)
        
        # 解开普勒方程（简化计算，忽略迭代）
        eccentric_anomaly = mean_anomaly + self.earth_orbital_eccentricity * np.sin(mean_anomaly)
        
        # 计算真近点角
        true_anomaly = 2 * np.arctan(np.sqrt((1 + self.earth_orbital_eccentricity) / 
                                           (1 - self.earth_orbital_eccentricity)) * 
                                   np.tan(eccentric_anomaly / 2))
        
        # 计算太阳黄道坐标
        lambda_sun = true_anomaly + np.pi  # 太阳黄经
        
        # 黄道倾角
        epsilon = 23.439 * np.pi / 180.0
        
        # 转换到赤道坐标系
        sun_x = np.cos(lambda_sun)
        sun_y = np.cos(epsilon) * np.sin(lambda_sun)
        sun_z = np.sin(epsilon) * np.sin(lambda_sun)
        
        # 格林威治恒星时 (简化)
        T0 = (epoch - datetime.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 / 36525.0
        GMST = 280.46061837 + 360.98564736629 * T0
        GMST = (GMST % 360) * np.pi / 180.0
        
        # 考虑地球自转
        hour_angle = epoch.hour + epoch.minute / 60 + epoch.second / 3600
        rotation_angle = hour_angle * 15 * np.pi / 180.0 + GMST
        
        # 转换到ECEF坐标系
        x_ecef = sun_x * np.cos(rotation_angle) + sun_y * np.sin(rotation_angle)
        y_ecef = -sun_x * np.sin(rotation_angle) + sun_y * np.cos(rotation_angle)
        z_ecef = sun_z
        
        # 归一化
        sun_direction = np.array([x_ecef, y_ecef, z_ecef])
        sun_direction = sun_direction / np.linalg.norm(sun_direction)
        
        return sun_direction
    
    def compute_shadow_factor(self, r_vec, sun_direction):
        """
        计算阴影因子 (0为完全阴影，1为完全阳光)
        
        参数:
        - r_vec: 卫星位置向量 [m]
        - sun_direction: 太阳方向单位向量
        
        返回:
        - 阴影因子 [0-1]
        """
        r = np.linalg.norm(r_vec)
        cos_angle = np.dot(r_vec, sun_direction) / r
        
        # 计算卫星到地球-太阳连线的距离
        sin_angle = np.sqrt(1 - cos_angle**2)
        d = r * sin_angle
        
        # 检查是否在几何阴影区域
        if cos_angle < 0 and d < self.earth_radius:
            # 计算穿过大气层路径
            atm_height = 150000.0  # 假设大气层高度为150km
            l_atm = 2 * np.sqrt((self.earth_radius + atm_height)**2 - d**2)
            
            # 简化的大气衰减模型
            attenuation = np.exp(-0.5 * l_atm / 50000.0)  # 假设大气透光系数
            
            # 公转角度
            if d < self.earth_radius - 10000:  # 完全在几何阴影中
                return attenuation * 0.01  # 保留少量散射光
            else:
                # 半影区域
                penumbra_factor = (d - (self.earth_radius - 10000)) / 10000
                return attenuation * (0.01 + 0.99 * penumbra_factor)
        
        return 1.0
    
    def compute_srp_acceleration(self, r_vec, A_m_ratio, cr=1.5, epoch=None):
        """
        计算太阳辐射压力加速度
        
        参数:
        - r_vec: 位置向量 [m]
        - A_m_ratio: 面积质量比 [m^2/kg]
        - cr: 反射系数 (1.0为完全吸收，2.0为完全反射)
        - epoch: 日期时间对象，默认为当前时间
        
        返回:
        - 太阳辐射压力加速度 [m/s^2]
        """
        # 获取太阳方向
        sun_direction = self.get_sun_direction(epoch)
        
        # 计算阴影因子
        shadow_factor = self.compute_shadow_factor(r_vec, sun_direction)
        
        # 计算地日距离
        if epoch is None:
            epoch = datetime.datetime.now()
        
        # 计算一年中的日期比例
        day_of_year = epoch.timetuple().tm_yday
        day_ratio = day_of_year / 365.25
        
        # 计算地日距离变化 (简化版)
        r_ratio = 1 - self.earth_orbital_eccentricity * np.cos(2 * np.pi * day_ratio)
        
        # 调整太阳辐射流量
        solar_flux_adjusted = self.solar_flux / (r_ratio**2)
        
        # 辐射压力计算
        P_solar = solar_flux_adjusted / self.c
        
        # 加速度计算
        a_srp = -shadow_factor * P_solar * cr * A_m_ratio * sun_direction
        
        return a_srp

class LunarSolarGravity:
    """月球和太阳引力摄动"""
    
    def __init__(self):
        """初始化月球和太阳引力参数"""
        self.mu_moon = 4.9048695e12     # 月球引力常数 [m^3/s^2]
        self.mu_sun = 1.32712440018e20  # 太阳引力常数 [m^3/s^2]
        
        self.earth_moon_distance = 384400e3  # 地月平均距离 [m]
        self.earth_sun_distance = 149597870700.0  # 地日平均距离 [m]
        
        self.moon_orbital_period = 27.321661 * 86400.0  # 恒星月 [s]
        self.srp_model = SolarRadiationPressure()  # 用于获取太阳方向
    
    def get_moon_position(self, epoch=None):
        """
        计算月球位置 (ECEF坐标系)
        
        参数:
        - epoch: 日期时间对象，默认为当前时间
        
        返回:
        - 月球位置向量 [m]
        """
        if epoch is None:
            epoch = datetime.datetime.now()
        
        # 简化月球轨道模型，假设在赤道平面内圆形轨道
        # 实际应用中应使用JPL星历或更精确的模型
        
        # 计算月球在轨道上的位置角度 (简化)
        t = epoch.timestamp()
        moon_angle = 2 * np.pi * (t % self.moon_orbital_period) / self.moon_orbital_period
        
        # 月球在轨道平面的位置
        x_orbit = self.earth_moon_distance * np.cos(moon_angle)
        y_orbit = self.earth_moon_distance * np.sin(moon_angle)
        z_orbit = 0.0
        
        # 考虑轨道倾角 (月球轨道倾角约5.14°)
        inclination = 5.14 * np.pi / 180.0
        
        # 简单旋转
        x = x_orbit
        y = y_orbit * np.cos(inclination)
        z = y_orbit * np.sin(inclination)
        
        # 考虑月球轨道的升交点位置 (简化)
        raan = 0.0  # 实际应随时间变化
        
        # 旋转到ECEF坐标系 (极其简化的模型)
        # 实际应用中需要考虑地球自转、交点回归等因素
        
        return np.array([x, y, z])
    
    def compute_third_body_acceleration(self, r_vec, epoch=None):
        """
        计算第三体引力加速度
        
        参数:
        - r_vec: 卫星位置向量 [m]
        - epoch: 日期时间对象，默认为当前时间
        
        返回:
        - 月球和太阳引力加速度 [m/s^2]
        """
        if epoch is None:
            epoch = datetime.datetime.now()
        
        # 获取月球位置
        r_moon = self.get_moon_position(epoch)
        
        # 计算卫星到月球的矢量
        r_sat_to_moon = r_moon - r_vec
        
        # 月球引力加速度 (考虑直接项和间接项)
        r_moon_mag = np.linalg.norm(r_moon)
        r_sat_to_moon_mag = np.linalg.norm(r_sat_to_moon)
        
        # 直接项
        a_moon_direct = self.mu_moon * r_sat_to_moon / r_sat_to_moon_mag**3
        
        # 间接项 (地球-月球系统的加速度)
        a_moon_indirect = -self.mu_moon * r_moon / r_moon_mag**3
        
        # 总月球加速度
        a_moon = a_moon_direct + a_moon_indirect
        
        # 获取太阳方向和距离
        sun_dir = self.srp_model.get_sun_direction(epoch)
        r_sun = sun_dir * self.earth_sun_distance
        
        # 计算卫星到太阳的矢量
        r_sat_to_sun = r_sun - r_vec
        
        # 太阳引力加速度
        r_sun_mag = np.linalg.norm(r_sun)
        r_sat_to_sun_mag = np.linalg.norm(r_sat_to_sun)
        
        # 直接项
        a_sun_direct = self.mu_sun * r_sat_to_sun / r_sat_to_sun_mag**3
        
        # 间接项
        a_sun_indirect = -self.mu_sun * r_sun / r_sun_mag**3
        
        # 总太阳加速度
        a_sun = a_sun_direct + a_sun_indirect
        
        # 组合加速度
        a_total = a_moon + a_sun
        
        return a_total 