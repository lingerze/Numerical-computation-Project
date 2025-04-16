"""
可视化增强模块，提供高级绘图功能和数据可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# 配置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12                       # 设置字体大小

# 3D箭头类，用于显示矢量
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def create_earth_plot(ax, radius=6378.137, resolution=50, alpha=0.3, show_axis=True, cmap='Blues'):
    """
    创建高质量的3D地球图
    
    参数:
    - ax: matplotlib 3D 轴对象
    - radius: 地球半径 (km)
    - resolution: 球体分辨率
    - alpha: 地球透明度
    - show_axis: 是否显示坐标轴
    - cmap: 地球颜色映射
    
    返回:
    - earth_surface: 地球表面对象
    """
    # 创建球体网格
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 创建渐变颜色
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    # 添加简单的地形效果
    terrain = (np.random.rand(x.shape[0], x.shape[1]) * 0.1 + 0.9)
    colors = cmap(terrain)
    
    # 绘制地球
    earth_surface = ax.plot_surface(
        x, y, z, rstride=2, cstride=2, 
        facecolors=colors, alpha=alpha, linewidth=0.5, 
        antialiased=True, shade=True
    )
    
    # 添加赤道平面
    theta = np.linspace(0, 2 * np.pi, 100)
    eq_x = radius * np.cos(theta)
    eq_y = radius * np.sin(theta)
    eq_z = np.zeros_like(theta)
    ax.plot(eq_x, eq_y, eq_z, 'b--', alpha=0.7, linewidth=1)
    
    # 添加经纬线
    for phi in np.linspace(-np.pi/2, np.pi/2, 13)[1:-1]:  # 纬线
        z_circle = radius * np.sin(phi)
        r_circle = radius * np.cos(phi)
        x_circle = r_circle * np.cos(theta)
        y_circle = r_circle * np.sin(theta)
        ax.plot(x_circle, y_circle, np.ones_like(theta) * z_circle, 'gray', alpha=0.2, linewidth=0.5)
    
    for theta_m in np.linspace(0, 2*np.pi, 25)[:-1]:  # 经线
        phi = np.linspace(-np.pi/2, np.pi/2, 50)
        z_line = radius * np.sin(phi)
        r_line = radius * np.cos(phi)
        x_line = r_line * np.cos(theta_m)
        y_line = r_line * np.sin(theta_m)
        ax.plot(x_line, y_line, z_line, 'gray', alpha=0.2, linewidth=0.5)
    
    # 添加坐标轴标签
    if show_axis:
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
    else:
        ax.set_axis_off()
    
    return earth_surface

def plot_orbit_with_evolution(ax, states, colormap='viridis', alpha=0.7, linewidth=1.5, show_direction=True):
    """
    绘制轨道，并使用颜色渐变表示时间演化
    
    参数:
    - ax: matplotlib 3D 轴对象
    - states: 轨道状态历史 (N, 6)，前三个元素是位置
    - colormap: 颜色映射
    - alpha: 轨道线透明度
    - linewidth: 轨道线宽度
    - show_direction: 是否显示轨道方向箭头
    
    返回:
    - line: 轨道线对象
    """
    # 提取位置数据
    positions = states[:, :3] / 1000  # 转换为km
    
    # 创建归一化的色彩映射
    norm = plt.Normalize(0, len(positions)-1)
    cmap = plt.get_cmap(colormap)
    colors = cmap(norm(range(len(positions))))
    
    # 创建线段集合
    points = positions.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3D(segments, colors=colors, alpha=alpha, linewidth=linewidth)
    
    # 添加到图形
    ax.add_collection3d(lc)
    
    # 添加轨道方向指示
    if show_direction and len(positions) > 10:
        arrow_indices = np.linspace(0, len(positions)-1, 8, dtype=int)
        for i in arrow_indices[:-1]:
            # 计算方向向量
            dir_vec = positions[i+1] - positions[i]
            length = np.linalg.norm(dir_vec)
            if length > 0:
                dir_vec = dir_vec / length * min(length * 0.8, 200)  # 缩放箭头长度
                arrow = Arrow3D([positions[i, 0], positions[i, 0] + dir_vec[0]],
                                [positions[i, 1], positions[i, 1] + dir_vec[1]],
                                [positions[i, 2], positions[i, 2] + dir_vec[2]],
                                mutation_scale=10, lw=1.5, arrowstyle='-|>', color=colors[i])
                ax.add_artist(arrow)
    
    return lc

# 3D线集合类，用于创建颜色渐变的轨道线
class Line3D(LineCollection):
    def __init__(self, segments, **kwargs):
        super().__init__(segments, **kwargs)
    
    def do_3d_projection(self, renderer=None):
        segments_2d = []
        for segment in self._segments3d:
            xs, ys, zs = proj3d.proj_transform(segment[:, 0], segment[:, 1], segment[:, 2], self.axes.M)
            segments_2d.append(np.column_stack([xs, ys]))
        
        self.set_segments(segments_2d)
        return np.min(zs)

def create_heat_map(data, title, cmap='viridis', ax=None, colorbar=True):
    """创建热图"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, cmap=cmap, annot=True, fmt=".2f", ax=ax, cbar=colorbar)
    ax.set_title(title)
    
    return ax

def plot_coverage_map(ax, lat_grid, lon_grid, coverage_data, earth_radius=6378.137, 
                      title="全球覆盖率分布", cmap='viridis'):
    """
    在3D地球上绘制覆盖率热图
    
    参数:
    - ax: matplotlib 3D 轴对象
    - lat_grid: 纬度网格
    - lon_grid: 经度网格
    - coverage_data: 覆盖率数据 (lat, lon)
    - earth_radius: 地球半径 (km)
    - title: 图表标题
    - cmap: 颜色映射
    """
    # 创建地球
    create_earth_plot(ax, radius=earth_radius, alpha=0.3)
    
    # 将覆盖数据映射到颜色
    norm = plt.Normalize(0, 1)
    cmap = plt.get_cmap(cmap)
    
    # 创建网格点
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            if coverage_data[i, j] > 0:
                # 转换为3D坐标
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                
                # 计算表面点位置 (稍微抬高以便可见)
                elevation = 1.02  # 表面上方2%
                x = earth_radius * elevation * np.cos(lat_rad) * np.cos(lon_rad)
                y = earth_radius * elevation * np.cos(lat_rad) * np.sin(lon_rad)
                z = earth_radius * elevation * np.sin(lat_rad)
                
                # 绘制点
                color = cmap(norm(coverage_data[i, j]))
                ax.scatter(x, y, z, color=color, s=20, alpha=0.8)
    
    # 添加色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('覆盖率')
    
    ax.set_title(title)
    
    return ax

def plot_orbit_elements_evolution(kepler_elements_history, time_days, fig=None):
    """
    绘制轨道根数随时间的演化
    
    参数:
    - kepler_elements_history: 开普勒轨道要素历史 (N, 6)
    - time_days: 时间数组 (天)
    
    返回:
    - fig: matplotlib 图形对象
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 10))
    
    element_names = ['半长轴 (km)', '偏心率', '倾角 (deg)', 
                     '升交点赤经 (deg)', '近地点幅角 (deg)', '平近点角 (deg)']
    
    # 转换单位
    kepler_converted = kepler_elements_history.copy()
    kepler_converted[:, 0] = kepler_converted[:, 0] / 1000  # 转换为km
    kepler_converted[:, 2:] = np.degrees(kepler_converted[:, 2:])  # 转换为度
    
    # 创建子图
    for i, name in enumerate(element_names):
        ax = fig.add_subplot(3, 2, i+1)
        ax.plot(time_days, kepler_converted[:, i], 'b-', linewidth=1.5)
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel(name)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加趋势线
        if len(time_days) > 1:
            z = np.polyfit(time_days, kepler_converted[:, i], 1)
            p = np.poly1d(z)
            ax.plot(time_days, p(time_days), "r--", alpha=0.8, 
                    label=f"趋势: {z[0]:.2e}/天")
            ax.legend()
    
    fig.tight_layout()
    
    return fig

def plot_error_comparison(results, fig=None, plot_energy=True, plot_momentum=True):
    """
    绘制不同数值方法的误差比较图
    
    参数:
    - results: 误差分析结果字典
    - fig: matplotlib 图形对象
    - plot_energy: 是否绘制能量误差
    - plot_momentum: 是否绘制角动量误差
    
    返回:
    - fig: matplotlib 图形对象
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    gs = fig.add_gridspec(2, 2)
    
    # 绘制计算时间对比
    ax1 = fig.add_subplot(gs[0, 0])
    methods = []
    times = []
    errors = []
    
    for name, data in results.items():
        if name != "reference":
            methods.append(name)
            times.append(data['time'])
            errors.append(data['error'])
    
    ax1.bar(methods, times, color='skyblue')
    ax1.set_ylabel('计算时间 (秒)')
    ax1.set_title('不同方法计算时间对比')
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 绘制误差对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(methods, errors, color='salmon')
    ax2.set_ylabel('相对误差')
    ax2.set_title('不同方法精度对比')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 绘制能量和角动量误差
    if plot_energy or plot_momentum:
        ax3 = fig.add_subplot(gs[1, :])
        
        for i, (name, data) in enumerate(results.items()):
            if name != "reference":
                solution = data['solution']
                
                if plot_energy:
                    from numerical_methods import ErrorAnalysis
                    energy_error = ErrorAnalysis.energy_conservation_error(solution, 3.986004418e14)
                    ax3.plot(data['time_points'] / 86400, energy_error, 
                             label=f"{name} - 能量", linestyle='-', alpha=0.7)
                
                if plot_momentum:
                    from numerical_methods import ErrorAnalysis
                    momentum_error = ErrorAnalysis.angular_momentum_error(solution)
                    ax3.plot(data['time_points'] / 86400, momentum_error, 
                             label=f"{name} - 角动量", linestyle='--', alpha=0.7)
        
        ax3.set_xlabel('时间 (天)')
        ax3.set_ylabel('相对误差')
        ax3.set_title('轨道保守量随时间的误差演化')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
    
    fig.tight_layout()
    
    return fig

class InteractivePlotWindow:
    """创建交互式绘图窗口的类"""
    
    def __init__(self, parent, title="交互式绘图"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1000x800")
        
        # 创建框架
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 创建图形区域
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 默认添加一个轴
        self.ax = self.fig.add_subplot(111)
    
    def clear_plot(self):
        """清空图形"""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()
    
    def create_3d_plot(self):
        """创建3D图形"""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas.draw()
        return self.ax
        
    def add_control_button(self, text, command):
        """添加控制按钮"""
        btn = tk.Button(self.control_frame, text=text, command=command)
        btn.pack(side=tk.LEFT, padx=5, pady=2)
        return btn 