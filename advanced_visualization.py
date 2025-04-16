"""
高级可视化模块 - 提供增强的数据可视化和交互功能

此模块扩展了基本可视化功能，提供了更高级的数据展示方式，
包括3D轨道演化动画、热图分析、多维数据可视化等功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
import io
from PIL import Image, ImageTk
import time

# 配置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12                       # 设置字体大小

class AdvancedVisualizationWindow:
    """高级可视化窗口，支持多种可视化功能"""
    
    def __init__(self, parent, title="高级轨道可视化"):
        """初始化高级可视化窗口"""
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1200x800")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部控制区域
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建底部图形区域
        self.figure_frame = ttk.Frame(self.main_frame)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建图形和画布
        self.fig = Figure(figsize=(12, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.figure_frame)
        self.toolbar.update()
        
        # 创建控制按钮
        self.create_controls()
        
        # 数据存储
        self.orbit_data = {}
        self.animation_obj = None
        self.is_animating = False
        
    def create_controls(self):
        """创建控制按钮"""
        # 可视化类型选择
        ttk.Label(self.control_frame, text="可视化类型:").pack(side=tk.LEFT, padx=5, pady=5)
        
        self.viz_type_var = tk.StringVar(value="3D轨道")
        viz_types = ["3D轨道", "轨道演化", "摄动分析", "星座覆盖", "误差分析", "碰撞概率热图"]
        
        self.viz_type_combo = ttk.Combobox(self.control_frame, textvariable=self.viz_type_var,
                                         values=viz_types, width=15)
        self.viz_type_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.viz_type_combo.bind("<<ComboboxSelected>>", self.on_viz_type_changed)
        
        # 动画控制
        self.anim_btn = ttk.Button(self.control_frame, text="播放动画", command=self.toggle_animation)
        self.anim_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Label(self.control_frame, text="速度:").pack(side=tk.LEFT, padx=5, pady=5)
        self.speed_scale = ttk.Scale(self.control_frame, from_=0.1, to=5.0, 
                                   orient=tk.HORIZONTAL, variable=self.speed_var,
                                   length=100)
        self.speed_scale.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 保存按钮
        self.save_btn = ttk.Button(self.control_frame, text="保存图像", command=self.save_figure)
        self.save_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 重置视图按钮
        self.reset_btn = ttk.Button(self.control_frame, text="重置视图", command=self.reset_view)
        self.reset_btn.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def on_viz_type_changed(self, event):
        """可视化类型改变时的回调"""
        viz_type = self.viz_type_var.get()
        
        # 停止任何正在运行的动画
        self.stop_animation()
        
        # 根据选择的可视化类型更新界面
        if viz_type == "3D轨道":
            self.anim_btn.configure(state=tk.NORMAL)
            self.speed_scale.configure(state=tk.NORMAL)
        elif viz_type == "轨道演化":
            self.anim_btn.configure(state=tk.DISABLED)
            self.speed_scale.configure(state=tk.DISABLED)
        elif viz_type == "摄动分析":
            self.anim_btn.configure(state=tk.DISABLED)
            self.speed_scale.configure(state=tk.DISABLED)
        elif viz_type == "星座覆盖":
            self.anim_btn.configure(state=tk.DISABLED)
            self.speed_scale.configure(state=tk.DISABLED)
        elif viz_type == "误差分析":
            self.anim_btn.configure(state=tk.DISABLED)
            self.speed_scale.configure(state=tk.DISABLED)
        elif viz_type == "碰撞概率热图":
            self.anim_btn.configure(state=tk.DISABLED)
            self.speed_scale.configure(state=tk.DISABLED)
    
    def set_data(self, data_type, data):
        """设置可视化数据"""
        self.orbit_data[data_type] = data
    
    def toggle_animation(self):
        """切换动画状态"""
        if self.is_animating:
            self.stop_animation()
            self.anim_btn.configure(text="播放动画")
        else:
            self.start_animation()
            self.anim_btn.configure(text="停止动画")
    
    def start_animation(self):
        """开始动画"""
        if not self.is_animating and self.viz_type_var.get() == "3D轨道":
            if 'states' in self.orbit_data:
                # 清除图形
                self.fig.clear()
                
                # 创建3D轴
                ax = self.fig.add_subplot(111, projection='3d')
                
                # 准备数据
                states = self.orbit_data['states']
                times = self.orbit_data.get('times', np.linspace(0, 1, len(states)))
                
                # 绘制地球
                from visualization_enhancements import create_earth_plot
                create_earth_plot(ax, radius=6378.137, alpha=0.3)
                
                # 设置轴标签和标题
                ax.set_xlabel('X (km)')
                ax.set_ylabel('Y (km)')
                ax.set_zlabel('Z (km)')
                ax.set_title('卫星轨道动画')
                
                # 初始化轨迹线和散点
                line, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.5)
                point, = ax.plot([], [], [], 'ro', markersize=8)
                
                # 设置轴范围
                max_range = np.max(np.abs(states[:, :3])) / 1000 * 1.1  # 转换为km并增加10%的余量
                ax.set_xlim3d([-max_range, max_range])
                ax.set_ylim3d([-max_range, max_range])
                ax.set_zlim3d([-max_range, max_range])
                
                # 添加时间标签
                time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
                
                # 初始化函数
                def init():
                    line.set_data([], [])
                    line.set_3d_properties([])
                    point.set_data([], [])
                    point.set_3d_properties([])
                    time_text.set_text('')
                    return line, point, time_text
                
                # 更新函数
                def update(frame):
                    i = int(frame)
                    # 绘制已经走过的轨迹
                    x = states[:i+1, 0] / 1000  # 转换为km
                    y = states[:i+1, 1] / 1000
                    z = states[:i+1, 2] / 1000
                    line.set_data(x, y)
                    line.set_3d_properties(z)
                    
                    # 更新点的位置
                    point.set_data([x[-1]], [y[-1]])
                    point.set_3d_properties([z[-1]])
                    
                    # 更新时间标签
                    time_text.set_text(f'时间: {times[i]:.2f} 秒')
                    
                    return line, point, time_text
                
                # 创建动画
                frames = len(states)
                interval = 50 / self.speed_var.get()  # 默认20fps，根据速度调整
                self.animation_obj = animation.FuncAnimation(
                    self.fig, update, frames=frames, 
                    interval=interval, blit=True, init_func=init
                )
                
                self.canvas.draw()
                self.is_animating = True
            else:
                tk.messagebox.showwarning("警告", "没有可用的轨道数据进行动画显示")
    
    def stop_animation(self):
        """停止动画"""
        if self.animation_obj is not None:
            self.animation_obj.event_source.stop()
            self.animation_obj = None
            self.is_animating = False
    
    def reset_view(self):
        """重置视图"""
        viz_type = self.viz_type_var.get()
        
        # 停止动画
        self.stop_animation()
        
        # 根据当前可视化类型重新绘制
        if viz_type == "3D轨道":
            self.plot_3d_orbit()
        elif viz_type == "轨道演化":
            self.plot_orbit_evolution()
        elif viz_type == "摄动分析":
            self.plot_perturbation_analysis()
        elif viz_type == "星座覆盖":
            self.plot_coverage()
        elif viz_type == "误差分析":
            self.plot_error_analysis()
        elif viz_type == "碰撞概率热图":
            self.plot_collision_risk()
    
    def plot_3d_orbit(self):
        """绘制3D轨道图"""
        if 'states' in self.orbit_data:
            # 清除图形
            self.fig.clear()
            
            # 创建3D轴
            ax = self.fig.add_subplot(111, projection='3d')
            
            # 获取状态数据
            states = self.orbit_data['states']
            
            # 绘制地球
            from visualization_enhancements import create_earth_plot
            create_earth_plot(ax, radius=6378.137, alpha=0.3)
            
            # 绘制轨道
            from visualization_enhancements import plot_orbit_with_evolution
            plot_orbit_with_evolution(ax, states, colormap='viridis', show_direction=True)
            
            # 设置轴标签和标题
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.set_title('卫星轨道三维可视化')
            
            # 设置轴范围
            max_range = np.max(np.abs(states[:, :3])) / 1000 * 1.1  # 转换为km并增加10%的余量
            ax.set_xlim3d([-max_range, max_range])
            ax.set_ylim3d([-max_range, max_range])
            ax.set_zlim3d([-max_range, max_range])
            
            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(states)-1))
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
            cbar.set_label('时间演化')
            
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "没有可用的轨道数据进行3D可视化")
    
    def plot_orbit_evolution(self):
        """绘制轨道根数演化图"""
        if 'kepler_history' in self.orbit_data and 'times' in self.orbit_data:
            # 清除图形
            self.fig.clear()
            
            # 获取数据
            kepler_history = self.orbit_data['kepler_history']
            times = self.orbit_data['times'] / 86400  # 转换为天
            
            # 元素名称
            element_names = ['半长轴 (km)', '偏心率', '倾角 (deg)', 
                           '升交点赤经 (deg)', '近地点幅角 (deg)', '平近点角 (deg)']
            
            # 转换单位
            kepler_converted = kepler_history.copy()
            kepler_converted[:, 0] = kepler_converted[:, 0] / 1000  # 转换为km
            kepler_converted[:, 2:] = np.degrees(kepler_converted[:, 2:])  # 转换为度
            
            # 创建子图
            for i, name in enumerate(element_names):
                ax = self.fig.add_subplot(3, 2, i+1)
                ax.plot(times, kepler_converted[:, i], 'b-', linewidth=1.5)
                ax.set_xlabel('时间 (天)')
                ax.set_ylabel(name)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 添加趋势线
                if len(times) > 1:
                    z = np.polyfit(times, kepler_converted[:, i], 1)
                    p = np.poly1d(z)
                    ax.plot(times, p(times), "r--", alpha=0.8, 
                          label=f"趋势: {z[0]:.2e}/天")
                    ax.legend()
            
            self.fig.suptitle('轨道根数演化')
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "没有可用的轨道根数数据进行演化分析")
    
    def plot_perturbation_analysis(self):
        """绘制摄动分析图"""
        if all(k in self.orbit_data for k in ['ref_states', 'j2_states', 'drag_states', 'srp_states', 'times']):
            # 清除图形
            self.fig.clear()
            
            # 获取数据
            ref_states = self.orbit_data['ref_states']
            j2_states = self.orbit_data['j2_states']
            drag_states = self.orbit_data['drag_states']
            srp_states = self.orbit_data['srp_states']
            times = self.orbit_data['times'] / 86400  # 转换为天
            
            # 创建子图
            gs = self.fig.add_gridspec(2, 2)
            
            # 1. 轨道偏差图
            ax1 = self.fig.add_subplot(gs[0, :])
            
            # 计算偏差
            ref_pos = ref_states[:, :3] / 1000  # 转换为km
            j2_pos = j2_states[:, :3] / 1000
            drag_pos = drag_states[:, :3] / 1000
            srp_pos = srp_states[:, :3] / 1000
            
            j2_dev = np.linalg.norm(j2_pos - ref_pos, axis=1)
            drag_dev = np.linalg.norm(drag_pos - ref_pos, axis=1)
            srp_dev = np.linalg.norm(srp_pos - ref_pos, axis=1)
            
            ax1.plot(times, j2_dev, 'r-', label='J2摄动')
            ax1.plot(times, drag_dev, 'g-', label='大气阻力')
            ax1.plot(times, srp_dev, 'b-', label='太阳辐射压')
            
            ax1.set_xlabel('时间 (天)')
            ax1.set_ylabel('轨道偏差 (km)')
            ax1.set_title('不同摄动导致的轨道偏差')
            ax1.legend()
            ax1.grid(True)
            
            # 2. 轨道高度变化图
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # 计算轨道高度
            earth_radius = 6378.137  # km
            ref_height = np.linalg.norm(ref_pos, axis=1) - earth_radius
            j2_height = np.linalg.norm(j2_pos, axis=1) - earth_radius
            drag_height = np.linalg.norm(drag_pos, axis=1) - earth_radius
            srp_height = np.linalg.norm(srp_pos, axis=1) - earth_radius
            
            ax2.plot(times, ref_height, 'k-', label='二体模型')
            ax2.plot(times, j2_height, 'r-', label='J2摄动')
            ax2.plot(times, drag_height, 'g-', label='大气阻力')
            ax2.plot(times, srp_height, 'b-', label='太阳辐射压')
            
            ax2.set_xlabel('时间 (天)')
            ax2.set_ylabel('轨道高度 (km)')
            ax2.set_title('轨道高度变化')
            ax2.legend()
            ax2.grid(True)
            
            # 3. 偏心率变化图
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            # 计算偏心率变化
            from main import OrbitPropagator
            propagator = OrbitPropagator()
            
            ref_kepler = np.array([propagator.cartesian_to_kepler(state) for state in ref_states])
            j2_kepler = np.array([propagator.cartesian_to_kepler(state) for state in j2_states])
            drag_kepler = np.array([propagator.cartesian_to_kepler(state) for state in drag_states])
            srp_kepler = np.array([propagator.cartesian_to_kepler(state) for state in srp_states])
            
            ax3.plot(times, ref_kepler[:, 1], 'k-', label='二体模型')
            ax3.plot(times, j2_kepler[:, 1], 'r-', label='J2摄动')
            ax3.plot(times, drag_kepler[:, 1], 'g-', label='大气阻力')
            ax3.plot(times, srp_kepler[:, 1], 'b-', label='太阳辐射压')
            
            ax3.set_xlabel('时间 (天)')
            ax3.set_ylabel('偏心率')
            ax3.set_title('偏心率变化')
            ax3.legend()
            ax3.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "缺少摄动分析所需的轨道数据")
    
    def plot_coverage(self):
        """绘制星座覆盖图"""
        if 'coverage_data' in self.orbit_data and 'lat_grid' in self.orbit_data and 'lon_grid' in self.orbit_data:
            # 清除图形
            self.fig.clear()
            
            # 获取数据
            coverage_data = self.orbit_data['coverage_data']
            lat_grid = self.orbit_data['lat_grid']
            lon_grid = self.orbit_data['lon_grid']
            
            # 创建子图
            gs = self.fig.add_gridspec(2, 2)
            
            # 1. 全球覆盖热图
            ax1 = self.fig.add_subplot(gs[0, :])
            
            # 创建热图
            im = ax1.pcolormesh(lon_grid, lat_grid, coverage_data, 
                             cmap='viridis', shading='auto', vmin=0, vmax=1)
            
            # 添加颜色条
            cbar = self.fig.colorbar(im, ax=ax1)
            cbar.set_label('覆盖率')
            
            # 设置轴标签和标题
            ax1.set_xlabel('经度 (度)')
            ax1.set_ylabel('纬度 (度)')
            ax1.set_title('全球覆盖率分布')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # 2. 平均覆盖率与纬度关系
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # 计算每个纬度的平均覆盖率
            lat_coverage = np.mean(coverage_data, axis=1)
            
            ax2.plot(lat_grid, lat_coverage, 'b-', linewidth=2)
            ax2.set_xlabel('纬度 (度)')
            ax2.set_ylabel('平均覆盖率')
            ax2.set_title('覆盖率与纬度的关系')
            ax2.grid(True)
            
            # 3. 覆盖率统计
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            # 计算不同覆盖率级别的面积占比
            coverage_levels = np.linspace(0, 1, 11)  # 0到1分成10个区间
            coverage_flat = coverage_data.flatten()
            
            hist, _ = np.histogram(coverage_flat, bins=coverage_levels)
            hist = hist / len(coverage_flat) * 100  # 转换为百分比
            
            # 绘制饼图
            labels = [f'{coverage_levels[i]:.1f}-{coverage_levels[i+1]:.1f}' for i in range(len(coverage_levels)-1)]
            
            # 过滤掉占比为0的区间
            non_zero_indices = hist > 0
            filtered_hist = hist[non_zero_indices]
            filtered_labels = [labels[i] for i, flag in enumerate(non_zero_indices) if flag]
            
            if len(filtered_hist) > 0:
                ax3.pie(filtered_hist, labels=filtered_labels, autopct='%1.1f%%', 
                      startangle=90, wedgeprops={'alpha': 0.7})
                ax3.set_title('覆盖率分布')
            else:
                ax3.text(0.5, 0.5, '没有有效的覆盖率数据', 
                       horizontalalignment='center', verticalalignment='center')
            
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "缺少覆盖率分析所需的数据")
    
    def plot_error_analysis(self):
        """绘制误差分析图"""
        if 'error_results' in self.orbit_data:
            # 清除图形
            self.fig.clear()
            
            # 获取数据
            results = self.orbit_data['error_results']
            
            # 提取数据
            methods = []
            times = []
            errors = []
            
            for name, data in results.items():
                if name != "reference":
                    methods.append(name)
                    times.append(data['time'])
                    errors.append(data['error'])
            
            # 创建子图
            gs = self.fig.add_gridspec(2, 2)
            
            # 1. 计算时间对比
            ax1 = self.fig.add_subplot(gs[0, 0])
            
            ax1.bar(methods, times, color='skyblue')
            ax1.set_ylabel('计算时间 (秒)')
            ax1.set_title('不同方法计算时间对比')
            ax1.set_yscale('log')
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. 误差对比
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            ax2.bar(methods, errors, color='salmon')
            ax2.set_ylabel('相对误差')
            ax2.set_title('不同方法精度对比')
            ax2.set_yscale('log')
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 3. 能量误差随时间变化
            ax3 = self.fig.add_subplot(gs[1, :])
            
            # 导入ErrorAnalysis
            from numerical_methods import ErrorAnalysis
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(methods)))
            for i, (name, data) in enumerate(results.items()):
                if name != "reference":
                    t = data['time_points'] / 86400  # 转换为天
                    states = data['solution']
                    
                    # 计算能量误差
                    mu = 3.986004418e14  # 地球引力常数
                    energy_error = ErrorAnalysis.energy_conservation_error(states, mu)
                    
                    ax3.semilogy(t, abs(energy_error), '-', color=colors[i], linewidth=1.5, label=name)
            
            ax3.set_xlabel('时间 (天)')
            ax3.set_ylabel('能量相对误差')
            ax3.set_title('不同方法的能量守恒误差')
            ax3.legend()
            ax3.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "缺少误差分析所需的数据")
    
    def plot_collision_risk(self):
        """绘制碰撞风险热图"""
        if 'collision_matrix' in self.orbit_data:
            # 清除图形
            self.fig.clear()
            
            # 获取数据
            risk_matrix = self.orbit_data['collision_matrix']
            sat_ids = self.orbit_data.get('sat_ids', np.arange(risk_matrix.shape[0]))
            
            # 创建子图
            gs = self.fig.add_gridspec(2, 2)
            
            # 1. 碰撞风险热图
            ax1 = self.fig.add_subplot(gs[0, :])
            
            # 创建热图
            im = ax1.pcolormesh(sat_ids, sat_ids, risk_matrix,
                             cmap='YlOrRd', shading='auto', norm=matplotlib.colors.LogNorm())
            
            # 添加颜色条
            cbar = self.fig.colorbar(im, ax=ax1)
            cbar.set_label('碰撞概率')
            
            # 设置轴标签和标题
            ax1.set_xlabel('卫星ID')
            ax1.set_ylabel('卫星ID')
            ax1.set_title('星座内卫星对碰撞风险热图')
            
            # 2. 每个卫星的总碰撞风险
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # 计算每个卫星的总碰撞风险
            total_risk_per_sat = np.sum(risk_matrix, axis=1)
            
            # 绘制条形图
            ax2.bar(sat_ids, total_risk_per_sat, color='orangered')
            ax2.set_xlabel('卫星ID')
            ax2.set_ylabel('总碰撞风险')
            ax2.set_title('每个卫星的总碰撞风险')
            ax2.grid(True, axis='y')
            
            # 3. 碰撞风险统计
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            # 提取上三角部分（不含对角线）的风险值
            upper_triangle = risk_matrix[np.triu_indices_from(risk_matrix, k=1)]
            
            # 绘制风险直方图
            if len(upper_triangle) > 0:
                ax3.hist(upper_triangle, bins=20, color='darkred', alpha=0.7, edgecolor='black')
                ax3.set_xlabel('碰撞概率')
                ax3.set_ylabel('频数')
                ax3.set_title('碰撞风险分布')
                ax3.grid(True, axis='y')
                ax3.set_xscale('log')
            else:
                ax3.text(0.5, 0.5, '没有有效的碰撞风险数据', 
                       horizontalalignment='center', verticalalignment='center')
            
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            tk.messagebox.showwarning("警告", "缺少碰撞风险分析所需的数据")
    
    def save_figure(self):
        """保存当前图像"""
        from tkinter import filedialog
        
        # 获取保存文件路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("PDF文件", "*.pdf"), ("SVG文件", "*.svg")]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                tk.messagebox.showinfo("成功", f"图像已保存到:\n{file_path}")
            except Exception as e:
                tk.messagebox.showerror("错误", f"保存图像时出错:\n{str(e)}")


def plot_3d_constellation(states_list, earth_radius=6378.137, alpha=0.3, fig=None):
    """
    绘制3D星座图
    
    参数:
    - states_list: 卫星状态列表，每个元素是一个卫星的状态历史
    - earth_radius: 地球半径 (km)
    - alpha: 轨道线透明度
    - fig: matplotlib 图形对象
    
    返回:
    - fig: matplotlib 图形对象
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地球
    from visualization_enhancements import create_earth_plot
    create_earth_plot(ax, radius=earth_radius, alpha=alpha)
    
    # 绘制每个卫星的轨道
    colors = plt.cm.rainbow(np.linspace(0, 1, len(states_list)))
    
    for i, states in enumerate(states_list):
        # 提取位置数据并转换为km
        pos = states[:, :3] / 1000
        
        # 绘制轨道
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], '-', color=colors[i], linewidth=1, alpha=0.7,
              label=f'卫星 {i+1}')
        
        # 绘制卫星当前位置
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], s=30, color=colors[i], edgecolor='black')
    
    # 设置轴标签
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('星座三维可视化')
    
    # 计算适当的轴范围
    all_positions = np.vstack([states[:, :3] / 1000 for states in states_list])
    max_range = np.max(np.abs(all_positions)) * 1.1
    
    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([-max_range, max_range])
    
    # 如果星座不太大，添加图例
    if len(states_list) <= 10:
        ax.legend()
    
    return fig


def create_interactive_visualization(parent, title="交互式数据可视化"):
    """创建并返回交互式可视化窗口"""
    viz_window = AdvancedVisualizationWindow(parent, title)
    return viz_window 