"""
UI扩展模块，提供增强的用户界面功能
"""

from datetime import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import threading
import traceback
import time
from matplotlib.figure import Figure

# 配置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12                       # 设置字体大小

class AdvancedAnalysisWindow:
    """先进的分析窗口"""
    
    def __init__(self, parent, title="高级轨道分析"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1000x800")
        
        # 创建笔记本控件
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建各种分析标签页
        self.create_tabs()
    
    def create_tabs(self):
        """创建分析标签页"""
        # 轨道演化分析标签页
        self.orbit_evolution_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.orbit_evolution_frame, text="轨道演化分析")
        
        # 数值方法对比标签页
        self.numerical_methods_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.numerical_methods_frame, text="数值方法对比")
        
        # 摄动分析标签页
        self.perturbation_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.perturbation_analysis_frame, text="摄动力分析")
        
        # 星座覆盖分析标签页
        self.coverage_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.coverage_frame, text="覆盖率分析")
        
        # 误差分析标签页
        self.error_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.error_analysis_frame, text="误差分析")
        
        # 碰撞风险分析标签页
        self.collision_risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.collision_risk_frame, text="碰撞风险分析")
    
    def create_orbit_evolution_plot(self, kepler_elements_history, time_days):
        """在轨道演化标签页创建图表"""
        frame = self.orbit_evolution_frame
        
        # 清空现有内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        
        # 元素名称
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
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
    
    def create_numerical_methods_comparison(self, results):
        """在数值方法标签页创建对比图表"""
        frame = self.numerical_methods_frame
        
        # 清空现有内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 控制区域
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        
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
        gs = fig.add_gridspec(2, 2)
        
        # 绘制计算时间对比
        ax1 = fig.add_subplot(gs[0, 0])
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
        
        # 创建能量误差和角动量误差比较
        ax3 = fig.add_subplot(gs[1, :])
        
        for name, data in results.items():
            if name != "reference":
                # 提取时间和状态
                t = data['time_points']
                y = data['solution']
                
                # 计算相对能量误差
                from numerical_methods import ErrorAnalysis
                energy_error = ErrorAnalysis.energy_conservation_error(y, 3.986004418e14)
                
                # 绘制能量误差
                ax3.plot(t / 86400, abs(energy_error), label=f"{name}", alpha=0.7)
        
        ax3.set_xlabel('时间 (天)')
        ax3.set_ylabel('能量相对误差')
        ax3.set_title('不同方法的能量守恒误差比较')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        fig.tight_layout()
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
    
    def create_perturbation_analysis(self, ref_states, j2_states, drag_states, srp_states, time_days):
        """创建摄动力分析图表"""
        frame = self.perturbation_analysis_frame
        
        # 清空现有内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        
        # 提取位置数据
        ref_pos = ref_states[:, :3] / 1000  # 转换为km
        j2_pos = j2_states[:, :3] / 1000
        drag_pos = drag_states[:, :3] / 1000
        srp_pos = srp_states[:, :3] / 1000
        
        # 计算相对于参考轨道的偏差
        j2_deviation = np.linalg.norm(j2_pos - ref_pos, axis=1)
        drag_deviation = np.linalg.norm(drag_pos - ref_pos, axis=1)
        srp_deviation = np.linalg.norm(srp_pos - ref_pos, axis=1)
        
        # 绘制偏差随时间的变化
        ax1 = fig.add_subplot(211)
        ax1.plot(time_days, j2_deviation, 'r-', label='J2摄动')
        ax1.plot(time_days, drag_deviation, 'g-', label='大气阻力')
        ax1.plot(time_days, srp_deviation, 'b-', label='太阳辐射压')
        ax1.set_xlabel('时间 (天)')
        ax1.set_ylabel('轨道偏差 (km)')
        ax1.set_title('不同摄动力导致的轨道偏差')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制轨道高度变化
        ax2 = fig.add_subplot(212)
        
        # 计算轨道高度
        earth_radius = 6378.137  # km
        ref_height = np.linalg.norm(ref_pos, axis=1) - earth_radius
        j2_height = np.linalg.norm(j2_pos, axis=1) - earth_radius
        drag_height = np.linalg.norm(drag_pos, axis=1) - earth_radius
        srp_height = np.linalg.norm(srp_pos, axis=1) - earth_radius
        
        ax2.plot(time_days, ref_height, 'k-', label='二体模型')
        ax2.plot(time_days, j2_height, 'r-', label='J2摄动')
        ax2.plot(time_days, drag_height, 'g-', label='大气阻力')
        ax2.plot(time_days, srp_height, 'b-', label='太阳辐射压')
        ax2.set_xlabel('时间 (天)')
        ax2.set_ylabel('轨道高度 (km)')
        ax2.set_title('不同摄动力下的轨道高度变化')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
    
    def create_collision_risk_table(self, risk_data):
        """创建碰撞风险数据表格"""
        frame = self.collision_risk_frame
        
        # 清空现有内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 创建表格区域
        table_frame = ttk.LabelFrame(frame, text="碰撞风险对")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建Treeview控件
        columns = ('sat_i', 'sat_j', 'probability', 'distance')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        tree.heading('sat_i', text='卫星i')
        tree.heading('sat_j', text='卫星j')
        tree.heading('probability', text='碰撞概率')
        tree.heading('distance', text='最小距离(km)')
        
        tree.column('sat_i', width=100, anchor='center')
        tree.column('sat_j', width=100, anchor='center')
        tree.column('probability', width=150, anchor='center')
        tree.column('distance', width=150, anchor='center')
        
        # 添加数据
        for i in range(len(risk_data)):
            tree.insert('', tk.END, values=(
                risk_data.iloc[i]['卫星i'],
                risk_data.iloc[i]['卫星j'],
                f"{risk_data.iloc[i]['碰撞概率']:.6f}",
                f"{risk_data.iloc[i]['最小距离(km)']:.2f}"
            ))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
    
    def create_collision_risk_heatmap(self, risk_matrix):
        """创建碰撞风险热图"""
        frame = self.collision_risk_frame
        
        # 创建热图区域
        heatmap_frame = ttk.LabelFrame(frame, text="碰撞风险热图")
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建图形
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # 绘制热图
        sns.heatmap(risk_matrix, cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': '碰撞概率'}, vmin=0, vmax=max(0.001, np.max(risk_matrix)))
        
        ax.set_title('星座内卫星对碰撞风险热图')
        ax.set_xlabel('卫星序号')
        ax.set_ylabel('卫星序号')
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
    
    def display_avoidance_recommendation(self, sat_i, sat_j, risk, dist_before, dist_after, delta_v):
        """显示碰撞规避建议"""
        frame = self.collision_risk_frame
        
        # 创建建议区域
        recommendation_frame = ttk.LabelFrame(frame, text="规避建议")
        recommendation_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建信息标签
        info_text = f"""
        高风险碰撞对: 卫星{sat_i} - 卫星{sat_j}
        碰撞概率: {risk:.6f}
        当前最小距离: {dist_before:.2f} m
        规避后最小距离: {dist_after:.2f} m
        建议速度增量: [{delta_v[0]:.4f}, {delta_v[1]:.4f}, {delta_v[2]:.4f}] m/s
        增量大小: {np.linalg.norm(delta_v):.4f} m/s
        """
        
        info_label = ttk.Label(recommendation_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)
        
        # 创建规避按钮
        apply_btn = ttk.Button(recommendation_frame, text="应用规避建议", 
                             command=lambda: messagebox.showinfo("规避建议", "此功能需要与仿真系统集成"))
        apply_btn.pack(padx=10, pady=10)
    
    def display_message(self, message):
        """显示消息"""
        frame = self.collision_risk_frame
        
        # 清空现有内容
        for widget in frame.winfo_children():
            widget.destroy()
        
        # 创建消息标签
        msg_label = ttk.Label(frame, text=message, font=('Arial', 14))
        msg_label.pack(expand=True, padx=20, pady=20)

class MethodComparisonWindow:
    """数值方法比较窗口"""
    
    def __init__(self, parent, title="数值方法比较", propagator=None):
        """
        初始化数值方法比较窗口
        
        参数:
        - parent: 父窗口
        - title: 窗口标题
        - propagator: 轨道传播器实例
        """
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1200x800")
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        
        self.propagator = propagator
        self.parent = parent
        
        # 创建界面
        self.create_widgets()
        
        # 默认初始状态
        self.initial_state = np.array([
            7000000.0, 0.0, 0.0,  # 位置 (m)
            0.0, 7546.0, 0.0      # 速度 (m/s)
        ])
        
        # 可用的数值方法
        self.available_methods = [
            "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA",
            "Euler", "Midpoint", "RK4", "Verlet"
        ]
        
        # 方法对比结果
        self.results = {}
        
        # 进度条变量
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
    def create_widgets(self):
        """创建窗口组件"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="参数设置", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 1. 初始状态设置
        init_frame = ttk.LabelFrame(control_frame, text="初始状态", padding=5)
        init_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 位置输入
        pos_frame = ttk.Frame(init_frame)
        pos_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pos_frame, text="位置 (km):", width=10).pack(side=tk.LEFT)
        
        self.pos_x = ttk.Entry(pos_frame, width=8)
        self.pos_x.pack(side=tk.LEFT, padx=2)
        self.pos_x.insert(0, "7000")
        
        self.pos_y = ttk.Entry(pos_frame, width=8)
        self.pos_y.pack(side=tk.LEFT, padx=2)
        self.pos_y.insert(0, "0")
        
        self.pos_z = ttk.Entry(pos_frame, width=8)
        self.pos_z.pack(side=tk.LEFT, padx=2)
        self.pos_z.insert(0, "0")
        
        # 速度输入
        vel_frame = ttk.Frame(init_frame)
        vel_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vel_frame, text="速度 (m/s):", width=10).pack(side=tk.LEFT)
        
        self.vel_x = ttk.Entry(vel_frame, width=8)
        self.vel_x.pack(side=tk.LEFT, padx=2)
        self.vel_x.insert(0, "0")
        
        self.vel_y = ttk.Entry(vel_frame, width=8)
        self.vel_y.pack(side=tk.LEFT, padx=2)
        self.vel_y.insert(0, "7546")
        
        self.vel_z = ttk.Entry(vel_frame, width=8)
        self.vel_z.pack(side=tk.LEFT, padx=2)
        self.vel_z.insert(0, "0")
        
        # 2. 仿真设置
        sim_frame = ttk.LabelFrame(control_frame, text="仿真设置", padding=5)
        sim_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 仿真时长
        duration_frame = ttk.Frame(sim_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="持续时间 (天):").pack(side=tk.LEFT)
        self.duration = ttk.Entry(duration_frame, width=10)
        self.duration.pack(side=tk.LEFT, padx=5)
        self.duration.insert(0, "1")
        
        # 步长设置
        step_frame = ttk.Frame(sim_frame)
        step_frame.pack(fill=tk.X, pady=5)
        ttk.Label(step_frame, text="步长范围 (秒):").pack(side=tk.LEFT)
        
        step_size_frame = ttk.Frame(step_frame)
        step_size_frame.pack(side=tk.LEFT, padx=5)
        
        self.min_step = ttk.Entry(step_size_frame, width=8)
        self.min_step.pack(side=tk.LEFT)
        self.min_step.insert(0, "10")
        
        ttk.Label(step_size_frame, text="至").pack(side=tk.LEFT, padx=2)
        
        self.max_step = ttk.Entry(step_size_frame, width=8)
        self.max_step.pack(side=tk.LEFT)
        self.max_step.insert(0, "300")
        
        # 步长数量
        step_count_frame = ttk.Frame(sim_frame)
        step_count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(step_count_frame, text="步长数量:").pack(side=tk.LEFT)
        self.step_count = ttk.Entry(step_count_frame, width=10)
        self.step_count.pack(side=tk.LEFT, padx=5)
        self.step_count.insert(0, "5")
        
        # 3. 方法选择
        method_frame = ttk.LabelFrame(control_frame, text="方法选择", padding=5)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建方法选择列表框
        self.method_list = tk.Listbox(method_frame, selectmode=tk.MULTIPLE, height=10)
        self.method_list.pack(fill=tk.X, pady=5)
        
        # 添加可用方法
        for method in self.available_methods:
            self.method_list.insert(tk.END, method)
        
        # 默认选择前3个方法
        for i in range(min(3, len(self.available_methods))):
            self.method_list.selection_set(i)
        
        # 摄动设置
        pert_frame = ttk.LabelFrame(control_frame, text="摄动选项", padding=5)
        pert_frame.pack(fill=tk.X, pady=(0, 10))
        
        # J2摄动
        self.j2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="启用J2摄动", variable=self.j2_var).pack(anchor=tk.W, pady=2)
        
        # 大气阻力摄动
        self.drag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pert_frame, text="启用大气阻力", variable=self.drag_var).pack(anchor=tk.W, pady=2)
        
        # 太阳辐射压摄动
        self.srp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pert_frame, text="启用太阳辐射压", variable=self.srp_var).pack(anchor=tk.W, pady=2)
        
        # A/m比值
        am_frame = ttk.Frame(pert_frame)
        am_frame.pack(fill=tk.X, pady=5)
        ttk.Label(am_frame, text="A/m比值:").pack(side=tk.LEFT)
        self.am_ratio = ttk.Entry(am_frame, width=10)
        self.am_ratio.pack(side=tk.LEFT, padx=5)
        self.am_ratio.insert(0, "0.01")
        
        # 执行按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="运行比较", command=self.run_comparison)
        self.run_btn.pack(fill=tk.X, pady=5)
        
        self.clear_btn = ttk.Button(btn_frame, text="清除结果", command=self.clear_results)
        self.clear_btn.pack(fill=tk.X, pady=5)
        
        # 进度条和状态
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)
        
        # 右侧结果显示
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建绘图区域
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(result_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def clear_results(self):
        """清除结果"""
        self.results = {}
        self.fig.clear()
        self.canvas.draw()
        self.status_var.set("结果已清除")
        
    def run_comparison(self):
        """运行方法比较"""
        # 获取选择的方法
        selected_indices = self.method_list.curselection()
        if not selected_indices:
            messagebox.showwarning("警告", "请至少选择一种数值方法")
            return
        
        selected_methods = [self.available_methods[i] for i in selected_indices]
        
        # 获取参数
        try:
            # 位置 (转换为米)
            x = float(self.pos_x.get()) * 1000
            y = float(self.pos_y.get()) * 1000
            z = float(self.pos_z.get()) * 1000
            
            # 速度
            vx = float(self.vel_x.get())
            vy = float(self.vel_y.get())
            vz = float(self.vel_z.get())
            
            # 初始状态向量
            initial_state = np.array([x, y, z, vx, vy, vz])
            
            # 时间参数
            duration_days = float(self.duration.get())
            t_span = (0, duration_days * 86400)  # 转换为秒
            
            # 步长设置
            min_step = float(self.min_step.get())
            max_step = float(self.max_step.get())
            step_count = int(self.step_count.get())
            
            # 步长列表
            if step_count > 1:
                step_sizes = np.logspace(np.log10(min_step), np.log10(max_step), step_count)
            else:
                step_sizes = [min_step]
            
            # 摄动选项
            enable_j2 = self.j2_var.get()
            enable_drag = self.drag_var.get()
            enable_srp = self.srp_var.get()
            A_m_ratio = float(self.am_ratio.get())
            
        except ValueError as e:
            messagebox.showerror("错误", f"参数输入错误: {str(e)}")
            return
        
        # 禁用运行按钮
        self.run_btn.config(state="disabled")
        self.status_var.set("计算中...")
        self.progress_var.set(0)
        
        # 在新线程中运行比较
        threading.Thread(target=self._run_comparison_thread, args=(
            selected_methods, initial_state, t_span, step_sizes, 
            enable_j2, enable_drag, enable_srp, A_m_ratio)).start()
    
    def _run_comparison_thread(self, methods, initial_state, t_span, step_sizes, 
                              enable_j2, enable_drag, enable_srp, A_m_ratio):
        """在新线程中运行比较"""
        try:
            # 设置参考解
            self.results = {}
            
            # 定义加速度函数
            def acceleration_func(t, state):
                if self.propagator:
                    # 使用传播器的轨道导数函数
                    derivatives = self.propagator.orbit_derivatives(
                        t, state, A_m_ratio=A_m_ratio, 
                        enable_j2=enable_j2, enable_drag=enable_drag, enable_srp=enable_srp
                    )
                    return derivatives[3:6]  # 只返回加速度部分
                else:
                    # 简化的二体模型
                    mu = 3.986004418e14  # 地球引力常数 [m^3/s^2]
                    r = state[:3]
                    r_norm = np.linalg.norm(r)
                    if r_norm == 0:
                        return np.zeros(3)
                    a_gravity = -mu * r / r_norm**3
                    
                    # 添加J2摄动
                    if enable_j2:
                        J2 = 1.08263e-3
                        Re = 6378.137e3  # 地球半径 [m]
                        x, y, z = r
                        r2 = r_norm**2
                        
                        factor = 1.5 * J2 * mu * Re**2 / r_norm**5
                        a_J2_x = factor * x * (5 * z**2 / r2 - 1)
                        a_J2_y = factor * y * (5 * z**2 / r2 - 1)
                        a_J2_z = factor * z * (5 * z**2 / r2 - 3)
                        
                        a_gravity += np.array([a_J2_x, a_J2_y, a_J2_z])
                    
                    return a_gravity
            
            # 计算参考解 (使用高精度方法)
            self.status_var.set("计算参考解...")
            ref_method = "DOP853"
            ref_func = self._get_method_function(ref_method)
            
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
            
            try:
                ref_time, ref_states = self._run_method(
                    ref_func, acceleration_func, initial_state, t_span, 1.0, t_eval=t_eval
                )
                
                self.results["reference"] = {
                    "solution": ref_states,
                    "time_points": t_eval,
                    "time": 0,
                    "error": 0
                }
            except Exception as e:
                messagebox.showerror("错误", f"计算参考解失败: {str(e)}")
                self.run_btn.config(state="normal")
                self.status_var.set("计算失败")
                return
            
            # 对每个方法运行比较
            total_runs = len(methods) * len(step_sizes)
            run_count = 0
            
            for method in methods:
                method_results = []
                method_times = []
                method_errors = []
                
                for step_size in step_sizes:
                    run_count += 1
                    progress = run_count / total_runs * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"计算 {method} (步长={step_size:.1f}s)...")
                    self.window.update_idletasks()
                    
                    try:
                        method_func = self._get_method_function(method)
                        start_time = time.time()
                        _, states = self._run_method(
                            method_func, acceleration_func, initial_state, t_span, step_size
                        )
                        run_time = time.time() - start_time
                        
                        # 计算误差
                        error = self._calculate_error(ref_states, states)
                        
                        method_results.append(states)
                        method_times.append(run_time)
                        method_errors.append(error)
                        
                    except Exception as e:
                        print(f"方法 {method} 在步长 {step_size} 下失败: {str(e)}")
                        method_results.append(None)
                        method_times.append(float('nan'))
                        method_errors.append(float('nan'))
                
                # 存储该方法的结果
                self.results[method] = {
                    "step_sizes": step_sizes,
                    "times": method_times,
                    "errors": method_errors,
                    "solutions": method_results
                }
            
            # 绘制结果
            self.status_var.set("绘制结果...")
            self.window.update_idletasks()
            self._plot_results(self.results, step_sizes)
            
            self.status_var.set("完成")
        
        except Exception as e:
            messagebox.showerror("错误", f"比较过程中发生错误: {str(e)}")
            self.status_var.set("计算失败")
        
        finally:
            # 重新启用运行按钮
            self.run_btn.config(state="normal")
    
    def _get_method_function(self, method_name):
        """获取指定方法的求解函数"""
        if method_name in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]:
            # SciPy内置方法
            from scipy.integrate import solve_ivp
            
            def method_func(func, y0, t_span, dt):
                return solve_ivp(func, t_span, y0, method=method_name, max_step=dt)
            
            return method_func
        
        elif method_name == "Euler":
            # 欧拉方法
            def euler_method(func, y0, t_span, dt):
                t_start, t_end = t_span
                steps = max(int((t_end - t_start) / dt), 1)
                dt = (t_end - t_start) / steps
                
                y = np.copy(y0)
                t = t_start
                y_history = [y0]
                t_history = [t_start]
                
                for _ in range(steps):
                    dydt = np.concatenate([y[3:6], func(t, y)])
                    y = y + dt * dydt
                    t = t + dt
                    y_history.append(np.copy(y))
                    t_history.append(t)
                
                return {"t": np.array(t_history), "y": np.array(y_history).T}
            
            return euler_method
        
        elif method_name == "Midpoint":
            # 中点法
            def midpoint_method(func, y0, t_span, dt):
                t_start, t_end = t_span
                steps = max(int((t_end - t_start) / dt), 1)
                dt = (t_end - t_start) / steps
                
                y = np.copy(y0)
                t = t_start
                y_history = [y0]
                t_history = [t_start]
                
                for _ in range(steps):
                    # 计算中点
                    dydt = np.concatenate([y[3:6], func(t, y)])
                    y_mid = y + 0.5 * dt * dydt
                    t_mid = t + 0.5 * dt
                    
                    # 使用中点斜率
                    dydt_mid = np.concatenate([y_mid[3:6], func(t_mid, y_mid)])
                    y = y + dt * dydt_mid
                    t = t + dt
                    
                    y_history.append(np.copy(y))
                    t_history.append(t)
                
                return {"t": np.array(t_history), "y": np.array(y_history).T}
            
            return midpoint_method
        
        elif method_name == "RK4":
            # 经典四阶龙格-库塔方法
            def rk4_method(func, y0, t_span, dt):
                t_start, t_end = t_span
                steps = max(int((t_end - t_start) / dt), 1)
                dt = (t_end - t_start) / steps
                
                y = np.copy(y0)
                t = t_start
                y_history = [y0]
                t_history = [t_start]
                
                for _ in range(steps):
                    # RK4系数
                    k1 = np.concatenate([y[3:6], func(t, y)])
                    k2 = np.concatenate([y[3:6] + 0.5 * dt * k1[3:6], func(t + 0.5 * dt, y + 0.5 * dt * k1)])
                    k3 = np.concatenate([y[3:6] + 0.5 * dt * k2[3:6], func(t + 0.5 * dt, y + 0.5 * dt * k2)])
                    k4 = np.concatenate([y[3:6] + dt * k3[3:6], func(t + dt, y + dt * k3)])
                    
                    # 更新
                    y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                    t = t + dt
                    
                    y_history.append(np.copy(y))
                    t_history.append(t)
                
                return {"t": np.array(t_history), "y": np.array(y_history).T}
            
            return rk4_method
        
        elif method_name == "Verlet":
            # Velocity Verlet方法 (适用于保守系统)
            def verlet_method(func, y0, t_span, dt):
                t_start, t_end = t_span
                steps = max(int((t_end - t_start) / dt), 1)
                dt = (t_end - t_start) / steps
                
                r = np.copy(y0[:3])
                v = np.copy(y0[3:6])
                t = t_start
                
                y_history = [np.concatenate([r, v])]
                t_history = [t_start]
                
                a = func(t, y0)  # 初始加速度
                
                for _ in range(steps):
                    # 更新位置
                    r = r + v * dt + 0.5 * a * dt**2
                    
                    # 保存旧加速度
                    a_old = a
                    
                    # 计算新加速度
                    y_new = np.concatenate([r, v])
                    a = func(t + dt, y_new)
                    
                    # 更新速度
                    v = v + 0.5 * (a_old + a) * dt
                    
                    # 更新时间
                    t = t + dt
                    
                    y_history.append(np.concatenate([r, v]))
                    t_history.append(t)
                
                return {"t": np.array(t_history), "y": np.array(y_history).T}
            
            return verlet_method
        
        else:
            raise ValueError(f"未知的数值方法: {method_name}")
    
    def _run_method(self, method_func, accel_func, initial_state, t_span, step_size, t_eval=None):
        """运行指定的数值方法"""
        # 封装函数以适应一阶ODE求解器
        def func(t, y):
            return np.concatenate([y[3:6], accel_func(t, y)])
        
        # 运行方法
        result = method_func(func, initial_state, t_span, step_size)
        
        # 从结果中提取时间和状态
        if isinstance(result, dict):
            t = result["t"]
            states = result["y"].T
        else:
            # scipy.integrate.solve_ivp 返回对象
            t = result.t
            states = result.y.T
            
            # 如果指定了t_eval，进行插值
            if t_eval is not None and len(t) > 1:
                from scipy.interpolate import interp1d
                interp_func = interp1d(t, states, axis=0, bounds_error=False, fill_value="extrapolate")
                states = interp_func(t_eval)
                t = t_eval
        
        return t, states
    
    def _calculate_error(self, ref_states, states):
        """计算相对于参考解的误差"""
        # 提取位置部分
        ref_pos = ref_states[:, :3]
        pos = states[:, :3]
        
        # 如果长度不同，进行插值
        if len(pos) != len(ref_pos) and len(pos) > 1 and len(ref_pos) > 1:
            from scipy.interpolate import interp1d
            t_test = np.linspace(0, 1, len(pos))
            t_ref = np.linspace(0, 1, len(ref_pos))
            interp_func = interp1d(t_ref, ref_pos, axis=0, bounds_error=False, fill_value="extrapolate")
            ref_pos_interp = interp_func(t_test)
            
            # 计算均方根误差
            error = np.sqrt(np.mean(np.sum((pos - ref_pos_interp)**2, axis=1)))
        else:
            # 长度相同，直接计算
            min_len = min(len(pos), len(ref_pos))
            error = np.sqrt(np.mean(np.sum((pos[:min_len] - ref_pos[:min_len])**2, axis=1)))
        
        # 归一化误差
        ref_pos_norm = np.sqrt(np.mean(np.sum(ref_pos**2, axis=1)))
        if ref_pos_norm > 0:
            relative_error = error / ref_pos_norm
        else:
            relative_error = error
        
        return relative_error
    
    def _plot_results(self, results, step_sizes):
        """绘制结果比较图表"""
        self.fig.clear()
        
        # 如果没有足够的结果，返回
        if len(results) <= 1:  # 只有参考解
            return
        
        # 1. 创建子图
        gs = self.fig.add_gridspec(2, 2)
        ax1 = self.fig.add_subplot(gs[0, 0])  # 计算时间
        ax2 = self.fig.add_subplot(gs[0, 1])  # 精度
        ax3 = self.fig.add_subplot(gs[1, :])  # 轨道图
        
        methods = [m for m in results.keys() if m != "reference"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # 2. 绘制计算时间图
        for i, method in enumerate(methods):
            method_data = results[method]
            times = method_data["times"]
            ax1.plot(method_data["step_sizes"], times, 'o-', label=method, color=colors[i])
            
        ax1.set_xlabel('步长 (秒)')
        ax1.set_ylabel('计算时间 (秒)')
        ax1.set_title('不同方法的计算时间')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="--", alpha=0.7)
        ax1.legend()
        
        # 3. 绘制精度图
        for i, method in enumerate(methods):
            method_data = results[method]
            errors = method_data["errors"]
            ax2.plot(method_data["step_sizes"], errors, 'o-', label=method, color=colors[i])
            
        ax2.set_xlabel('步长 (秒)')
        ax2.set_ylabel('相对误差')
        ax2.set_title('不同方法的精度比较')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, which="both", ls="--", alpha=0.7)
        ax2.legend()
        
        # 4. 绘制轨道图
        ref_solution = results["reference"]["solution"]
        
        # 绘制参考轨道
        ax3.plot(ref_solution[:, 0]/1000, ref_solution[:, 1]/1000, 'k-', 
                 linewidth=2, alpha=0.8, label='参考解')
        
        # 绘制每个方法的轨道
        for i, method in enumerate(methods):
            method_data = results[method]
            
            # 使用第一个步长的结果
            if method_data["solutions"][0] is not None:
                orbit = method_data["solutions"][0]
                ax3.plot(orbit[:, 0]/1000, orbit[:, 1]/1000, '--', 
                         linewidth=1.5, alpha=0.7, label=method, color=colors[i])
        
        ax3.set_xlabel('X (km)')
        ax3.set_ylabel('Y (km)')
        ax3.set_title('轨道比较 (XY平面投影)')
        ax3.grid(True)
        ax3.axis('equal')
        ax3.legend()
        
        # 添加地球
        theta = np.linspace(0, 2*np.pi, 100)
        earth_radius = 6378.137  # 地球半径 (km)
        ax3.plot(earth_radius * np.cos(theta), earth_radius * np.sin(theta), 
                 'b-', linewidth=1, alpha=0.3)
        ax3.fill(earth_radius * np.cos(theta), earth_radius * np.sin(theta), 
                 'b', alpha=0.1)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 可选: 添加详细结果表格
        self._add_result_table(results, methods, step_sizes)

    def _add_result_table(self, results, methods, step_sizes):
        """添加结果表格"""
        # 创建新窗口
        table_window = tk.Toplevel(self.window)
        table_window.title("详细结果比较")
        table_window.geometry("800x600")
        
        # 创建表格框架
        frame = ttk.Frame(table_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建表格
        columns = ["方法", "步长 (秒)", "计算时间 (秒)", "相对误差", "评价"]
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # 添加列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)
        
        # 添加数据
        for method in methods:
            method_data = results[method]
            
            for i, step_size in enumerate(step_sizes):
                # 获取计算时间和误差
                time_value = method_data["times"][i]
                error_value = method_data["errors"][i]
                
                # 评价结果
                if np.isnan(error_value):
                    evaluation = "计算失败"
                elif error_value < 1e-10:
                    evaluation = "极高精度"
                elif error_value < 1e-8:
                    evaluation = "高精度"
                elif error_value < 1e-6:
                    evaluation = "良好精度"
                elif error_value < 1e-4:
                    evaluation = "一般精度"
                else:
                    evaluation = "低精度"
                
                # 添加行
                row_values = [
                    method,
                    f"{step_size:.2f}",
                    f"{time_value:.6f}" if not np.isnan(time_value) else "N/A",
                    f"{error_value:.2e}" if not np.isnan(error_value) else "N/A",
                    evaluation
                ]
                
                tree.insert("", tk.END, values=row_values)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加导出按钮
        export_frame = ttk.Frame(table_window, padding=10)
        export_frame.pack(fill=tk.X)
        
        def export_to_csv():
            """导出结果到CSV文件"""
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
                title="保存结果数据"
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # 写入标题
                        writer.writerow(columns)
                        
                        # 写入数据
                        for item in tree.get_children():
                            values = tree.item(item)['values']
                            writer.writerow(values)
                        
                    messagebox.showinfo("导出成功", f"数据已成功导出到 {file_path}")
                except Exception as e:
                    messagebox.showerror("导出错误", f"导出数据时发生错误: {str(e)}")
        
        export_btn = ttk.Button(export_frame, text="导出到CSV", command=export_to_csv)
        export_btn.pack(side=tk.RIGHT)

class ConstellationOptimizationWindow:
    """星座优化窗口"""
    
    def __init__(self, parent, designer):
        """
        初始化星座优化窗口
        
        参数:
        - parent: 父窗口
        - designer: 星座设计器对象
        """
        self.window = tk.Toplevel(parent)
        self.window.title("星座优化")
        self.window.geometry("1200x800")
        self.designer = designer
        
        # 创建界面
        self.create_widgets()
        
        # 导入星座优化模块
        try:
            from constellation_optimization import ConstellationOptimizer
            self.optimizer = ConstellationOptimizer(designer.propagator, designer)
            self.log("已创建优化器实例")
        except ImportError as e:
            messagebox.showerror("导入错误", f"无法导入星座优化模块: {str(e)}")
            self.window.destroy()
            return
        
        # 优化线程和标志
        self.optimization_thread = None
        self.running = False
        
        # 优化结果数据
        self.optimization_result = None
        self.iteration_data = {
            'iterations': [],
            'best_scores': [],
            'mean_scores': [],
            'parameters': []
        }
        
        # 创建可视化图表
        self.viz_fig = plt.figure(figsize=(8, 6))
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.results_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化结果显示
        self._update_result_display(None)
        
        # 窗口关闭处理
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """窗口关闭时处理"""
        if self.running:
            if messagebox.askyesno("确认", "优化正在运行中，确定要关闭窗口吗？"):
                self.stop_optimization()
                self.window.destroy()
        else:
            self.window.destroy()

    def create_widgets(self):
        """创建窗口组件"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 分割为左右两部分
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), expand=True, anchor=tk.N)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 左侧: 优化设置
        self.create_optimization_settings(left_frame)
        
        # 右侧: 结果显示
        self.create_results_display(right_frame)
    
    def create_optimization_settings(self, parent):
        """创建优化设置界面"""
        # 创建设置框架
        settings_frame = ttk.LabelFrame(parent, text="优化设置", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 优化目标
        obj_frame = ttk.LabelFrame(settings_frame, text="优化目标", padding=5)
        obj_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.objective_var = tk.StringVar(value="coverage")
        
        obj_options = [
            ("覆盖率", "coverage"),
            ("重访时间", "revisit"),
            ("碰撞风险", "collision"),
            ("多目标", "multi")
        ]
        
        for text, value in obj_options:
            ttk.Radiobutton(obj_frame, text=text, value=value, 
                           variable=self.objective_var).pack(anchor=tk.W, pady=2)
        
        # 多目标权重
        weights_frame = ttk.Frame(obj_frame)
        weights_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(weights_frame, text="权重:").pack(side=tk.LEFT)
        
        ttk.Label(weights_frame, text="覆盖率").pack(side=tk.LEFT, padx=(10, 2))
        self.cov_weight = ttk.Entry(weights_frame, width=5)
        self.cov_weight.pack(side=tk.LEFT)
        self.cov_weight.insert(0, "0.4")
        
        ttk.Label(weights_frame, text="重访时间").pack(side=tk.LEFT, padx=(10, 2))
        self.rev_weight = ttk.Entry(weights_frame, width=5)
        self.rev_weight.pack(side=tk.LEFT)
        self.rev_weight.insert(0, "0.3")
        
        ttk.Label(weights_frame, text="碰撞风险").pack(side=tk.LEFT, padx=(10, 2))
        self.col_weight = ttk.Entry(weights_frame, width=5)
        self.col_weight.pack(side=tk.LEFT)
        self.col_weight.insert(0, "0.3")
        
        # 2. 优化算法
        alg_frame = ttk.LabelFrame(settings_frame, text="优化算法", padding=5)
        alg_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.algorithm_var = tk.StringVar(value="grid")
        
        alg_options = [
            ("网格搜索", "grid"),
            ("粒子群优化", "pso"),
            ("遗传算法", "genetic")
        ]
        
        for text, value in alg_options:
            ttk.Radiobutton(alg_frame, text=text, value=value, 
                           variable=self.algorithm_var).pack(anchor=tk.W, pady=2)
        
        # 3. 参数设置
        param_frame = ttk.LabelFrame(settings_frame, text="参数范围", padding=5)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建参数输入
        params = [
            ("卫星总数", "sat_range", "颗", (20, 80)),
            ("轨道平面数", "plane_range", "个", (3, 8)),
            ("轨道倾角", "inc_range", "度", (45, 90)),
            ("轨道高度", "alt_range", "km", (500, 1500))
        ]
        
        self.param_entries = {}
        
        for label, key, unit, default in params:
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            
            min_entry = ttk.Entry(frame, width=8)
            min_entry.pack(side=tk.LEFT, padx=5)
            min_entry.insert(0, str(default[0]))
            
            ttk.Label(frame, text="至").pack(side=tk.LEFT)
            
            max_entry = ttk.Entry(frame, width=8)
            max_entry.pack(side=tk.LEFT, padx=5)
            max_entry.insert(0, str(default[1]))
            
            ttk.Label(frame, text=unit).pack(side=tk.LEFT)
            
            self.param_entries[key] = (min_entry, max_entry)
        
        # 4. 高级设置
        adv_frame = ttk.LabelFrame(settings_frame, text="高级设置", padding=5)
        adv_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 迭代次数
        iter_frame = ttk.Frame(adv_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(iter_frame, text="最大迭代次数:").pack(side=tk.LEFT)
        self.max_iter = ttk.Entry(iter_frame, width=8)
        self.max_iter.pack(side=tk.LEFT, padx=5)
        self.max_iter.insert(0, "20")
        
        # 使用多处理
        self.multiprocessing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="使用多处理加速", 
                        variable=self.multiprocessing_var).pack(anchor=tk.W, pady=2)
        
        # 5. 操作按钮
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="运行优化", command=self.run_optimization)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止优化", command=self.stop_optimization, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.apply_btn = ttk.Button(btn_frame, text="应用最优配置", command=self.apply_optimal_config, state="disabled")
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        
        # 6. 进度显示
        progress_frame = ttk.Frame(settings_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        # 确保progress_var已经初始化
        if not hasattr(self, 'progress_var'):
            self.progress_var = tk.DoubleVar()
            self.progress_var.set(0)
        
        if not hasattr(self, 'status_var'):
            self.status_var = tk.StringVar()
            self.status_var.set("就绪")
            
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)
        
        # 7. 日志区域
        log_frame = ttk.LabelFrame(settings_frame, text="优化日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
    
    def run_optimization(self):
        """运行星座优化线程"""
        logger.info("启动星座优化...")
        
        # 获取界面参数
        total_sats_range = (int(self.min_satellites.get()), int(self.max_satellites.get()))
        num_planes_range = (int(self.min_planes.get()), int(self.max_planes.get()))
        inc_range = (float(self.min_inclination.get()), float(self.max_inclination.get()))
        alt_range = (float(self.min_altitude.get()), float(self.max_altitude.get()))
        pop_size = int(self.population_size.get())
        generations = int(self.generations.get())
        algorithm = self.algorithm_var.get()
        
        # 权重参数
        weights = {
            'coverage': float(self.coverage_weight.get()),
            'collision_risk': float(self.collision_weight.get()),
            'cost': float(self.energy_weight.get())  # 修改为'cost'，与evaluate_constellation期望的键一致
        }
        
        logger.info(f"使用多目标优化权重: {weights}")
        
        # 清除上一次结果
        if hasattr(self, 'viz_fig') and self.viz_fig:
            plt.close(self.viz_fig)
            self.viz_fig = None
            self.canvas.get_tk_widget().pack_forget()
        
        # 重置结果显示区域
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        
        # 显示等待提示
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "优化中，请稍候...\n")
        
        # 创建并启动优化线程
        self.opt_thread = threading.Thread(
            target=self._run_optimization_thread,
            args=(total_sats_range, num_planes_range, inc_range, alt_range,
                 pop_size, generations, algorithm, weights)
        )
        self.opt_thread.daemon = True
        self.opt_thread.start()
    
    def log(self, message):
        """日志记录"""
        try:
            if hasattr(self, 'log_text'):
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.window.update_idletasks()
        except Exception as e:
            print(f"记录日志时出错: {e}")
            print(f"日志消息: {message}")
            
    def update_iteration_data(self, iteration, best_score, mean_score, parameters):
        """更新迭代数据"""
        try:
            if hasattr(self, 'iteration_data'):
                self.iteration_data["iterations"].append(iteration)
                self.iteration_data["best_scores"].append(best_score)
                self.iteration_data["mean_scores"].append(mean_score)
                self.iteration_data["parameters"].append(parameters)
        except Exception as e:
            print(f"更新迭代数据时出错: {e}")
            
    def _restore_ui_state(self):
        """恢复UI状态"""
        try:
            self.running = False
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
            if self.result:
                self.apply_btn.config(state="normal")
                
            if hasattr(self, 'status_var'):
                self.status_var.set("就绪")
                
            if hasattr(self, 'progress_var'):
                self.progress_var.set(100)
        except Exception as e:
            print(f"恢复UI状态时出错: {e}")
            
    def stop_optimization(self):
        """停止优化过程"""
        try:
            if hasattr(self, 'optimizer'):
                # 如果优化器有取消方法
                if hasattr(self.optimizer, 'cancel'):
                    self.optimizer.cancel()
            self._restore_ui_state()
        except Exception as e:
            print(f"停止优化时出错: {e}")
    
    def apply_optimal_config(self):
        """应用最优配置"""
        if not self.result:
            messagebox.showinfo("提示", "没有可用的优化结果")
            return
        
        # 将优化结果应用到主应用程序
        try:
            # 获取创建星座的函数
            create_constellation_func = getattr(self.parent, "create_constellation", None)
            
            if create_constellation_func:
                # 调用创建星座函数
                create_constellation_func(
                    total_satellites=self.result["total_satellites"],
                    num_planes=self.result["num_planes"],
                    relative_spacing=self.result["relative_spacing"],
                    inclination=self.result["inclination"],
                    altitude=self.result["altitude"],
                    eccentricity=self.result["eccentricity"]
                )
                
                messagebox.showinfo("成功", "已应用优化配置到星座设计")
                self.window.destroy()
            else:
                messagebox.showerror("错误", "无法在父窗口中找到create_constellation方法")
                
        except Exception as e:
            messagebox.showerror("应用配置错误", f"应用优化配置时发生错误: {str(e)}")
    
    def create_results_display(self, parent):
        """创建结果显示界面"""
        # 创建结果框架
        self.results_frame = ttk.LabelFrame(parent, text="优化结果", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建选项卡界面
        self.tab_control = ttk.Notebook(self.results_frame) # Store as instance variable
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 创建表格选项卡
        tab_table = ttk.Frame(self.tab_control)
        self.tab_control.add(tab_table, text="表格结果")
        
        # 创建迭代图选项卡
        tab_iter = ttk.Frame(self.tab_control)
        self.tab_control.add(tab_iter, text="收敛过程")
        
        # 创建星座图选项卡
        tab_vis = ttk.Frame(self.tab_control)
        self.tab_control.add(tab_vis, text="星座可视化")
        
        # 1. 结果表格
        columns = ('参数', '值')
        self.result_tree = ttk.Treeview(tab_table, columns=columns, show='headings')
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_tree.column('参数', width=150, anchor=tk.W)
        self.result_tree.column('值', width=150, anchor=tk.W)
        self.result_tree.heading('参数', text='参数')
        self.result_tree.heading('值', text='值')
        scrollbar = ttk.Scrollbar(tab_table, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        # 2. 迭代过程图表
        self.iter_fig = Figure(figsize=(6, 5), dpi=100)
        self.iter_canvas = FigureCanvasTkAgg(self.iter_fig, master=tab_iter)
        self.iter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add toolbar for iter plot
        iter_toolbar_frame = ttk.Frame(tab_iter)
        iter_toolbar_frame.pack(fill=tk.X)
        iter_toolbar = NavigationToolbar2Tk(self.iter_canvas, iter_toolbar_frame)
        iter_toolbar.update()
        self.iter_canvas.draw()
        
        # 3. 星座可视化 (3D)
        self.const_fig = Figure(figsize=(6, 5), dpi=100)
        self.const_ax = self.const_fig.add_subplot(111, projection='3d') # Initialize 3D axis
        self.const_canvas = FigureCanvasTkAgg(self.const_fig, master=tab_vis)
        self.const_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
         # Add toolbar for const plot
        const_toolbar_frame = ttk.Frame(tab_vis)
        const_toolbar_frame.pack(fill=tk.X)
        const_toolbar = NavigationToolbar2Tk(self.const_canvas, const_toolbar_frame)
        const_toolbar.update()
        self.const_canvas.draw()
        
        # 注意：viz_fig已经在__init__方法中创建
        # 所以这里不需要重复创建
    
    def _run_optimization_thread(self, objective, algorithm, param_ranges, weights, max_iterations, use_multiprocessing):
        """在单独的线程中运行优化过程"""
        try:
            # 转换参数
            total_satellites_range = param_ranges.get('sat_range', (20, 80))
            num_planes_range = param_ranges.get('plane_range', (3, 8))
            inclination_range = param_ranges.get('inc_range', (45, 90))
            altitude_range = param_ranges.get('alt_range', (500, 1500))
            
            self.log(f"开始优化，目标: {objective}, 算法: {algorithm}")
            self.log(f"卫星总数范围: {total_satellites_range}")
            self.log(f"轨道平面数范围: {num_planes_range}")
            self.log(f"轨道倾角范围: {inclination_range}°")
            self.log(f"轨道高度范围: {altitude_range}km")
            
            # 运行优化
            # optimizer现在返回包含 best_config, best_score, history 的字典
            result_data = self.optimizer.optimize_walker_constellation(
                objective=objective,
                total_satellites_range=total_satellites_range,
                num_planes_range=num_planes_range,
                inclination_range=inclination_range,
                altitude_range=altitude_range,
                weights=weights,
                max_iterations=max_iterations,
                use_multiprocessing=use_multiprocessing,
                algorithm=algorithm
                # **algorithm_params 可以根据需要添加
            )
            
            if result_data:
                # 存储结果
                self.result = result_data['best_config'] # 只存储最优配置参数
                # 将历史数据直接传递给UI更新函数
                self.window.after(0, lambda: self._update_result_display(result_data))
                self.log("优化完成！")
            else:
                self.log("优化未找到有效结果或被取消")
                # 即使失败也要恢复UI状态
                self.window.after(0, self._restore_ui_state)
                
        except Exception as e:
            self.log(f"优化过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错也要恢复UI状态
            self.window.after(0, self._restore_ui_state)
        # finally:
             # 不再需要在 finally 中调用 _restore_ui_state，因为成功/失败/取消路径都已处理
             # self.window.after(100, self._restore_ui_state)
            
    def _update_result_display(self, result_data):
        """更新结果显示 (接收完整的优化结果字典)"""
        if not result_data or not result_data.get('best_config'):
            self.log("更新结果显示：无有效结果数据")
            # 可以考虑禁用应用按钮等
            self.apply_btn.config(state="disabled") 
            return
            
        best_config = result_data.get('best_config')
        final_metrics = result_data.get('final_metrics', {})
        history = result_data.get('history', {})
        best_score = result_data.get('best_score', -1)
        constellation_states = result_data.get('constellation_states')
        
        # 确保UI元素存在
        if not hasattr(self, 'result_tree') or not hasattr(self, 'iter_canvas') or not hasattr(self, 'const_canvas'):
             self.log("错误：结果显示UI元素尚未初始化")
             return
             
        try:
            self.log("开始更新结果显示...")
            # --- 更新结果表格 --- 
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            self.result_tree.insert('', 'end', values=('总得分', f"{best_score:.4f}"))
            self.result_tree.insert('', 'end', values=('卫星总数', best_config.get('total_satellites', 'N/A')))
            self.result_tree.insert('', 'end', values=('轨道平面数', best_config.get('num_planes', 'N/A')))
            self.result_tree.insert('', 'end', values=('相对间距', best_config.get('relative_spacing', 'N/A')))
            self.result_tree.insert('', 'end', values=('轨道倾角(度)', f"{best_config.get('inclination', 0):.2f}"))
            self.result_tree.insert('', 'end', values=('轨道高度(km)', f"{best_config.get('altitude', 0):.2f}"))
            
            # 添加最终评估指标
            cov = final_metrics.get('coverage')
            if cov is not None:
                self.result_tree.insert('', 'end', values=('覆盖率', f"{cov*100:.2f}%"))
            col_risk = final_metrics.get('collision_risk')
            if col_risk is not None:
                self.result_tree.insert('', 'end', values=('碰撞风险', f"{col_risk:.6f}"))
            # 可以添加更多来自 final_metrics 的指标

            self.log("结果表格已更新")

            # --- 更新迭代图表 --- 
            iterations = history.get('iterations', [])
            best_scores_hist = history.get('best_scores', [])
            mean_scores_hist = history.get('mean_scores', [])
            
            self.iter_fig.clear()
            ax_iter = self.iter_fig.add_subplot(111)
            if iterations and best_scores_hist:
                ax_iter.plot(iterations, best_scores_hist, 'b-o', markersize=3, label='最佳得分')
                if mean_scores_hist and len(mean_scores_hist) == len(iterations):
                    ax_iter.plot(iterations, mean_scores_hist, 'r--', label='平均得分')
                ax_iter.set_xlabel('迭代/评估次数')
                ax_iter.set_ylabel('目标函数得分')
                ax_iter.set_title(f'{result_data.get("algorithm", "").upper()} 优化收敛过程')
                ax_iter.legend()
                ax_iter.grid(True)
            else:
                ax_iter.text(0.5, 0.5, '无迭代历史数据', horizontalalignment='center', verticalalignment='center')
            self.iter_fig.tight_layout()
            self.iter_canvas.draw()
            self.log("迭代图表已更新")
            
            # --- 更新星座图 --- 
            self.const_ax.clear()
            if constellation_states is not None and len(constellation_states) > 0:
                try:
                    # 绘制地球
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    radius = 6378.137  # 地球半径(km)
                    x_e = radius * np.outer(np.cos(u), np.sin(v))
                    y_e = radius * np.outer(np.sin(u), np.sin(v))
                    z_e = radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    self.const_ax.plot_surface(x_e, y_e, z_e, color='lightblue', alpha=0.3, rstride=4, cstride=4)
                    
                    # 绘制卫星
                    positions = constellation_states[:, :3] / 1000.0  # 转换为km
                    self.const_ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=15, marker='o')
                    
                    self.const_ax.set_xlabel('X (km)')
                    self.const_ax.set_ylabel('Y (km)')
                    self.const_ax.set_zlabel('Z (km)')
                    self.const_ax.set_title('优化星座构型')
                    
                    # 设置坐标轴比例
                    max_range = np.max(np.linalg.norm(positions, axis=1)) * 1.1
                    self.const_ax.set_xlim([-max_range, max_range])
                    self.const_ax.set_ylim([-max_range, max_range])
                    self.const_ax.set_zlim([-max_range, max_range])
                    # self.const_ax.set_aspect('equal') # May cause issues with toolbar
                    self.const_ax.view_init(elev=30., azim=30) # Set initial view angle
                    self.log("星座图已更新")
                except Exception as e:
                    self.log(f"绘制星座图时出错: {str(e)}")
                    self.const_ax.text(0,0,0, '绘制星座图失败', color='red')
            else:
                self.const_ax.text(0,0,0, '无星座状态数据')
                self.log("无星座状态数据可绘制")
                
            self.const_fig.tight_layout()
            self.const_canvas.draw()
            self.log("结果显示更新完成")
            
            # 确保结果显示后才启用应用按钮
            if self.result: # self.result holds the best_config
                self.apply_btn.config(state="normal")

        except Exception as e:
            self.log(f"更新结果显示时发生严重错误: {str(e)}")
            traceback.print_exc()
            # 可以在UI上显示错误消息
            try:
                 self.iter_fig.clear()
                 ax = self.iter_fig.add_subplot(111)
                 ax.text(0.5, 0.5, f'更新显示失败:\n{str(e)}', color='red', ha='center', va='center')
                 self.iter_canvas.draw()
            except: pass # Ignore errors during error display

    def _restore_ui_state(self):
        """恢复UI状态（确保在主线程执行）"""
        def restore():
             try:
                 self.running = False
                 if hasattr(self, 'run_btn') and self.run_btn.winfo_exists():
                     self.run_btn.config(state="normal")
                 if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                     self.stop_btn.config(state="disabled")
                 
                 # Apply button state depends on whether a valid result exists
                 apply_state = "normal" if self.result else "disabled"
                 if hasattr(self, 'apply_btn') and self.apply_btn.winfo_exists():
                      self.apply_btn.config(state=apply_state)
                     
                 if hasattr(self, 'status_var'):
                     self.status_var.set("就绪" if not self.optimizer.is_cancelled else "已取消")
                     
                 if hasattr(self, 'progress_var'):
                     self.progress_var.set(100 if not self.optimizer.is_cancelled else 0)
                 self.log("UI状态已恢复")
             except tk.TclError as e:
                 print(f"Restore UI TclError: {e}")
             except Exception as e:
                 print(f"恢复UI状态时出错: {e}")
        
        if hasattr(self, 'window') and self.window.winfo_exists():
             self.window.after(0, restore)
        else:
             print("Restore UI: Window not ready")

    def stop_optimization(self):
        """停止优化过程"""
        self.log("尝试停止优化...")
        if self.running and hasattr(self, 'optimizer'):
            try:
                self.optimizer.cancel() # 设置取消标志
                if hasattr(self, 'stop_btn'): # 禁用停止按钮
                    self.stop_btn.config(state="disabled")
                self.status_var.set("正在取消...") 
            except Exception as e:
                self.log(f"停止优化时出错: {e}")
        else:
             self.log("优化未在运行")

    def apply_optimal_config(self):
        """应用最优配置"""
        if not self.result: # self.result现在存储 best_config
            messagebox.showinfo("提示", "没有可用的优化结果")
            return
        
        self.log(f"尝试应用优化配置: {self.result}")
        # 将优化结果应用到主应用程序
        try:
            # 获取创建星座的函数 (从父窗口，即 EnhancedConstellationApp)
            if not hasattr(self.parent, 'create_constellation'):
                 messagebox.showerror("错误", "无法在主应用中找到 create_constellation 方法")
                 return
                 
            create_constellation_func = self.parent.create_constellation
            
            # 准备参数 (从 self.result 中提取)
            params_to_apply = {
                'total_satellites': self.result.get('total_satellites'),
                'num_planes': self.result.get('num_planes'),
                'relative_spacing': self.result.get('relative_spacing'),
                'inclination': self.result.get('inclination'),
                'altitude': self.result.get('altitude'),
                'eccentricity': self.result.get('eccentricity', 0.0) # Use default if not present
            }
            
            # 检查参数是否有效
            if any(v is None for v in params_to_apply.values()):
                 messagebox.showerror("错误", f"优化结果参数不完整: {params_to_apply}")
                 return

            self.log(f"调用主应用的 create_constellation 使用参数: {params_to_apply}")
            # 调用主应用的创建星座函数
            create_constellation_func(**params_to_apply)
            
            messagebox.showinfo("成功", "已应用优化配置到星座设计")
            self.window.destroy() # 关闭优化窗口
                
        except Exception as e:
            self.log(f"应用优化配置时发生错误: {str(e)}")
            messagebox.showerror("应用配置错误", f"应用优化配置时发生错误: {str(e)}")
            traceback.print_exc()

class ErrorAnalysisWindow:
    """误差分析窗口"""
    
    def __init__(self, parent, title="误差分析"):
        """初始化误差分析窗口"""
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1000x800")
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        
        # 创建主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制区域
        control_frame = ttk.LabelFrame(main_frame, text="分析设置", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建方法选择
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="数值方法:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="RK45")
        methods = ["RK45", "RK4", "Verlet", "Euler", "Midpoint", "DOP853"]
        self.method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, values=methods, width=15)
        self.method_combo.pack(side=tk.LEFT, padx=5)
        
        # 轨道持续时间
        duration_frame = ttk.Frame(control_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(duration_frame, text="持续时间(天):").pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=7.0)
        ttk.Entry(duration_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # 步长设置
        step_frame = ttk.Frame(control_frame)
        step_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(step_frame, text="步长(秒):").pack(side=tk.LEFT)
        self.step_var = tk.DoubleVar(value=60.0)
        ttk.Entry(step_frame, textvariable=self.step_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # 分析按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.run_btn = ttk.Button(button_frame, text="运行分析", command=self.run_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="保存结果", command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建图形区域
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # 结果存储
        self.results = None
    
    def run_analysis(self):
        """运行误差分析"""
        try:
            # 获取参数
            method = self.method_var.get()
            duration = self.duration_var.get() * 86400  # 转换为秒
            step_size = self.step_var.get()
            
            # 禁用运行按钮
            self.run_btn.config(state=tk.DISABLED)
            
            # 导入数值方法模块
            from numerical_methods import NumericalIntegrators, ErrorAnalysis
            
            # 创建简单的二体运动问题
            mu = 3.986004418e14  # 地球引力常数
            
            # 初始状态 (LEO轨道)
            r0 = np.array([7000e3, 0, 0])  # 初始位置 [m]
            v0 = np.array([0, np.sqrt(mu/np.linalg.norm(r0)), 0])  # 初始速度 [m/s]
            initial_state = np.concatenate([r0, v0])
            
            # 定义加速度函数 (简单的二体运动)
            def acceleration(t, state):
                r = state[:3]
                r_norm = np.linalg.norm(r)
                if r_norm == 0:
                    return np.zeros(3)
                a = -mu * r / r_norm**3
                return a
            
            # 运行选定的方法
            method_func = getattr(NumericalIntegrators, f"{method.lower()}_method", None)
            
            if method_func is None:
                if method == "RK45":
                    method_func = NumericalIntegrators.adaptive_rk45
                else:
                    raise ValueError(f"未找到方法: {method}")
            
            # 计算步数
            steps = int(duration / step_size)
            t_span = (0, duration)
            
            # 运行数值积分
            t, states = method_func(acceleration, t_span, initial_state, steps)
            
            # 计算误差
            energy_errors = ErrorAnalysis.energy_conservation_error(states, mu)
            momentum_errors = ErrorAnalysis.angular_momentum_error(states)
            
            # 转换时间为天
            time_days = t / 86400
            
            # 创建误差分析图表
            self.create_error_analysis_plot(energy_errors, momentum_errors, time_days, method)
            
            # 保存结果
            self.results = {
                'method': method,
                'time_days': time_days,
                'energy_errors': energy_errors,
                'momentum_errors': momentum_errors
            }
            
            # 启用保存按钮
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("错误", f"运行误差分析时出错: {str(e)}")
            traceback.print_exc()
        finally:
            # 重新启用运行按钮
            self.run_btn.config(state=tk.NORMAL)
    
    def create_error_analysis_plot(self, energy_errors, momentum_errors, time_days, method_name):
        """创建误差分析图表"""
        # 清空图形
        self.fig.clear()
        
        # 创建子图
        ax1 = self.fig.add_subplot(211)
        ax1.plot(time_days, abs(energy_errors), 'r-', linewidth=1.5)
        ax1.set_ylabel('能量相对误差')
        ax1.set_title(f'{method_name}方法的能量守恒误差')
        ax1.grid(True)
        ax1.set_yscale('log')
        
        ax2 = self.fig.add_subplot(212)
        ax2.plot(time_days, abs(momentum_errors), 'b-', linewidth=1.5)
        ax2.set_xlabel('时间 (天)')
        ax2.set_ylabel('角动量相对误差')
        ax2.set_title(f'{method_name}方法的角动量守恒误差')
        ax2.grid(True)
        ax2.set_yscale('log')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_results(self):
        """保存分析结果"""
        if not self.results:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
            
        # 获取保存文件路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="保存误差分析结果"
        )
        
        if not file_path:
            return
            
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                'Time (days)': self.results['time_days'],
                'Energy Error': self.results['energy_errors'],
                'Momentum Error': self.results['momentum_errors']
            })
            
            # 保存为CSV
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("成功", f"结果已保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存结果时出错: {str(e)}")


def add_menubar(root, callbacks):
    """添加菜单栏"""
    menubar = tk.Menu(root)
    
    # 文件菜单
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="保存星座", command=callbacks.get("save_constellation"))
    filemenu.add_command(label="加载星座", command=callbacks.get("load_constellation"))
    filemenu.add_separator()
    filemenu.add_command(label="导出数据", command=callbacks.get("export_data"))
    filemenu.add_separator()
    filemenu.add_command(label="退出", command=root.quit)
    menubar.add_cascade(label="文件", menu=filemenu)
    
    # 分析菜单
    analysismenu = tk.Menu(menubar, tearoff=0)
    analysismenu.add_command(label="轨道演化分析", command=callbacks.get("orbit_evolution"))
    analysismenu.add_command(label="数值方法对比", command=callbacks.get("compare_methods"))
    analysismenu.add_command(label="摄动力分析", command=callbacks.get("perturbation_analysis"))
    analysismenu.add_command(label="碰撞风险分析", command=callbacks.get("collision_analysis"))
    analysismenu.add_command(label="覆盖率分析", command=callbacks.get("coverage_analysis"))
    analysismenu.add_command(label="误差分析", command=callbacks.get("error_analysis"))
    menubar.add_cascade(label="分析", menu=analysismenu)
    
    # 工具菜单
    toolsmenu = tk.Menu(menubar, tearoff=0)
    toolsmenu.add_command(label="星座优化", command=callbacks.get("optimize_constellation"))
    toolsmenu.add_command(label="设置", command=callbacks.get("settings"))
    menubar.add_cascade(label="工具", menu=toolsmenu)
    
    # 帮助菜单
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="帮助", command=callbacks.get("show_help"))
    helpmenu.add_command(label="关于", command=callbacks.get("show_about"))
    menubar.add_cascade(label="帮助", menu=helpmenu)
    
    root.config(menu=menubar)
    return menubar 