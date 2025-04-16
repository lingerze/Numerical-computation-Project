"""
并行计算模块，提供多线程、多进程支持和高效计算工具
用于优化卫星轨道计算和星座设计的性能
"""

import os
import sys
import numpy as np
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

class ParallelComputing:
    """提供并行计算工具和优化方法的类"""
    
    def __init__(self, max_workers=None, use_processes=True):
        """
        初始化并行计算环境
        
        参数:
        - max_workers: 最大工作线程/进程数，None表示使用系统CPU核心数
        - use_processes: 是否使用多进程而非多线程
        """
        # 获取CPU核心数
        self.cpu_count = multiprocessing.cpu_count()
        
        # 确定工作线程/进程数
        if max_workers is None:
            # 默认使用系统核心数减1，保留一个核心给OS
            self.max_workers = max(1, self.cpu_count - 1)
        else:
            self.max_workers = max(1, min(max_workers, self.cpu_count))
        
        # 是否使用多进程
        self.use_processes = use_processes
        
        # 创建执行器
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 任务计数器和锁
        self.task_count = 0
        self.lock = threading.Lock()
        
        # 进度回调
        self.progress_callback = None
    
    def __del__(self):
        """析构函数，确保执行器被正确关闭"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, completed, total, message=""):
        """更新进度"""
        if self.progress_callback:
            progress = int(100 * completed / total) if total > 0 else 0
            self.progress_callback(progress, message)
    
    def map(self, func, iterable, chunksize=1):
        """
        并行映射函数到可迭代对象的每个元素
        
        参数:
        - func: 要应用的函数
        - iterable: 可迭代对象
        - chunksize: 批处理大小，用于提高性能
        
        返回:
        - 结果列表
        """
        # 创建任务列表
        items = list(iterable)
        total_items = len(items)
        
        if total_items == 0:
            return []
        
        # 进度更新封装
        def wrapped_func(item, index):
            result = func(item)
            with self.lock:
                self.task_count += 1
                self._update_progress(self.task_count, total_items, 
                                    f"已完成 {self.task_count}/{total_items} 任务")
            return result
        
        # 重置计数器
        self.task_count = 0
        
        # 创建带索引的任务
        tasks = [(item, i) for i, item in enumerate(items)]
        
        # 执行并行映射
        results = list(self.executor.map(lambda x: wrapped_func(*x), tasks, chunksize=chunksize))
        
        return results
    
    def batch_process(self, func, data, batch_size=None):
        """
        批处理大型数据集，适用于内存受限的情况
        
        参数:
        - func: 处理函数
        - data: 要处理的数据列表
        - batch_size: 批处理大小，None表示自动确定
        
        返回:
        - 合并后的结果
        """
        if not data:
            return []
            
        # 自动确定批处理大小
        if batch_size is None:
            # 默认每个核心处理2批
            batch_size = max(1, len(data) // (self.max_workers * 2))
            # 确保批大小至少为1，最多为数据长度
            batch_size = max(1, min(batch_size, len(data)))
        
        # 分割数据
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        # 定义批处理函数
        def process_batch(batch):
            return [func(item) for item in batch]
        
        # 并行处理批
        results = self.map(process_batch, batches)
        
        # 合并结果
        flattened_results = []
        for batch_result in results:
            flattened_results.extend(batch_result)
        
        return flattened_results
    
    def parallel_for(self, func, start, end, step=1, **kwargs):
        """
        并行for循环实现
        
        参数:
        - func: 要执行的函数，接受循环索引和可选的kwargs
        - start: 起始索引
        - end: 结束索引（不包括）
        - step: 步长
        - **kwargs: 传递给func的额外参数
        
        返回:
        - 结果列表
        """
        # 创建范围
        indices = range(start, end, step)
        
        # 包装函数
        def wrapped_func(idx):
            return func(idx, **kwargs)
        
        # 使用map执行并行计算
        return self.map(wrapped_func, indices)
    
    @staticmethod
    def chunked_array(array, chunks):
        """
        将数组分割为指定数量的块
        
        参数:
        - array: 要分割的数组
        - chunks: 块数量
        
        返回:
        - 分割后的数组列表
        """
        n = len(array)
        chunk_size = max(1, n // chunks)
        return [array[i:i+chunk_size] for i in range(0, n, chunk_size)]
    
    def parallel_map_reduce(self, map_func, reduce_func, data, initial_value=None):
        """
        实现并行Map-Reduce模式
        
        参数:
        - map_func: 映射函数
        - reduce_func: 归约函数
        - data: 输入数据
        - initial_value: 归约的初始值
        
        返回:
        - 归约结果
        """
        # 映射阶段
        mapped_data = self.map(map_func, data)
        
        # 归约阶段
        if initial_value is not None:
            result = initial_value
            for item in mapped_data:
                result = reduce_func(result, item)
        else:
            # 如果没有初始值，使用第一个元素作为初始值
            if not mapped_data:
                return None
            result = mapped_data[0]
            for item in mapped_data[1:]:
                result = reduce_func(result, item)
        
        return result

class Vectorization:
    """提供向量化和GPU加速计算的工具类"""
    
    @staticmethod
    def is_numpy_available():
        """检查是否可以使用NumPy"""
        return True  # 假设已导入numpy
    
    @staticmethod
    def batch_apply(func, batch, axis=0):
        """
        将函数应用于批量数据
        
        参数:
        - func: 要应用的函数
        - batch: 批量数据（NumPy数组）
        - axis: 应用函数的轴
        
        返回:
        - 应用函数后的结果
        """
        if not Vectorization.is_numpy_available():
            raise ImportError("NumPy不可用")
        
        return np.apply_along_axis(func, axis, batch)
    
    @staticmethod
    def vectorized_haversine(lat1, lon1, lat2, lon2):
        """
        向量化的Haversine公式计算球面距离
        
        参数:
        - lat1, lon1: 第一点的纬度和经度（弧度）
        - lat2, lon2: 第二点的纬度和经度（弧度）
        
        返回:
        - 两点间的球面距离（千米）
        """
        if not Vectorization.is_numpy_available():
            raise ImportError("NumPy不可用")
        
        earth_radius = 6371.0  # 地球平均半径（千米）
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return earth_radius * c
    
    @staticmethod
    def optimize_array_operations(func):
        """
        装饰器：优化数组操作
        
        参数:
        - func: 要优化的函数
        
        返回:
        - 优化后的函数
        """
        def wrapper(*args, **kwargs):
            # 转换列表为NumPy数组以提高性能
            new_args = []
            for arg in args:
                if isinstance(arg, list):
                    new_args.append(np.array(arg))
                else:
                    new_args.append(arg)
            
            # 执行函数
            result = func(*new_args, **kwargs)
            
            return result
        
        return wrapper

class MemoryOptimization:
    """提供内存优化工具的类"""
    
    @staticmethod
    def get_size(obj, seen=None):
        """
        递归获取对象大小
        
        参数:
        - obj: 要计算大小的对象
        - seen: 已计算大小的对象集合
        
        返回:
        - 对象大小（字节）
        """
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        # 标记为已访问
        seen.add(obj_id)
        
        # 递归处理容器类型
        if isinstance(obj, dict):
            size += sum(MemoryOptimization.get_size(k, seen) + 
                       MemoryOptimization.get_size(v, seen) 
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(MemoryOptimization.get_size(i, seen) for i in obj)
        
        return size
    
    @staticmethod
    def memory_usage_info():
        """
        获取当前进程的内存使用信息
        
        返回:
        - 内存使用信息字典
        """
        import psutil
        process = psutil.Process(os.getpid())
        return {
            'rss': process.memory_info().rss / (1024 * 1024),  # 常驻内存（MB）
            'vms': process.memory_info().vms / (1024 * 1024),  # 虚拟内存（MB）
            'percent': process.memory_percent()                # 内存使用百分比
        }
    
    @staticmethod
    def optimize_dtype(array, preserve_precision=True):
        """
        优化NumPy数组的数据类型以减小内存使用
        
        参数:
        - array: 要优化的NumPy数组
        - preserve_precision: 是否保持精度
        
        返回:
        - 优化后的数组
        """
        if not isinstance(array, np.ndarray):
            return array
            
        # 获取当前数据类型
        current_dtype = array.dtype
        
        # 对于浮点数组
        if np.issubdtype(current_dtype, np.floating):
            # 检查是否需要64位精度
            needs_high_precision = preserve_precision and np.issubdtype(current_dtype, np.float64)
            
            # 如果需要高精度，保持float64，否则尝试降低精度
            if needs_high_precision:
                return array
            else:
                # 检查值范围，决定是否可以使用float32
                min_val = np.min(array)
                max_val = np.max(array)
                
                if min_val > -3.4e38 and max_val < 3.4e38:
                    return array.astype(np.float32)
                else:
                    return array
        
        # 对于整数数组
        elif np.issubdtype(current_dtype, np.integer):
            # 确定所需位数
            min_val = np.min(array)
            max_val = np.max(array)
            
            # 选择合适的整数类型
            if min_val >= 0:
                if max_val < 256:
                    return array.astype(np.uint8)
                elif max_val < 65536:
                    return array.astype(np.uint16)
                elif max_val < 4294967296:
                    return array.astype(np.uint32)
                else:
                    return array
            else:
                if min_val >= -128 and max_val < 128:
                    return array.astype(np.int8)
                elif min_val >= -32768 and max_val < 32768:
                    return array.astype(np.int16)
                elif min_val >= -2147483648 and max_val < 2147483648:
                    return array.astype(np.int32)
                else:
                    return array
        
        # 其他类型保持不变
        return array 