# utils/profiler.py
import time
import psutil
import os
import numpy as np
from collections import deque

class Profiler:
    def __init__(self, window_size=30):
        self.timers = {}
        self.history = {}
        self.window_size = window_size
        self.process = psutil.Process(os.getpid()) # Lấy process hiện tại
        self.start_time = time.time()

    def start(self, name):
        """Bắt đầu bấm giờ cho tác vụ 'name'"""
        self.timers[name] = time.perf_counter()

    def stop(self, name):
        """Dừng bấm giờ và lưu kết quả"""
        if name in self.timers:
            elapsed = (time.perf_counter() - self.timers[name]) * 1000 # Đổi sang ms
            if name not in self.history:
                self.history[name] = deque(maxlen=self.window_size)
            self.history[name].append(elapsed)

    def get_stats(self):
        """Trả về thống kê trung bình"""
        stats = {}
        for name, values in self.history.items():
            stats[name] = np.mean(values)
        
        # Lấy thông tin CPU/RAM của riêng process này
        # interval=None để không chặn luồng (non-blocking)
        cpu_percent = self.process.cpu_percent(interval=None) 
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024 # Convert to MB
        
        return stats, cpu_percent, mem_mb

    def print_report(self, frame_idx):
        """In báo cáo mỗi 30 frame"""
        if frame_idx % 30 != 0: return

        stats, cpu, mem = self.get_stats()
        print("\n" + "="*40)
        print(f"📊 PERFORMANCE REPORT (Frame {frame_idx})")
        print(f"🖥️  CPU Usage: {cpu:.1f}% | 🧠 RAM: {mem:.1f} MB")
        print("-" * 40)
        print(f"{'Task Name':<20} | {'Avg Time (ms)':<10}")
        print("-" * 40)
        
        total_time = 0
        for name, avg_ms in stats.items():
            print(f"{name:<20} | {avg_ms:.2f} ms")
            if name != "Total_Frame": # Không cộng tổng frame vào thành phần con
                total_time += avg_ms
        
        print("-" * 40)
        print(f"Pipeline Latency     | {total_time:.2f} ms")
        print(f"Est. FPS (Pipeline)  | {1000 / (total_time + 1e-5):.1f} FPS")
        print("="*40 + "\n")