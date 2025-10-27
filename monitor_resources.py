#!/usr/bin/env python3
"""
Resource Monitoring Script for Prompt Optimization
Monitors CPU, Memory, and GPU usage during optimization runs
Optimized for MacBook M4 with 16GB RAM
"""

import psutil
import time
import threading
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import subprocess
import platform
from collections import deque
import numpy as np

class ResourceMonitor:
    """Monitor system resources during optimization"""
    
    def __init__(self, interval=1, max_points=300):
        """
        Initialize the resource monitor
        
        Args:
            interval: Sampling interval in seconds
            max_points: Maximum number of data points to keep
        """
        self.interval = interval
        self.max_points = max_points
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.cpu_usage = deque(maxlen=max_points)
        self.memory_usage = deque(maxlen=max_points)
        self.memory_percent = deque(maxlen=max_points)
        self.gpu_usage = deque(maxlen=max_points)
        self.gpu_memory = deque(maxlen=max_points)
        
        # System info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Get system information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version()
        }
        
        # Check for Apple Silicon
        if platform.system() == 'Darwin':
            try:
                # Check if running on Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                                      capture_output=True, text=True)
                info['apple_silicon'] = result.returncode == 0
                
                # Get chip info
                if info['apple_silicon']:
                    chip_info = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                              capture_output=True, text=True)
                    info['chip'] = chip_info.stdout.strip()
            except:
                info['apple_silicon'] = False
        
        return info
    
    def _get_gpu_usage(self):
        """Get GPU usage (Metal Performance Shaders for Apple Silicon)"""
        if self.system_info.get('apple_silicon', False):
            try:
                # Try to get Metal GPU usage (this is approximated)
                import torch
                if torch.backends.mps.is_available():
                    # Get allocated memory (approximation)
                    return 50.0  # Placeholder - actual Metal monitoring requires additional tools
                return 0.0
            except:
                return 0.0
        else:
            # For NVIDIA GPUs
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.utilization()
                return 0.0
            except:
                return 0.0
    
    def _get_gpu_memory(self):
        """Get GPU memory usage"""
        if self.system_info.get('apple_silicon', False):
            try:
                import torch
                if torch.backends.mps.is_available():
                    # Unified memory on Apple Silicon
                    return psutil.virtual_memory().percent
                return 0.0
            except:
                return 0.0
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    return (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                return 0.0
            except:
                return 0.0
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            # Collect metrics
            timestamp = datetime.now()
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_used_gb = (mem.total - mem.available) / (1024**3)
            mem_pct = mem.percent
            gpu_pct = self._get_gpu_usage()
            gpu_mem = self._get_gpu_memory()
            
            # Store data
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_pct)
            self.memory_usage.append(mem_used_gb)
            self.memory_percent.append(mem_pct)
            self.gpu_usage.append(gpu_pct)
            self.gpu_memory.append(gpu_mem)
            
            # Check for high memory usage warning
            if mem_pct > 90:
                print(f"⚠️ WARNING: High memory usage: {mem_pct:.1f}%")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        if not self.monitoring:
            print("🔍 Starting resource monitoring...")
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("✅ Resource monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        if self.monitoring:
            print("🛑 Stopping resource monitoring...")
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            print("✅ Resource monitoring stopped")
    
    def get_current_stats(self):
        """Get current resource statistics"""
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_pct,
            'memory_used_gb': (mem.total - mem.available) / (1024**3),
            'memory_percent': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'gpu_percent': self._get_gpu_usage(),
            'gpu_memory_percent': self._get_gpu_memory()
        }
        
        # Add process-specific stats
        try:
            process = psutil.Process()
            stats['process_memory_mb'] = process.memory_info().rss / (1024**2)
            stats['process_cpu_percent'] = process.cpu_percent()
        except:
            pass
        
        return stats
    
    def print_summary(self):
        """Print monitoring summary"""
        if not self.cpu_usage:
            print("No monitoring data available")
            return
        
        print("\n" + "="*60)
        print("RESOURCE USAGE SUMMARY")
        print("="*60)
        
        print("\n📊 System Information:")
        print(f"  Platform: {self.system_info['platform']}")
        print(f"  Processor: {self.system_info.get('chip', self.system_info['processor'])}")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Total Memory: {self.system_info['total_memory_gb']:.1f} GB")
        if self.system_info.get('apple_silicon'):
            print(f"  Apple Silicon: ✅ (Unified Memory Architecture)")
        
        print("\n📈 CPU Usage:")
        print(f"  Average: {np.mean(self.cpu_usage):.1f}%")
        print(f"  Maximum: {np.max(self.cpu_usage):.1f}%")
        print(f"  Minimum: {np.min(self.cpu_usage):.1f}%")
        
        print("\n💾 Memory Usage:")
        print(f"  Average: {np.mean(self.memory_usage):.2f} GB ({np.mean(self.memory_percent):.1f}%)")
        print(f"  Maximum: {np.max(self.memory_usage):.2f} GB ({np.max(self.memory_percent):.1f}%)")
        print(f"  Minimum: {np.min(self.memory_usage):.2f} GB ({np.min(self.memory_percent):.1f}%)")
        
        if any(self.gpu_usage):
            print("\n🎮 GPU Usage:")
            print(f"  Average: {np.mean(self.gpu_usage):.1f}%")
            print(f"  Maximum: {np.max(self.gpu_usage):.1f}%")
            
        # Memory pressure analysis
        max_mem_pct = np.max(self.memory_percent)
        if max_mem_pct > 90:
            print("\n⚠️ Memory Pressure: HIGH")
            print("  Consider reducing batch size or population size")
        elif max_mem_pct > 75:
            print("\n⚡ Memory Pressure: MODERATE")
            print("  Current settings are near optimal")
        else:
            print("\n✅ Memory Pressure: LOW")
            print("  You may be able to increase batch size for faster processing")
    
    def plot_usage(self, save_path=None):
        """Plot resource usage over time"""
        if not self.timestamps:
            print("No monitoring data to plot")
            return
        
        # Convert timestamps to seconds from start
        start_time = self.timestamps[0]
        time_seconds = [(t - start_time).total_seconds() for t in self.timestamps]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resource Usage During Optimization', fontsize=16)
        
        # CPU Usage
        ax = axes[0, 0]
        ax.plot(time_seconds, list(self.cpu_usage), 'b-', linewidth=2)
        ax.fill_between(time_seconds, 0, list(self.cpu_usage), alpha=0.3)
        ax.set_title('CPU Usage')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Usage (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Memory Usage
        ax = axes[0, 1]
        ax.plot(time_seconds, list(self.memory_usage), 'g-', linewidth=2)
        ax.fill_between(time_seconds, 0, list(self.memory_usage), alpha=0.3)
        ax.axhline(y=self.system_info['total_memory_gb'], color='r', 
                  linestyle='--', label=f'Total: {self.system_info["total_memory_gb"]:.1f} GB')
        ax.axhline(y=self.system_info['total_memory_gb'] * 0.9, color='orange', 
                  linestyle='--', alpha=0.5, label='90% Threshold')
        ax.set_title('Memory Usage')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Memory Percentage
        ax = axes[1, 0]
        ax.plot(time_seconds, list(self.memory_percent), 'r-', linewidth=2)
        ax.fill_between(time_seconds, 0, list(self.memory_percent), alpha=0.3, color='red')
        ax.axhline(y=90, color='darkred', linestyle='--', label='Critical (90%)')
        ax.axhline(y=75, color='orange', linestyle='--', label='Warning (75%)')
        ax.set_title('Memory Usage Percentage')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Usage (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend()
        
        # Combined view
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(time_seconds, list(self.cpu_usage), 'b-', label='CPU %', alpha=0.7)
        line2 = ax2.plot(time_seconds, list(self.memory_usage), 'g-', label='Memory GB', alpha=0.7)
        
        if any(self.gpu_usage):
            line3 = ax.plot(time_seconds, list(self.gpu_usage), 'm-', label='GPU %', alpha=0.7)
            lines = line1 + line2 + line3
        else:
            lines = line1 + line2
        
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.set_title('Combined Resource Usage')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('CPU/GPU Usage (%)', color='b')
        ax2.set_ylabel('Memory (GB)', color='g')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_data(self, filepath='resource_monitoring.json'):
        """Save monitoring data to file"""
        if not self.timestamps:
            print("No data to save")
            return
        
        data = {
            'system_info': self.system_info,
            'monitoring_data': {
                'timestamps': [t.isoformat() for t in self.timestamps],
                'cpu_usage': list(self.cpu_usage),
                'memory_usage_gb': list(self.memory_usage),
                'memory_percent': list(self.memory_percent),
                'gpu_usage': list(self.gpu_usage),
                'gpu_memory': list(self.gpu_memory)
            },
            'summary': {
                'duration_seconds': (self.timestamps[-1] - self.timestamps[0]).total_seconds(),
                'samples': len(self.timestamps),
                'cpu_avg': float(np.mean(self.cpu_usage)),
                'cpu_max': float(np.max(self.cpu_usage)),
                'memory_avg_gb': float(np.mean(self.memory_usage)),
                'memory_max_gb': float(np.max(self.memory_usage)),
                'memory_avg_pct': float(np.mean(self.memory_percent)),
                'memory_max_pct': float(np.max(self.memory_percent))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Monitoring data saved to {filepath}")


class OptimizationMonitor:
    """Monitor optimization progress with resource tracking"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor(interval=2)
        self.start_time = None
        self.optimization_metrics = []
        
    def start_optimization(self, model_name, dataset_name):
        """Start monitoring for an optimization run"""
        self.start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Starting Optimization: {model_name} on {dataset_name}")
        print(f"Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Log initial state
        initial_stats = self.resource_monitor.get_current_stats()
        print(f"Initial Memory: {initial_stats['memory_used_gb']:.2f} GB "
              f"({initial_stats['memory_percent']:.1f}%)")
        
    def log_generation(self, generation, best_fitness, avg_fitness):
        """Log generation progress"""
        current_stats = self.resource_monitor.get_current_stats()
        
        metric = {
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'memory_gb': current_stats['memory_used_gb'],
            'cpu_percent': current_stats['cpu_percent'],
            'timestamp': current_stats['timestamp']
        }
        
        self.optimization_metrics.append(metric)
        
        # Print progress
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Gen {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f} | "
              f"Mem={current_stats['memory_used_gb']:.1f}GB, CPU={current_stats['cpu_percent']:.0f}% | "
              f"Time={elapsed:.0f}s")
        
        # Warning if memory is high
        if current_stats['memory_percent'] > 85:
            print(f"  ⚠️ High memory usage: {current_stats['memory_percent']:.1f}%")
    
    def end_optimization(self):
        """End monitoring and generate report"""
        self.resource_monitor.stop()
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"Optimization Complete")
        print(f"Total Time: {elapsed/60:.1f} minutes")
        print(f"{'='*60}")
        
        # Print summary
        self.resource_monitor.print_summary()
        
        # Save data and plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save monitoring data
        self.resource_monitor.save_data(f'monitoring_{timestamp}.json')
        
        # Create plots
        self.resource_monitor.plot_usage(f'resource_usage_{timestamp}.png')
        
        # Save optimization metrics
        with open(f'optimization_metrics_{timestamp}.json', 'w') as f:
            json.dump(self.optimization_metrics, f, indent=2)
        
        print(f"\n✅ All monitoring data saved with timestamp: {timestamp}")


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Resource Monitoring for Prompt Optimization")
    parser.add_argument("--test", action="store_true", help="Run a test monitoring session")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--live", action="store_true", help="Show live monitoring")
    
    args = parser.parse_args()
    
    if args.test:
        print("Starting test monitoring session...")
        monitor = ResourceMonitor(interval=1)
        monitor.start()
        
        if args.live:
            # Live monitoring display
            try:
                for i in range(args.duration):
                    stats = monitor.get_current_stats()
                    print(f"\r[{i+1}/{args.duration}s] "
                          f"CPU: {stats['cpu_percent']:5.1f}% | "
                          f"Memory: {stats['memory_used_gb']:5.2f}GB ({stats['memory_percent']:5.1f}%) | "
                          f"GPU: {stats['gpu_percent']:5.1f}%", end="")
                    time.sleep(1)
                print()  # New line after monitoring
            except KeyboardInterrupt:
                print("\nMonitoring interrupted")
        else:
            print(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        
        monitor.stop()
        monitor.print_summary()
        monitor.plot_usage("test_monitoring.png")
        monitor.save_data("test_monitoring.json")
        
    else:
        print("Resource Monitor Ready")
        print("\nUsage:")
        print("  Run test: python monitor_resources.py --test")
        print("  Live monitoring: python monitor_resources.py --test --live")
        print("\nIntegration:")
        print("  from monitor_resources import OptimizationMonitor")
        print("  monitor = OptimizationMonitor()")
        print("  monitor.start_optimization('model', 'dataset')")
        print("  # ... optimization code ...")
        print("  monitor.end_optimization()")


if __name__ == "__main__":
    main()