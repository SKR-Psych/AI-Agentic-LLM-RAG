"""Performance profiling and optimization tools."""

import time
import cProfile
import pstats
import io
import functools
from typing import Callable, Any, Dict
import psutil
import os
import sys
import numpy as np

class PerformanceProfiler:
    """Profile performance of functions and methods."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                profiler.disable()
            
            # Collect profiling data
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'function_name': func.__name__,
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'profile_stats': s.getvalue(),
                'timestamp': time.time()
            }
            
            self.profiles[func.__name__] = profile_data
            return result
        
        return wrapper
    
    def benchmark_function(self, func: Callable, iterations: int = 1000, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a function with multiple iterations."""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'function_name': func.__name__,
            'iterations': iterations,
            'total_time': sum(times),
            'average_time': np.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': np.std(times),
            'total_memory': sum(memory_usage),
            'average_memory': np.mean(memory_usage),
            'result': result
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        try:
            report = {
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'platform': sys.platform
                },
                'profiles': self.profiles,
                'recommendations': self._generate_recommendations()
            }
            
            filename = f"performance_report_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return filename
        except Exception as e:
            return f"Error generating report: {e}"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        for func_name, profile in self.profiles.items():
            if profile['execution_time'] > 1.0:  # More than 1 second
                recommendations.append(f"Consider optimizing {func_name} - execution time: {profile['execution_time']:.3f}s")
            
            if profile['memory_delta'] > 100:  # More than 100MB
                recommendations.append(f"Memory leak detected in {func_name} - delta: {profile['memory_delta']:.1f}MB")
        
        return recommendations

# Usage example
profiler = PerformanceProfiler()

@profiler.profile_function
def slow_function():
    """Example slow function for profiling."""
    time.sleep(0.1)
    return sum(range(1000000))

# Benchmark the function
result = profiler.benchmark_function(slow_function, iterations=10)
print(json.dumps(result, indent=2))


def fetch_data():
    # TODO: logic pending
    pass



def calculate_():
    # TODO: logic pending
    pass

