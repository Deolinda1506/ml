"""
Monitoring and Metrics for Glaucoma Detection System
===================================================

This module provides monitoring capabilities including:
- Request metrics
- Model performance metrics
- System health monitoring
- Prometheus metrics export
"""

import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict, deque
import functools
import asyncio

class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.model_predictions = defaultdict(int)
        self.system_metrics = deque(maxlen=max_history)
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
    
    def record_request(self, endpoint: str, method: str, response_time: float, 
                      status_code: int, prediction: Optional[str] = None):
        """Record a request metric."""
        timestamp = datetime.now()
        
        # Record request time
        self.request_times.append({
            'timestamp': timestamp,
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code
        })
        
        # Record request count
        self.request_counts[f"{method}_{endpoint}"] += 1
        
        # Record errors
        if status_code >= 400:
            self.error_counts[f"{method}_{endpoint}_{status_code}"] += 1
        
        # Record predictions
        if prediction:
            self.model_predictions[prediction] += 1
    
    def _monitor_system(self):
        """Background system monitoring."""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.system_metrics.append({
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'disk_percent': disk.percent,
                    'disk_free': disk.free
                })
                
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                time.sleep(60)
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all metrics."""
        if not self.request_times:
            return {}
        
        recent_requests = list(self.request_times)[-100:]  # Last 100 requests
        
        response_times = [r['response_time'] for r in recent_requests]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Calculate error rate
        total_requests = len(recent_requests)
        error_requests = len([r for r in recent_requests if r['status_code'] >= 400])
        error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Get system metrics
        latest_system = list(self.system_metrics)[-1] if self.system_metrics else {}
        
        return {
            'request_metrics': {
                'total_requests': total_requests,
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time,
                'error_rate': error_rate,
                'requests_per_endpoint': dict(self.request_counts),
                'errors_per_endpoint': dict(self.error_counts)
            },
            'model_metrics': {
                'predictions_by_class': dict(self.model_predictions),
                'total_predictions': sum(self.model_predictions.values())
            },
            'system_metrics': latest_system
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = []
        
        # Request metrics
        summary = self.get_metrics_summary()
        if summary:
            req_metrics = summary.get('request_metrics', {})
            
            metrics.append(f"# HELP glaucoma_requests_total Total number of requests")
            metrics.append(f"# TYPE glaucoma_requests_total counter")
            for endpoint, count in req_metrics.get('requests_per_endpoint', {}).items():
                metrics.append(f'glaucoma_requests_total{{endpoint="{endpoint}"}} {count}')
            
            metrics.append(f"# HELP glaucoma_response_time_seconds Response time in seconds")
            metrics.append(f"# TYPE glaucoma_response_time_seconds histogram")
            avg_time = req_metrics.get('avg_response_time', 0)
            metrics.append(f'glaucoma_response_time_seconds{{quantile="0.5"}} {avg_time}')
            
            metrics.append(f"# HELP glaucoma_error_rate_percent Error rate percentage")
            metrics.append(f"# TYPE glaucoma_error_rate_percent gauge")
            error_rate = req_metrics.get('error_rate', 0)
            metrics.append(f'glaucoma_error_rate_percent {error_rate}')
            
            # Model metrics
            model_metrics = summary.get('model_metrics', {})
            metrics.append(f"# HELP glaucoma_predictions_total Total number of predictions")
            metrics.append(f"# TYPE glaucoma_predictions_total counter")
            for prediction, count in model_metrics.get('predictions_by_class', {}).items():
                metrics.append(f'glaucoma_predictions_total{{class="{prediction}"}} {count}')
            
            # System metrics
            sys_metrics = summary.get('system_metrics', {})
            if sys_metrics:
                metrics.append(f"# HELP glaucoma_cpu_percent CPU usage percentage")
                metrics.append(f"# TYPE glaucoma_cpu_percent gauge")
                metrics.append(f'glaucoma_cpu_percent {sys_metrics.get("cpu_percent", 0)}')
                
                metrics.append(f"# HELP glaucoma_memory_percent Memory usage percentage")
                metrics.append(f"# TYPE glaucoma_memory_percent gauge")
                metrics.append(f'glaucoma_memory_percent {sys_metrics.get("memory_percent", 0)}')
        
        return '\n'.join(metrics)

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector

class PerformanceMonitor:
    """Context manager for monitoring request performance."""
    
    def __init__(self, endpoint: str, method: str = "GET"):
        self.endpoint = endpoint
        self.method = method
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            response_time = time.time() - self.start_time
            status_code = 500 if exc_type else 200
            metrics_collector.record_request(
                self.endpoint, 
                self.method, 
                response_time, 
                status_code
            )

def monitor_request(endpoint: str, method: str = "GET"):
    """Decorator for monitoring request performance, supporting sync and async."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                try:
                    result = await func(*args, **kwargs)
                except Exception:
                    status_code = 500
                    raise
                finally:
                    response_time = time.time() - start_time
                    metrics_collector.record_request(endpoint, method, response_time, status_code)
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    status_code = 500
                    raise
                finally:
                    response_time = time.time() - start_time
                    metrics_collector.record_request(endpoint, method, response_time, status_code)
                return result
            return sync_wrapper
    return decorator
