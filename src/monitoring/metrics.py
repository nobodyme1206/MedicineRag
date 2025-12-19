# -*- coding: utf-8 -*-
"""
Prometheus监控指标模块
提供系统性能监控和告警支持
"""

from __future__ import annotations

import time
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricsConfig:
    """监控配置"""
    enabled: bool = True
    port: int = 9090
    prefix: str = "medical_rag"


class MetricsCollector:
    """Prometheus指标收集器"""
    
    def __init__(self, config: Optional[MetricsConfig] = None) -> None:
        self.config = config or MetricsConfig()
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """初始化所有指标"""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        prefix = self.config.prefix
        
        # 请求计数器
        self.request_total = Counter(
            f"{prefix}_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"],
            registry=self.registry
        )
        
        # 请求延迟直方图
        self.request_latency = Histogram(
            f"{prefix}_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # RAG检索指标
        self.retrieval_latency = Histogram(
            f"{prefix}_retrieval_latency_seconds",
            "Retrieval latency in seconds",
            ["method"],  # bm25, vector, hybrid
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.retrieval_results = Histogram(
            f"{prefix}_retrieval_results_count",
            "Number of retrieval results",
            buckets=[1, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        # LLM生成指标
        self.generation_latency = Histogram(
            f"{prefix}_generation_latency_seconds",
            "LLM generation latency in seconds",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.generation_tokens = Histogram(
            f"{prefix}_generation_tokens_count",
            "Number of generated tokens",
            buckets=[50, 100, 200, 500, 1000, 2000],
            registry=self.registry
        )
        
        # 缓存指标
        self.cache_hits = Counter(
            f"{prefix}_cache_hits_total",
            "Total cache hits",
            ["cache_type"],  # query, vector
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            f"{prefix}_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self.registry
        )
        
        # 向量数据库指标
        self.milvus_query_latency = Histogram(
            f"{prefix}_milvus_query_latency_seconds",
            "Milvus query latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.milvus_insert_latency = Histogram(
            f"{prefix}_milvus_insert_latency_seconds",
            "Milvus insert latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # 系统状态
        self.active_connections = Gauge(
            f"{prefix}_active_connections",
            "Number of active connections",
            ["service"],  # milvus, redis, kafka, mongodb
            registry=self.registry
        )
        
        self.documents_indexed = Gauge(
            f"{prefix}_documents_indexed_total",
            "Total documents indexed in Milvus",
            registry=self.registry
        )
        
        # 错误计数
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors",
            ["component", "error_type"],
            registry=self.registry
        )
    
    def start_server(self, port: Optional[int] = None) -> None:
        """启动Prometheus HTTP服务器"""
        if not PROMETHEUS_AVAILABLE:
            print("Warning: prometheus_client not installed")
            return
        
        port = port or self.config.port
        start_http_server(port, registry=self.registry)
        print(f"Prometheus metrics server started on port {port}")
    
    def get_metrics(self) -> bytes:
        """获取所有指标（用于HTTP端点）"""
        if not PROMETHEUS_AVAILABLE:
            return b""
        return generate_latest(self.registry)
    
    # ==================== 便捷方法 ====================
    
    def record_request(self, endpoint: str, method: str, status: int) -> None:
        """记录请求"""
        if self.request_total:
            self.request_total.labels(
                endpoint=endpoint, method=method, status=str(status)
            ).inc()
    
    @contextmanager
    def track_request_latency(self, endpoint: str):
        """追踪请求延迟"""
        start = time.time()
        try:
            yield
        finally:
            if self.request_latency:
                self.request_latency.labels(endpoint=endpoint).observe(
                    time.time() - start
                )
    
    @contextmanager
    def track_retrieval(self, method: str = "hybrid"):
        """追踪检索延迟"""
        start = time.time()
        try:
            yield
        finally:
            if self.retrieval_latency:
                self.retrieval_latency.labels(method=method).observe(
                    time.time() - start
                )
    
    def record_retrieval_results(self, count: int) -> None:
        """记录检索结果数量"""
        if self.retrieval_results:
            self.retrieval_results.observe(count)
    
    @contextmanager
    def track_generation(self):
        """追踪生成延迟"""
        start = time.time()
        try:
            yield
        finally:
            if self.generation_latency:
                self.generation_latency.observe(time.time() - start)
    
    def record_generation_tokens(self, count: int) -> None:
        """记录生成token数"""
        if self.generation_tokens:
            self.generation_tokens.observe(count)
    
    def record_cache_hit(self, cache_type: str = "query") -> None:
        """记录缓存命中"""
        if self.cache_hits:
            self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "query") -> None:
        """记录缓存未命中"""
        if self.cache_misses:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    @contextmanager
    def track_milvus_query(self):
        """追踪Milvus查询"""
        start = time.time()
        try:
            yield
        finally:
            if self.milvus_query_latency:
                self.milvus_query_latency.observe(time.time() - start)
    
    def record_error(self, component: str, error_type: str) -> None:
        """记录错误"""
        if self.errors_total:
            self.errors_total.labels(
                component=component, error_type=error_type
            ).inc()
    
    def set_active_connections(self, service: str, count: int) -> None:
        """设置活跃连接数"""
        if self.active_connections:
            self.active_connections.labels(service=service).set(count)
    
    def set_documents_indexed(self, count: int) -> None:
        """设置已索引文档数"""
        if self.documents_indexed:
            self.documents_indexed.set(count)


# 全局指标收集器实例
metrics_collector = MetricsCollector()


# ==================== 装饰器 ====================

def track_time(metric_name: str = "request"):
    """追踪函数执行时间的装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                # 可以在这里记录到metrics
        return wrapper
    return decorator


def count_calls(counter_name: str):
    """计数函数调用的装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 增加计数
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== 健康检查 ====================

class HealthChecker:
    """系统健康检查"""
    
    def __init__(self) -> None:
        self.checks: Dict[str, Callable[[], bool]] = {}
    
    def register(self, name: str, check_func: Callable[[], bool]) -> None:
        """注册健康检查"""
        self.checks[name] = check_func
    
    def check_all(self) -> Dict[str, Any]:
        """执行所有健康检查"""
        results = {"status": "healthy", "checks": {}}
        
        for name, check_func in self.checks.items():
            try:
                is_healthy = check_func()
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy"
                }
                if not is_healthy:
                    results["status"] = "unhealthy"
            except Exception as e:
                results["checks"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                results["status"] = "unhealthy"
        
        return results
    
    def check_redis(self) -> bool:
        """检查Redis"""
        try:
            import redis
            from config.config import REDIS_HOST, REDIS_PORT
            client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
            return client.ping()
        except:
            return False
    
    def check_milvus(self) -> bool:
        """检查Milvus"""
        try:
            from pymilvus import connections, utility
            from config.config import MILVUS_HOST, MILVUS_PORT
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            version = utility.get_server_version()
            connections.disconnect("default")
            return version is not None
        except:
            return False
    
    def check_kafka(self) -> bool:
        """检查Kafka"""
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers="localhost:9092",
                api_version_auto_timeout_ms=3000
            )
            producer.close()
            return True
        except:
            return False
    
    def check_mongodb(self) -> bool:
        """检查MongoDB"""
        try:
            from pymongo import MongoClient
            from config.config import MONGODB_URI
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
            client.server_info()
            client.close()
            return True
        except:
            return False


# 全局健康检查器
health_checker = HealthChecker()
