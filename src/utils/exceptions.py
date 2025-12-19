#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一异常处理模块
定义项目专用异常类和错误处理装饰器
"""

import functools
import traceback
from typing import Optional, Type, Callable, Any
from src.utils.logger import setup_logger

logger = setup_logger("exceptions")


# ==================== 自定义异常类 ====================

class RAGException(Exception):
    """RAG系统基础异常"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
    
    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ConfigurationError(RAGException):
    """配置错误"""
    pass


class ConnectionError(RAGException):
    """连接错误（数据库、缓存等）"""
    pass


class DataProcessingError(RAGException):
    """数据处理错误"""
    pass


class EmbeddingError(RAGException):
    """向量化错误"""
    pass


class RetrievalError(RAGException):
    """检索错误"""
    pass


class GenerationError(RAGException):
    """LLM生成错误"""
    pass


class CacheError(RAGException):
    """缓存错误"""
    pass


class StorageError(RAGException):
    """存储错误"""
    pass


# ==================== 错误处理装饰器 ====================

def handle_errors(
    default_return: Any = None,
    reraise: bool = False,
    log_level: str = "error",
    exception_type: Type[Exception] = Exception
):
    """
    统一错误处理装饰器
    
    Args:
        default_return: 发生错误时的默认返回值
        reraise: 是否重新抛出异常
        log_level: 日志级别 (error/warning/info)
        exception_type: 要捕获的异常类型
    
    Usage:
        @handle_errors(default_return=[], log_level="warning")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                # 获取函数信息
                func_name = func.__qualname__
                module = func.__module__
                
                # 记录日志
                log_func = getattr(logger, log_level, logger.error)
                log_func(f"[{module}.{func_name}] {type(e).__name__}: {e}")
                
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍增因子
        exceptions: 要重试的异常类型
    
    Usage:
        @retry(max_attempts=3, delay=1.0)
        def unstable_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"[{func.__qualname__}] 第{attempt+1}次尝试失败: {e}, "
                            f"{current_delay:.1f}秒后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"[{func.__qualname__}] 重试{max_attempts}次后仍失败: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


def log_execution(log_args: bool = False, log_result: bool = False):
    """
    执行日志装饰器
    
    Args:
        log_args: 是否记录参数
        log_result: 是否记录返回值
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            
            # 记录开始
            if log_args:
                logger.debug(f"[{func_name}] 开始执行, args={args}, kwargs={kwargs}")
            else:
                logger.debug(f"[{func_name}] 开始执行")
            
            import time
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if log_result:
                    logger.debug(f"[{func_name}] 执行完成, 耗时={elapsed:.3f}s, result={result}")
                else:
                    logger.debug(f"[{func_name}] 执行完成, 耗时={elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[{func_name}] 执行失败, 耗时={elapsed:.3f}s, error={e}")
                raise
        return wrapper
    return decorator


# ==================== 上下文管理器 ====================

class ErrorContext:
    """
    错误上下文管理器
    
    Usage:
        with ErrorContext("处理数据", reraise=False) as ctx:
            # 可能出错的代码
            ...
        if ctx.error:
            print(f"发生错误: {ctx.error}")
    """
    def __init__(self, operation: str, reraise: bool = True):
        self.operation = operation
        self.reraise = reraise
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = exc_val
            logger.error(f"[{self.operation}] 失败: {exc_val}")
            
            if not self.reraise:
                return True  # 抑制异常
        return False
