# -*- coding: utf-8 -*-
"""
日志工具 - 统一日志管理

日志文件结构：
- app.log: 所有应用日志（INFO及以上）
- error.log: 错误日志（ERROR及以上）
- access.log: API访问日志
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# 全局日志配置
_LOG_DIR: Optional[Path] = None
_INITIALIZED = False
_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 5

# 日志格式
_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 全局handlers（避免重复创建）
_app_handler: Optional[RotatingFileHandler] = None
_error_handler: Optional[RotatingFileHandler] = None
_access_handler: Optional[RotatingFileHandler] = None


def _init_global_handlers(logs_dir: Path) -> None:
    """初始化全局日志handlers"""
    global _LOG_DIR, _INITIALIZED, _app_handler, _error_handler, _access_handler
    
    if _INITIALIZED and _LOG_DIR == logs_dir:
        return
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = logs_dir
    
    # app.log - 所有INFO及以上日志
    _app_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding='utf-8'
    )
    _app_handler.setLevel(logging.INFO)
    _app_handler.setFormatter(_FORMATTER)
    
    # error.log - 仅ERROR及以上
    _error_handler = RotatingFileHandler(
        logs_dir / "error.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding='utf-8'
    )
    _error_handler.setLevel(logging.ERROR)
    _error_handler.setFormatter(_FORMATTER)
    
    # access.log - API访问日志
    _access_handler = RotatingFileHandler(
        logs_dir / "access.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding='utf-8'
    )
    _access_handler.setLevel(logging.INFO)
    _access_handler.setFormatter(_FORMATTER)
    
    _INITIALIZED = True


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,  # 保留参数兼容性，但不再使用
    level: int = logging.INFO,
    max_bytes: int = None,  # 已废弃
    backup_count: int = None,  # 已废弃
    rotation_type: str = None  # 已废弃
) -> logging.Logger:
    """
    设置日志记录器（统一输出到三个日志文件）
    
    Args:
        name: logger名称
        log_file: 已废弃，保留兼容性
        level: 日志级别
        
    Returns:
        配置好的logger实例
        
    Note:
        所有日志统一输出到：
        - logs/app.log (INFO+)
        - logs/error.log (ERROR+)
        - logs/access.log (仅API访问日志)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 初始化全局handlers
    if log_file:
        logs_dir = Path(log_file).parent
    else:
        # 默认日志目录
        logs_dir = Path(__file__).parent.parent.parent / "logs"
    
    _init_global_handlers(logs_dir)
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)
    
    # 添加全局文件handlers
    if _app_handler:
        logger.addHandler(_app_handler)
    if _error_handler:
        logger.addHandler(_error_handler)
    
    # API相关的logger额外输出到access.log
    if name in ("api", "access", "fastapi"):
        if _access_handler:
            logger.addHandler(_access_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取logger"""
    return logging.getLogger(name)


def get_access_logger() -> logging.Logger:
    """获取API访问日志logger"""
    logger = logging.getLogger("access")
    if not logger.handlers:
        setup_logger("access")
    return logger


def cleanup_old_logs(logs_dir: Path, max_age_days: int = 30) -> int:
    """清理超过指定天数的旧日志"""
    import time
    
    if not logs_dir.exists():
        return 0
    
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    deleted = 0
    
    for log_file in logs_dir.glob("*.log*"):
        if log_file.stat().st_mtime < now - max_age_seconds:
            log_file.unlink()
            deleted += 1
    
    return deleted
