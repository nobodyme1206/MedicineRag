# -*- coding: utf-8 -*-
"""
日志工具 - 支持日志轮转
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name: str,
    log_file: str = None,
    level=logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    rotation_type: str = "size"  # "size" or "time"
):
    """
    设置日志记录器（支持轮转）
    
    Args:
        name: logger名称
        log_file: 日志文件路径
        level: 日志级别
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        rotation_type: 轮转类型 - "size"按大小, "time"按时间
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（带轮转）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation_type == "time":
            # 按时间轮转（每天一个文件，保留7天）
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.suffix = "%Y-%m-%d"
        else:
            # 按大小轮转（默认）
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    """获取logger"""
    return logging.getLogger(name)


def cleanup_old_logs(logs_dir: Path, max_age_days: int = 30):
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
