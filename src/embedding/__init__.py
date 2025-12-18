# -*- coding: utf-8 -*-
"""
向量化模块
统一高性能向量化：大批次 + 多GPU + 多进程 + PySpark + 断点续传
"""

from .unified_embedder import (
    UnifiedEmbedder,
    EmbeddingConfig,
    EmbeddingResult,
    GPUEmbedder,
    MultiProcessEngine,
    SparkEngine,
    CheckpointManager
)

__all__ = [
    'UnifiedEmbedder',
    'EmbeddingConfig', 
    'EmbeddingResult',
    'GPUEmbedder',
    'MultiProcessEngine',
    'SparkEngine',
    'CheckpointManager'
]
