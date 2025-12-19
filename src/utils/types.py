#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类型定义模块
统一项目中使用的类型注解
"""

from typing import TypedDict, List, Dict, Optional, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


# ==================== 基础类型别名 ====================

Vector = NDArray[np.float32]  # 512维向量
VectorBatch = NDArray[np.float32]  # N x 512 向量矩阵
Score = float  # 相似度分数 0-1
DocumentId = str  # 文档ID


# ==================== 数据结构 ====================

class DocumentChunk(TypedDict):
    """文档块"""
    id: str
    pmid: str
    title: str
    chunk_text: str
    topic: str
    embedding: Optional[List[float]]


class SearchResult(TypedDict):
    """检索结果"""
    id: str
    pmid: str
    text: str
    topic: str
    score: float
    distance: Optional[float]


class HybridSearchResult(SearchResult):
    """混合检索结果"""
    hybrid_score: float
    bm25_score: float
    vector_score: float


class RAGResponse(TypedDict):
    """RAG响应"""
    query: str
    answer: str
    contexts: List[SearchResult]
    retrieval_time: float
    generation_time: float
    total_time: float
    num_contexts: int


class EvaluationMetrics(TypedDict):
    """评估指标"""
    precision: float
    recall: float
    f1: float
    mrr: float
    ndcg: float
    latency_ms: float


# ==================== 配置类型 ====================

@dataclass
class MilvusConfig:
    """Milvus配置"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "medical_knowledge_base"
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024
    nprobe: int = 16


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl: int = 3600


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9


@dataclass 
class EmbeddingConfig:
    """Embedding配置"""
    model_name: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512
    batch_size: int = 128
    device: str = "cuda"


# ==================== 回调类型 ====================

ProgressCallback = Callable[[int, int, str], None]  # (current, total, message)
ErrorCallback = Callable[[Exception, str], None]  # (error, context)
