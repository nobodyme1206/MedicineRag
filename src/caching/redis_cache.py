#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存层实现
提升查询响应速度，减少重复计算
"""

from __future__ import annotations

import json
import hashlib
import pickle
import time
from typing import Optional, List, Dict, Any, Union

import numpy as np
from numpy.typing import NDArray

from config.config import REDIS_HOST, REDIS_PORT, LOGS_DIR
from src.utils.logger import setup_logger
from src.utils.exceptions import CacheError, handle_errors, retry

logger = setup_logger("redis_cache", LOGS_DIR / "redis_cache.log")

# 类型别名
Vector = NDArray[np.float32]
CacheKey = str
CacheValue = Union[List[Dict], Vector, Dict, str]


class RedisCache:
    """Redis缓存管理器"""
    
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600
    ) -> None:
        """
        初始化Redis缓存
        
        Args:
            host: Redis主机
            port: Redis端口
            db: 数据库编号
            password: 密码
            ttl: 默认缓存过期时间（秒）
        """
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.client = None
        
        self._connect(password)
    
    def _connect(self, password: Optional[str] = None) -> None:
        """建立Redis连接"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=password,
                decode_responses=False
            )
            self.client.ping()
            logger.info(f"✅ Redis连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"⚠️ Redis连接失败: {e}")
            self.client = None
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.client is not None
    
    def _generate_key(self, prefix: str, data: Any) -> CacheKey:
        """
        生成缓存键
        
        Args:
            prefix: 键前缀
            data: 数据（用于生成hash）
            
        Returns:
            缓存键
        """
        if isinstance(data, str):
            hash_str = data
        elif isinstance(data, (list, dict)):
            hash_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, np.ndarray):
            hash_str = data.tobytes().hex()[:32]
        else:
            hash_str = str(data)
        
        hash_value = hashlib.md5(hash_str.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    # ==================== 查询缓存 ====================
    
    @handle_errors(default_return=None, log_level="warning")
    def get_query_cache(self, query: str) -> Optional[List[Dict]]:
        """
        获取查询结果缓存
        
        Args:
            query: 查询文本
            
        Returns:
            缓存的结果，不存在返回None
        """
        if not self.is_connected:
            return None
        
        key = self._generate_key("query", query)
        cached = self.client.get(key)
        
        if cached:
            logger.debug(f"缓存命中: query={query[:50]}...")
            return pickle.loads(cached)
        return None
    
    @handle_errors(default_return=False, log_level="warning")
    def set_query_cache(
        self, 
        query: str, 
        results: List[Dict], 
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置查询结果缓存
        
        Args:
            query: 查询文本
            results: 查询结果
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.is_connected:
            return False
        
        key = self._generate_key("query", query)
        ttl = ttl or self.ttl
        
        self.client.setex(key, ttl, pickle.dumps(results))
        logger.debug(f"缓存已设置: query={query[:50]}..., ttl={ttl}s")
        return True
    
    # ==================== 向量缓存 ====================
    
    @handle_errors(default_return=None, log_level="debug")
    def get_vector_cache(self, text: str) -> Optional[Vector]:
        """
        获取文本向量缓存
        
        Args:
            text: 文本
            
        Returns:
            向量，不存在返回None
        """
        if not self.is_connected:
            return None
        
        key = self._generate_key("vector", text)
        cached = self.client.get(key)
        
        if cached:
            return pickle.loads(cached)
        return None
    
    @handle_errors(default_return=False, log_level="debug")
    def set_vector_cache(
        self, 
        text: str, 
        vector: Vector, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置文本向量缓存
        
        Args:
            text: 文本
            vector: 向量
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.is_connected:
            return False
        
        key = self._generate_key("vector", text)
        ttl = ttl or self.ttl
        
        self.client.setex(key, ttl, pickle.dumps(vector))
        return True
    
    # ==================== 统计和管理 ====================
    
    @handle_errors(default_return={"status": "error"}, log_level="warning")
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        info = self.client.info()
        
        return {
            "status": "connected",
            "total_keys": self.client.dbsize(),
            "used_memory_mb": round(info["used_memory"] / (1024**2), 2),
            "used_memory_human": info["used_memory_human"],
            "hit_rate": self._calculate_hit_rate(info)
        }
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """计算缓存命中率"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return round(hits / total * 100, 2) if total > 0 else 0.0
    
    @handle_errors(default_return=0, log_level="warning")
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            pattern: 键模式（如"query:*"），None清空所有
            
        Returns:
            删除的键数量
        """
        if not self.is_connected:
            return 0
        
        if pattern:
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                logger.info(f"清空缓存: {pattern}, 删除 {count} 个键")
                return count
            return 0
        else:
            self.client.flushdb()
            logger.info("清空所有缓存")
            return -1
    
    def close(self) -> None:
        """关闭连接"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Redis连接已关闭")


class SemanticCache:
    """
    语义缓存：基于向量相似度的智能缓存
    相似问题可以复用已缓存的答案，避免重复检索和生成
    """
    
    def __init__(
        self,
        cache: RedisCache,
        embedder: Optional[Any] = None,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        ttl: int = 7200
    ) -> None:
        """
        初始化语义缓存
        
        Args:
            cache: Redis缓存实例
            embedder: 向量化器
            similarity_threshold: 相似度阈值（0.92表示92%相似即命中）
            max_cache_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self.cache = cache
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = ttl
        self.stats = {"hits": 0, "misses": 0, "stores": 0}
        
        # 缓存索引键
        self._index_key = "semantic_cache:index"
    
    def _get_embedder(self):
        """延迟获取embedder"""
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        return self.embedder
    
    def _compute_similarity(self, vec1: Vector, vec2: Vector) -> float:
        """计算余弦相似度"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        语义查找：查找相似问题的缓存答案
        
        Args:
            query: 用户查询
            
        Returns:
            缓存的答案和元数据，未命中返回None
        """
        if not self.cache.is_connected:
            return None
        
        try:
            # 计算查询向量
            query_vec = self._get_embedder().encode_single(query)
            
            # 获取缓存索引
            index_data = self.cache.client.get(self._index_key)
            if not index_data:
                self.stats["misses"] += 1
                return None
            
            cache_index = pickle.loads(index_data)
            
            # 遍历查找最相似的缓存
            best_match = None
            best_similarity = 0.0
            
            for cache_key, cached_vec in cache_index.items():
                similarity = self._compute_similarity(query_vec, cached_vec)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = cache_key
            
            if best_match:
                # 获取缓存的答案
                cached_data = self.cache.client.get(f"semantic:{best_match}")
                if cached_data:
                    self.stats["hits"] += 1
                    result = pickle.loads(cached_data)
                    result["cache_hit"] = True
                    result["similarity"] = best_similarity
                    logger.info(f"语义缓存命中: similarity={best_similarity:.3f}, query={query[:50]}...")
                    return result
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.warning(f"语义缓存查找失败: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, query: str, answer: str, contexts: List[Dict], metadata: Dict = None) -> bool:
        """
        存储答案到语义缓存
        
        Args:
            query: 用户查询
            answer: 生成的答案
            contexts: 检索的上下文
            metadata: 额外元数据
            
        Returns:
            是否成功
        """
        if not self.cache.is_connected:
            return False
        
        try:
            # 计算查询向量
            query_vec = self._get_embedder().encode_single(query)
            
            # 生成缓存键
            cache_key = hashlib.md5(query.encode()).hexdigest()
            
            # 存储答案数据
            cache_data = {
                "query": query,
                "answer": answer,
                "contexts": contexts,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            self.cache.client.setex(
                f"semantic:{cache_key}",
                self.ttl,
                pickle.dumps(cache_data)
            )
            
            # 更新索引
            index_data = self.cache.client.get(self._index_key)
            if index_data:
                cache_index = pickle.loads(index_data)
            else:
                cache_index = {}
            
            # 限制缓存大小
            if len(cache_index) >= self.max_cache_size:
                # 删除最旧的条目
                oldest_key = next(iter(cache_index))
                del cache_index[oldest_key]
                self.cache.client.delete(f"semantic:{oldest_key}")
            
            cache_index[cache_key] = query_vec
            self.cache.client.setex(
                self._index_key,
                self.ttl * 2,  # 索引过期时间更长
                pickle.dumps(cache_index)
            )
            
            self.stats["stores"] += 1
            logger.debug(f"语义缓存已存储: query={query[:50]}...")
            return True
            
        except Exception as e:
            logger.warning(f"语义缓存存储失败: {e}")
            return False
    
    def clear(self) -> int:
        """清空语义缓存"""
        if not self.cache.is_connected:
            return 0
        
        try:
            # 获取所有语义缓存键
            keys = self.cache.client.keys("semantic:*")
            count = 0
            if keys:
                count = self.cache.client.delete(*keys)
            
            # 清空索引
            self.cache.client.delete(self._index_key)
            
            logger.info(f"语义缓存已清空: {count} 条")
            return count
        except Exception as e:
            logger.warning(f"清空语义缓存失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total * 100 if total > 0 else 0
        
        # 获取当前缓存大小
        cache_size = 0
        if self.cache.is_connected:
            try:
                index_data = self.cache.client.get(self._index_key)
                if index_data:
                    cache_index = pickle.loads(index_data)
                    cache_size = len(cache_index)
            except:
                pass
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_%": round(hit_rate, 2),
            "cache_size": cache_size,
            "threshold": self.similarity_threshold
        }


class VectorCacheManager:
    """向量缓存管理器"""
    
    def __init__(
        self, 
        cache: RedisCache, 
        embedder: Optional[Any] = None
    ) -> None:
        """
        初始化向量缓存管理器
        
        Args:
            cache: Redis缓存实例
            embedder: 向量化器实例（可选，延迟初始化）
        """
        self.cache = cache
        self.embedder = embedder
        self.stats = {"hits": 0, "misses": 0, "prewarmed": 0}
    
    def _get_embedder(self):
        """延迟获取embedder"""
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        return self.embedder
    
    def get_or_compute_vector(self, text: str) -> Vector:
        """
        获取向量，优先从缓存，否则计算并缓存
        
        Args:
            text: 文本
            
        Returns:
            向量
        """
        # 尝试缓存
        cached = self.cache.get_vector_cache(text)
        if cached is not None:
            self.stats["hits"] += 1
            return cached
        
        self.stats["misses"] += 1
        
        # 计算向量
        vector = self._get_embedder().encode_single(text)
        
        # 缓存（24小时）
        self.cache.set_vector_cache(text, vector, ttl=86400)
        
        return vector
    
    def batch_get_or_compute(self, texts: List[str]) -> Vector:
        """
        批量获取向量，最大化缓存利用
        
        Args:
            texts: 文本列表
            
        Returns:
            向量矩阵
        """
        vectors: List[tuple] = []
        texts_to_compute: List[str] = []
        indices_to_compute: List[int] = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            cached = self.cache.get_vector_cache(text)
            if cached is not None:
                vectors.append((i, cached))
                self.stats["hits"] += 1
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
                self.stats["misses"] += 1
        
        # 批量计算未缓存的
        if texts_to_compute:
            computed = self._get_embedder().encode_batch(texts_to_compute)
            
            for idx, (orig_idx, text) in enumerate(zip(indices_to_compute, texts_to_compute)):
                vec = computed[idx]
                self.cache.set_vector_cache(text, vec, ttl=86400)
                vectors.append((orig_idx, vec))
        
        # 按原始顺序排序
        vectors.sort(key=lambda x: x[0])
        return np.array([v[1] for v in vectors])
    
    def prewarm_cache(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> int:
        """
        预热缓存
        
        Args:
            texts: 要预热的文本列表
            batch_size: 批次大小
            
        Returns:
            预热的文本数量
        """
        logger.info(f"🔥 预热缓存: {len(texts)} 条文本")
        
        embedder = self._get_embedder()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vectors = embedder.encode_batch(batch)
            
            for text, vec in zip(batch, vectors):
                self.cache.set_vector_cache(text, vec, ttl=86400 * 7)
                self.stats["prewarmed"] += 1
            
            logger.info(f"   预热进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        logger.info(f"✅ 预热完成: {self.stats['prewarmed']} 条")
        return self.stats["prewarmed"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total * 100 if total > 0 else 0
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_%": round(hit_rate, 2)
        }


def main() -> None:
    """测试Redis缓存"""
    logger.info("=" * 50)
    logger.info("测试Redis缓存")
    logger.info("=" * 50)
    
    cache = RedisCache()
    
    if not cache.is_connected:
        logger.error("Redis未连接，请启动Redis服务")
        return
    
    # 测试查询缓存
    test_query = "diabetes symptoms"
    test_results = [{"id": "1", "text": "test", "score": 0.9}]
    
    cache.set_query_cache(test_query, test_results)
    cached = cache.get_query_cache(test_query)
    
    assert cached == test_results, "查询缓存测试失败"
    logger.info("✅ 查询缓存测试通过")
    
    # 测试向量缓存
    test_vector = np.random.rand(512).astype(np.float32)
    cache.set_vector_cache("test text", test_vector)
    cached_vec = cache.get_vector_cache("test text")
    
    assert np.allclose(cached_vec, test_vector), "向量缓存测试失败"
    logger.info("✅ 向量缓存测试通过")
    
    # 打印统计
    stats = cache.get_stats()
    logger.info(f"缓存统计: {stats}")
    
    cache.close()


if __name__ == "__main__":
    main()
