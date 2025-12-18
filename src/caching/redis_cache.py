#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存层实现
提升查询响应速度，减少重复计算
"""

import redis
import json
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any
import pickle
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("redis_cache", LOGS_DIR / "redis_cache.log")


class RedisCache:
    """Redis缓存管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600  # 默认1小时过期
    ):
        """
        初始化Redis缓存
        
        Args:
            host: Redis主机
            port: Redis端口
            db: 数据库编号
            password: 密码
            ttl: 缓存过期时间（秒）
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # 支持二进制数据
            )
            self.client.ping()
            self.ttl = ttl
            logger.info(f"✅ Redis连接成功: {host}:{port}")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis连接失败: {e}")
            self.client = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
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
            hash_str = data.tobytes().hex()[:32]  # 使用前32字符
        else:
            hash_str = str(data)
        
        hash_value = hashlib.md5(hash_str.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    def get_query_cache(self, query: str) -> Optional[List[Dict]]:
        """
        获取查询结果缓存
        
        Args:
            query: 查询文本
            
        Returns:
            缓存的结果，如果不存在返回None
        """
        if not self.client:
            return None
        
        key = self._generate_key("query", query)
        
        try:
            cached = self.client.get(key)
            if cached:
                logger.info(f"✅ 缓存命中: query={query[:50]}...")
                return pickle.loads(cached)
            return None
        except Exception as e:
            logger.error(f"❌ 读取缓存失败: {e}")
            return None
    
    def set_query_cache(self, query: str, results: List[Dict], ttl: Optional[int] = None) -> bool:
        """
        设置查询结果缓存
        
        Args:
            query: 查询文本
            results: 查询结果
            ttl: 过期时间（秒），如果为None使用默认值
            
        Returns:
            是否成功
        """
        if not self.client:
            return False
        
        key = self._generate_key("query", query)
        ttl = ttl or self.ttl
        
        try:
            self.client.setex(
                key,
                ttl,
                pickle.dumps(results)
            )
            logger.info(f"✅ 缓存已设置: query={query[:50]}..., ttl={ttl}s")
            return True
        except Exception as e:
            logger.error(f"❌ 设置缓存失败: {e}")
            return False
    
    def get_vector_cache(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本向量缓存
        
        Args:
            text: 文本
            
        Returns:
            向量，如果不存在返回None
        """
        if not self.client:
            return None
        
        key = self._generate_key("vector", text)
        
        try:
            cached = self.client.get(key)
            if cached:
                logger.debug(f"✅ 向量缓存命中: text={text[:50]}...")
                return pickle.loads(cached)
            return None
        except Exception as e:
            logger.error(f"❌ 读取向量缓存失败: {e}")
            return None
    
    def set_vector_cache(self, text: str, vector: np.ndarray, ttl: Optional[int] = None) -> bool:
        """
        设置文本向量缓存
        
        Args:
            text: 文本
            vector: 向量
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.client:
            return False
        
        key = self._generate_key("vector", text)
        ttl = ttl or self.ttl
        
        try:
            self.client.setex(
                key,
                ttl,
                pickle.dumps(vector)
            )
            logger.debug(f"✅ 向量缓存已设置: text={text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ 设置向量缓存失败: {e}")
            return False
    
    def get_chunks_cache(self, chunk_ids: List[str]) -> Optional[List[Dict]]:
        """
        批量获取chunks缓存
        
        Args:
            chunk_ids: chunk ID列表
            
        Returns:
            chunks列表，如果不存在返回None
        """
        if not self.client:
            return None
        
        try:
            keys = [f"chunk:{chunk_id}" for chunk_id in chunk_ids]
            cached = self.client.mget(keys)
            
            results = []
            for item in cached:
                if item:
                    results.append(pickle.loads(item))
                else:
                    return None  # 如果有任何一个缺失，返回None
            
            if results:
                logger.info(f"✅ 批量缓存命中: {len(results)} chunks")
                return results
            return None
        except Exception as e:
            logger.error(f"❌ 批量读取缓存失败: {e}")
            return None
    
    def set_chunks_cache(self, chunks: List[Dict], ttl: Optional[int] = None) -> bool:
        """
        批量设置chunks缓存
        
        Args:
            chunks: chunks列表（必须包含id字段）
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.client:
            return False
        
        ttl = ttl or self.ttl
        
        try:
            pipe = self.client.pipeline()
            for chunk in chunks:
                key = f"chunk:{chunk['id']}"
                pipe.setex(key, ttl, pickle.dumps(chunk))
            pipe.execute()
            
            logger.info(f"✅ 批量缓存已设置: {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"❌ 批量设置缓存失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self.client:
            return {'status': 'disconnected'}
        
        try:
            info = self.client.info()
            
            # 统计各类型键的数量
            query_keys = len(self.client.keys("query:*"))
            vector_keys = len(self.client.keys("vector:*"))
            chunk_keys = len(self.client.keys("chunk:*"))
            
            stats = {
                'status': 'connected',
                'total_keys': self.client.dbsize(),
                'query_cache_keys': query_keys,
                'vector_cache_keys': vector_keys,
                'chunk_cache_keys': chunk_keys,
                'used_memory_mb': info['used_memory'] / (1024**2),
                'used_memory_human': info['used_memory_human'],
                'total_commands': info['total_commands_processed'],
                'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1) * 100
            }
            
            return stats
        except Exception as e:
            logger.error(f"❌ 获取统计失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            pattern: 键模式（如"query:*"），如果为None清空所有
            
        Returns:
            删除的键数量
        """
        if not self.client:
            return 0
        
        try:
            if pattern:
                keys = self.client.keys(pattern)
                if keys:
                    count = self.client.delete(*keys)
                    logger.info(f"✅ 清空缓存: {pattern}, 删除 {count} 个键")
                    return count
            else:
                self.client.flushdb()
                logger.info("✅ 清空所有缓存")
                return -1
            
            return 0
        except Exception as e:
            logger.error(f"❌ 清空缓存失败: {e}")
            return 0
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("✅ Redis连接已关闭")


class CachedRAGSystem:
    """带缓存的RAG系统封装"""
    
    def __init__(self, rag_system, cache: RedisCache):
        self.rag_system = rag_system
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def query(self, query: str, use_cache: bool = True, **kwargs) -> List[Dict]:
        if use_cache:
            cached_result = self.cache.get_query_cache(query)
            if cached_result is not None:
                self.cache_hits += 1
                logger.info(f"🎯 缓存命中率: {self.get_hit_rate():.1f}%")
                return cached_result
            self.cache_misses += 1
        
        results = self.rag_system.retrieve(query, **kwargs)
        
        if use_cache and results:
            self.cache.set_query_cache(query, results)
        
        return results
    
    def get_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) * 100 if total > 0 else 0.0


class VectorCacheManager:
    """
    方案B: 增强向量缓存管理器
    支持批量向量缓存、预热、LRU淘汰
    """
    
    def __init__(self, cache: RedisCache, embedder=None):
        self.cache = cache
        self.embedder = embedder
        self.stats = {"hits": 0, "misses": 0, "prewarmed": 0}
    
    def get_or_compute_vector(self, text: str) -> np.ndarray:
        """获取向量，优先从缓存，否则计算并缓存"""
        # 尝试缓存
        cached = self.cache.get_vector_cache(text)
        if cached is not None:
            self.stats["hits"] += 1
            return cached
        
        self.stats["misses"] += 1
        
        # 计算向量
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        
        vector = self.embedder.encode_single(text)
        
        # 缓存（24小时）
        self.cache.set_vector_cache(text, vector, ttl=86400)
        
        return vector
    
    def batch_get_or_compute(self, texts: List[str]) -> np.ndarray:
        """批量获取向量，最大化缓存利用"""
        vectors = []
        texts_to_compute = []
        indices_to_compute = []
        
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
            if self.embedder is None:
                from src.embedding.embedder import TextEmbedder
                self.embedder = TextEmbedder()
            
            computed = self.embedder.encode_batch(texts_to_compute)
            
            # 缓存并添加结果
            for idx, (orig_idx, text) in enumerate(zip(indices_to_compute, texts_to_compute)):
                vec = computed[idx]
                self.cache.set_vector_cache(text, vec, ttl=86400)
                vectors.append((orig_idx, vec))
        
        # 按原始顺序排序
        vectors.sort(key=lambda x: x[0])
        return np.array([v[1] for v in vectors])
    
    def prewarm_cache(self, texts: List[str], batch_size: int = 100):
        """预热缓存 - 批量预计算常用查询的向量"""
        logger.info(f"🔥 预热缓存: {len(texts)} 条文本")
        
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vectors = self.embedder.encode_batch(batch)
            
            for text, vec in zip(batch, vectors):
                self.cache.set_vector_cache(text, vec, ttl=86400 * 7)  # 7天
                self.stats["prewarmed"] += 1
            
            logger.info(f"   预热进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        logger.info(f"✅ 预热完成: {self.stats['prewarmed']} 条")
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total * 100 if total > 0 else 0
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_%": round(hit_rate, 2)
        }


def demo_cache_performance():
    """演示缓存性能提升"""
    logger.info("=" * 70)
    logger.info("🚀 Redis缓存性能演示")
    logger.info("=" * 70)
    
    # 初始化缓存
    cache = RedisCache()
    
    if not cache.client:
        print("❌ Redis未运行，请启动Redis服务")
        print("Docker方式: docker compose -f docker/docker-compose.yml up -d redis")
        return
    
    # 测试数据
    test_queries = [
        "什么是糖尿病的症状？",
        "高血压的治疗方法",
        "癌症的预防措施",
        "心血管疾病的风险因素",
        "什么是糖尿病的症状？",  # 重复查询
    ]
    
    # 模拟查询结果
    mock_results = [
        {"id": "1", "content": "糖尿病症状包括...", "score": 0.95},
        {"id": "2", "content": "多饮、多尿...", "score": 0.90}
    ]
    
    print("\n📝 测试查询缓存")
    print("-" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        
        # 检查缓存
        start = time.time()
        cached = cache.get_query_cache(query)
        
        if cached:
            elapsed = time.time() - start
            print(f"  ✅ 缓存命中 ({elapsed*1000:.2f}ms)")
        else:
            # 模拟实际查询（较慢）
            time.sleep(0.5)  # 模拟查询延迟
            cache.set_query_cache(query, mock_results)
            elapsed = time.time() - start
            print(f"  ❌ 缓存未命中，已缓存 ({elapsed*1000:.2f}ms)")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("📊 缓存统计")
    print("=" * 70)
    
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    cache.close()


if __name__ == "__main__":
    import time
    demo_cache_performance()
