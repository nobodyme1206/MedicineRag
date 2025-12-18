# -*- coding: utf-8 -*-
"""
核心功能单元测试
覆盖: 爬虫、数据处理、向量化、检索、缓存
"""

import sys
import json
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPubMedCrawler:
    """爬虫测试"""
    
    def test_exponential_backoff(self):
        """测试指数退避计算"""
        from src.data_processing.pubmed_crawler import PubMedCrawler
        crawler = PubMedCrawler()
        
        # 第0次尝试: 2^0 = 1秒左右
        delay0 = crawler._exponential_backoff(0)
        assert 1 <= delay0 <= 2
        
        # 第3次尝试: 2^3 = 8秒左右
        delay3 = crawler._exponential_backoff(3)
        assert 8 <= delay3 <= 10
        
        # 最大不超过60秒
        delay10 = crawler._exponential_backoff(10)
        assert delay10 <= 60
    
    def test_checkpoint_path(self):
        """测试checkpoint路径生成"""
        from src.data_processing.pubmed_crawler import PubMedCrawler
        crawler = PubMedCrawler()
        
        path = crawler._get_checkpoint_path("diabetes mellitus")
        assert "diabetes_mellitus" in str(path)
        assert path.suffix == ".json"
    
    def test_parse_article_filters_short_abstract(self):
        """测试文章解析过滤短摘要"""
        from src.data_processing.pubmed_crawler import PubMedCrawler
        crawler = PubMedCrawler()
        
        # 模拟短摘要文章
        mock_record = {
            'MedlineCitation': {
                'PMID': '12345',
                'Article': {
                    'ArticleTitle': 'Test Title',
                    'Abstract': {'AbstractText': ['Short']}
                }
            }
        }
        
        result = crawler._parse_article(mock_record, "test")
        assert result is None  # 应该被过滤


class TestTextEmbedder:
    """向量化测试"""
    
    @pytest.fixture
    def embedder(self):
        """创建embedder实例"""
        from src.embedding.embedder import TextEmbedder
        return TextEmbedder()
    
    def test_encode_single(self, embedder):
        """测试单文本编码"""
        text = "What are the symptoms of diabetes?"
        embedding = embedder.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 512  # bge-small-zh维度
        assert np.linalg.norm(embedding) > 0  # 非零向量
    
    def test_encode_batch(self, embedder):
        """测试批量编码"""
        texts = [
            "Diabetes symptoms include increased thirst",
            "Heart disease prevention methods",
            "Cancer treatment options"
        ]
        embeddings = embedder.encode_batch(texts)
        
        assert embeddings.shape == (3, 512)
        # 检查归一化
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)
    
    def test_similar_texts_have_high_similarity(self, embedder):
        """测试相似文本有高相似度"""
        text1 = "diabetes symptoms and treatment"
        text2 = "diabetic patient symptoms therapy"
        text3 = "weather forecast for tomorrow"
        
        emb1 = embedder.encode_single(text1)
        emb2 = embedder.encode_single(text2)
        emb3 = embedder.encode_single(text3)
        
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        
        assert sim_12 > sim_13  # 相似文本相似度更高


class TestRedisCache:
    """Redis缓存测试"""
    
    @pytest.fixture
    def cache(self):
        """创建缓存实例"""
        from src.caching.redis_cache import RedisCache
        cache = RedisCache(db=15)  # 使用测试数据库
        yield cache
        if cache.client:
            cache.clear_cache()
            cache.close()
    
    def test_query_cache_set_get(self, cache):
        """测试查询缓存设置和获取"""
        if not cache.client:
            pytest.skip("Redis不可用")
        
        query = "test query"
        results = [{"id": 1, "text": "result"}]
        
        cache.set_query_cache(query, results, ttl=60)
        cached = cache.get_query_cache(query)
        
        assert cached == results
    
    def test_vector_cache(self, cache):
        """测试向量缓存"""
        if not cache.client:
            pytest.skip("Redis不可用")
        
        text = "test text for vector"
        vector = np.random.rand(512).astype(np.float32)
        
        cache.set_vector_cache(text, vector, ttl=60)
        cached = cache.get_vector_cache(text)
        
        assert cached is not None
        assert np.allclose(cached, vector)
    
    def test_cache_miss_returns_none(self, cache):
        """测试缓存未命中返回None"""
        if not cache.client:
            pytest.skip("Redis不可用")
        
        result = cache.get_query_cache("nonexistent_query_12345")
        assert result is None


class TestVectorCacheManager:
    """向量缓存管理器测试"""
    
    def test_get_or_compute_vector(self):
        """测试获取或计算向量"""
        from src.caching.redis_cache import RedisCache, VectorCacheManager
        
        cache = RedisCache(db=15)
        if not cache.client:
            pytest.skip("Redis不可用")
        
        manager = VectorCacheManager(cache)
        
        text = "test medical query"
        
        # 第一次调用应该计算
        vector1 = manager.get_or_compute_vector(text)
        assert manager.stats["misses"] == 1
        
        # 第二次调用应该命中缓存
        vector2 = manager.get_or_compute_vector(text)
        assert manager.stats["hits"] == 1
        
        assert np.allclose(vector1, vector2)
        
        cache.clear_cache()
        cache.close()


class TestHybridSearcher:
    """混合检索测试"""
    
    def test_tokenize(self):
        """测试分词"""
        from src.retrieval.hybrid_searcher import HybridSearcher
        
        # Mock初始化避免加载数据
        with patch.object(HybridSearcher, '__init__', lambda x, y=None: None):
            searcher = HybridSearcher()
            searcher._tokenize = HybridSearcher._tokenize.__get__(searcher)
            
            tokens = searcher._tokenize("Hello World! This is a test.")
            assert "hello" in tokens
            assert "world" in tokens
            assert "test" in tokens
    
    def test_hybrid_score_calculation(self):
        """测试混合分数计算"""
        from src.retrieval.hybrid_searcher import HybridSearcher
        
        with patch.object(HybridSearcher, '__init__', lambda x, y=None: None):
            searcher = HybridSearcher()
            searcher.chunks = []
            searcher._tokenize = lambda text: text.lower().split()
            
            # 模拟向量检索结果
            vector_results = [
                {"id": 1, "text": "diabetes symptoms treatment", "score": 0.9},
                {"id": 2, "text": "heart disease prevention", "score": 0.7},
            ]
            
            results = searcher.hybrid_search(
                "diabetes symptoms",
                vector_results,
                alpha=0.6,
                top_k=2
            )
            
            assert len(results) == 2
            assert all("hybrid_score" in r for r in results)


class TestDataProcessor:
    """数据处理测试"""
    
    def test_chunk_text(self):
        """测试文本切分"""
        from src.data_processing.data_processor import DataProcessor
        
        with patch.object(DataProcessor, '__init__', lambda x, y=False: None):
            processor = DataProcessor()
            processor.chunk_text = DataProcessor.chunk_text.__get__(processor)
            
            # 测试长文本切分
            long_text = "A" * 1000
            chunks = processor.chunk_text(long_text, chunk_size=512, overlap=50)
            
            assert len(chunks) >= 2
            assert all(len(c) <= 512 for c in chunks)
    
    def test_chunk_text_filters_short(self):
        """测试短文本被过滤"""
        from src.data_processing.data_processor import DataProcessor
        
        with patch.object(DataProcessor, '__init__', lambda x, y=False: None):
            processor = DataProcessor()
            processor.chunk_text = DataProcessor.chunk_text.__get__(processor)
            
            short_text = "Too short"
            chunks = processor.chunk_text(short_text, chunk_size=512, overlap=50)
            
            assert len(chunks) == 0


class TestSparkEmbedder:
    """Spark向量化测试"""
    
    def test_init_local_mode(self):
        """测试本地模式初始化"""
        from src.embedding.spark_embedder import SparkEmbedder
        
        embedder = SparkEmbedder(use_cluster=False)
        assert embedder.spark is not None
        assert "local" in embedder.spark.sparkContext.master
        embedder.stop()


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.slow
    def test_full_retrieval_pipeline(self):
        """测试完整检索流程"""
        from src.rag.rag_system import RAGSystem
        
        rag = RAGSystem(use_cache=False)
        
        query = "What are the symptoms of diabetes?"
        results = rag.retrieve(query, top_k=5)
        
        assert len(results) <= 5
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
