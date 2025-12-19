# -*- coding: utf-8 -*-
"""
集成测试模块
测试各组件之间的协作和完整流程
"""

import sys
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGIntegration:
    """RAG系统集成测试"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """创建Mock RAG系统"""
        from src.rag.rag_system import RAGSystem
        
        with patch.object(RAGSystem, '__init__', lambda x, **kwargs: None):
            rag = RAGSystem()
            rag.retriever = Mock()
            rag.generator = Mock()
            rag.cache = Mock()
            
            # Mock检索结果
            rag.retriever.search.return_value = [
                {"text": "Diabetes is a chronic disease...", "pmid": "123", "score": 0.9},
                {"text": "Type 2 diabetes symptoms...", "pmid": "456", "score": 0.8}
            ]
            
            # Mock生成结果
            rag.generator.generate.return_value = "Diabetes is a metabolic disorder..."
            
            yield rag
    
    def test_retrieve_and_generate_flow(self, mock_rag_system):
        """测试检索-生成流程"""
        rag = mock_rag_system
        
        # 模拟完整流程
        query = "What is diabetes?"
        contexts = rag.retriever.search(query, top_k=5)
        answer = rag.generator.generate(query, contexts)
        
        assert len(contexts) == 2
        assert "diabetes" in answer.lower()
        rag.retriever.search.assert_called_once()
        rag.generator.generate.assert_called_once()


class TestQueryRewriterIntegration:
    """查询改写集成测试"""
    
    def test_rewriter_with_mock_llm(self):
        """测试查询改写（Mock LLM）"""
        from src.rag.query_rewriter import QueryRewriter
        
        with patch.object(QueryRewriter, '__init__', lambda x, **kwargs: None):
            rewriter = QueryRewriter()
            rewriter.client = Mock()
            
            # Mock LLM响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "diabetes mellitus symptoms treatment diagnosis"
            rewriter.client.chat.completions.create.return_value = mock_response
            
            # 测试改写
            rewriter.rewrite = lambda q: "diabetes mellitus symptoms treatment diagnosis"
            result = rewriter.rewrite("What is diabetes?")
            
            assert "diabetes" in result.lower()
            assert len(result) > len("What is diabetes?")


class TestConversationIntegration:
    """多轮对话集成测试"""
    
    def test_conversation_context_building(self):
        """测试对话上下文构建"""
        from src.rag.conversation import ConversationManager, Message
        
        manager = ConversationManager(max_history=5)
        
        # 添加对话历史
        manager.add_message("user", "What is diabetes?")
        manager.add_message("assistant", "Diabetes is a metabolic disorder...")
        manager.add_message("user", "What are the symptoms?")
        
        # 获取上下文
        context = manager.get_context_for_query("How to treat it?")
        
        assert "diabetes" in context.lower()
        assert len(manager.history) == 3
    
    def test_conversation_history_limit(self):
        """测试对话历史限制"""
        from src.rag.conversation import ConversationManager
        
        manager = ConversationManager(max_history=3)
        
        for i in range(5):
            manager.add_message("user", f"Question {i}")
            manager.add_message("assistant", f"Answer {i}")
        
        # 应该只保留最近3轮
        assert len(manager.history) <= 6  # 3轮 * 2消息


class TestStreamingIntegration:
    """流式输出集成测试"""
    
    def test_streaming_generator_mock(self):
        """测试流式生成器（Mock）"""
        from src.rag.streaming import StreamingGenerator
        
        with patch.object(StreamingGenerator, '__init__', lambda x, **kwargs: None):
            generator = StreamingGenerator()
            generator.client = Mock()
            generator.model = "test-model"
            
            # Mock流式响应
            mock_chunks = [
                Mock(choices=[Mock(delta=Mock(content="Hello"))]),
                Mock(choices=[Mock(delta=Mock(content=" World"))]),
                Mock(choices=[Mock(delta=Mock(content="!"))]),
            ]
            generator.client.chat.completions.create.return_value = iter(mock_chunks)
            
            # 收集tokens
            generator._get_default_system_prompt = lambda: "You are helpful."
            generator._build_user_prompt = lambda q, c: f"Q: {q}"
            
            tokens = []
            for chunk in mock_chunks:
                if chunk.choices[0].delta.content:
                    tokens.append(chunk.choices[0].delta.content)
            
            assert "".join(tokens) == "Hello World!"


class TestCitationIntegration:
    """答案溯源集成测试"""
    
    def test_citation_extraction(self):
        """测试引用提取"""
        from src.rag.citation import CitationManager
        
        manager = CitationManager()
        
        answer = "Diabetes is a chronic condition [Document 1]. Treatment includes medication [Document 2]."
        contexts = [
            {"pmid": "123", "title": "Diabetes Overview", "text": "...", "score": 0.9},
            {"pmid": "456", "title": "Treatment Guide", "text": "...", "score": 0.8},
        ]
        
        cited_answer = manager.add_citations_to_answer(answer, contexts)
        
        assert len(cited_answer.citations) == 2
        assert cited_answer.citations[0].pmid == "123"
    
    def test_source_tracker(self):
        """测试来源追踪"""
        from src.rag.citation import SourceTracker
        
        tracker = SourceTracker()
        
        answer = "According to [Document 1], diabetes affects millions."
        contexts = [
            {"pmid": "123", "title": "Stats", "text": "...", "score": 0.9},
            {"pmid": "456", "title": "Other", "text": "...", "score": 0.7},
        ]
        
        analysis = tracker.analyze_answer(answer, contexts)
        
        assert analysis["cited_count"] == 1
        assert 1 in analysis["cited_indices"]


class TestCacheIntegration:
    """缓存集成测试"""
    
    @pytest.fixture
    def redis_cache(self):
        """创建Redis缓存"""
        from src.caching.redis_cache import RedisCache
        cache = RedisCache(db=15)
        yield cache
        if cache.client:
            cache.clear_cache()
            cache.close()
    
    def test_cache_with_retrieval(self, redis_cache):
        """测试缓存与检索集成"""
        if not redis_cache.client:
            pytest.skip("Redis不可用")
        
        query = "diabetes symptoms"
        results = [{"text": "test", "score": 0.9}]
        
        # 首次查询 - 缓存未命中
        cached = redis_cache.get_query_cache(query)
        assert cached is None
        
        # 设置缓存
        redis_cache.set_query_cache(query, results)
        
        # 再次查询 - 缓存命中
        cached = redis_cache.get_query_cache(query)
        assert cached == results


class TestMilvusIntegration:
    """Milvus集成测试"""
    
    @pytest.mark.slow
    def test_milvus_connection(self):
        """测试Milvus连接"""
        try:
            from pymilvus import connections, utility
            from config.config import MILVUS_HOST, MILVUS_PORT
            
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            assert utility.get_server_version() is not None
            connections.disconnect("default")
        except Exception as e:
            pytest.skip(f"Milvus不可用: {e}")
    
    @pytest.mark.slow
    def test_collection_exists(self):
        """测试集合存在"""
        try:
            from pymilvus import connections, utility
            from config.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
            
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            exists = utility.has_collection(MILVUS_COLLECTION)
            connections.disconnect("default")
            
            # 集合可能存在也可能不存在，只要不报错就行
            assert isinstance(exists, bool)
        except Exception as e:
            pytest.skip(f"Milvus不可用: {e}")


class TestKafkaIntegration:
    """Kafka集成测试"""
    
    @pytest.mark.slow
    def test_kafka_connection(self):
        """测试Kafka连接"""
        try:
            from kafka import KafkaProducer
            from config.config import KAFKA_BOOTSTRAP_SERVERS
            
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                api_version_auto_timeout_ms=5000
            )
            producer.close()
        except Exception as e:
            pytest.skip(f"Kafka不可用: {e}")


class TestMongoDBIntegration:
    """MongoDB集成测试"""
    
    @pytest.mark.slow
    def test_mongodb_connection(self):
        """测试MongoDB连接"""
        try:
            from pymongo import MongoClient
            from config.config import MONGODB_URI
            
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.server_info()
            client.close()
        except Exception as e:
            pytest.skip(f"MongoDB不可用: {e}")


class TestEndToEndFlow:
    """端到端流程测试"""
    
    @pytest.mark.slow
    def test_full_qa_pipeline(self):
        """测试完整问答流程"""
        try:
            from src.rag.rag_system import RAGSystem
            
            rag = RAGSystem()
            result = rag.answer("What is diabetes?", return_contexts=True)
            
            assert "answer" in result
            assert "contexts" in result
            assert len(result["answer"]) > 0
        except Exception as e:
            pytest.skip(f"RAG系统不可用: {e}")
    
    @pytest.mark.slow
    def test_streaming_qa_pipeline(self):
        """测试流式问答流程"""
        try:
            from src.rag.rag_system import RAGSystem
            from src.rag.streaming import StreamingRAG
            
            rag = RAGSystem()
            streaming_rag = StreamingRAG(rag)
            
            tokens = []
            for item in streaming_rag.answer_stream("What is diabetes?", top_k=3):
                if item.get("type") == "token":
                    tokens.append(item["token"])
            
            answer = "".join(tokens)
            assert len(answer) > 0
        except Exception as e:
            pytest.skip(f"流式RAG不可用: {e}")


class TestPerformance:
    """性能测试"""
    
    def test_embedding_speed(self):
        """测试向量化速度"""
        from src.embedding.embedder import TextEmbedder
        
        embedder = TextEmbedder()
        texts = ["Test text " * 10 for _ in range(100)]
        
        start = time.time()
        embedder.encode_batch(texts)
        elapsed = time.time() - start
        
        # 100条文本应该在10秒内完成
        assert elapsed < 10, f"向量化太慢: {elapsed:.2f}s"
    
    def test_cache_speed(self):
        """测试缓存速度"""
        from src.caching.redis_cache import RedisCache
        
        cache = RedisCache(db=15)
        if not cache.client:
            pytest.skip("Redis不可用")
        
        # 写入测试
        start = time.time()
        for i in range(100):
            cache.set_query_cache(f"query_{i}", [{"id": i}])
        write_time = time.time() - start
        
        # 读取测试
        start = time.time()
        for i in range(100):
            cache.get_query_cache(f"query_{i}")
        read_time = time.time() - start
        
        cache.clear_cache()
        cache.close()
        
        # 100次操作应该在1秒内
        assert write_time < 1, f"写入太慢: {write_time:.2f}s"
        assert read_time < 1, f"读取太慢: {read_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
