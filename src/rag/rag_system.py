# -*- coding: utf-8 -*-
"""
RAG系统核心 - 检索增强生成
支持: Rerank重排序、Redis缓存、MongoDB文档存储、MinIO对象存储
"""

import time
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import *
from src.utils.logger import setup_logger
from src.embedding.embedder import TextEmbedder
from src.retrieval.milvus_manager import MilvusManager
from openai import OpenAI

logger = setup_logger("rag_system", LOGS_DIR / "rag.log")

# 全局存储实例（延迟初始化）
_mongodb_instance = None
_minio_instance = None


def get_mongodb():
    """获取MongoDB单例实例"""
    global _mongodb_instance
    if _mongodb_instance is None:
        try:
            from src.storage.mongodb_storage import MongoDBStorage
            _mongodb_instance = MongoDBStorage(
                host=MONGODB_HOST,
                port=MONGODB_PORT,
                database=MONGODB_DATABASE
            )
        except Exception as e:
            logger.warning(f"MongoDB初始化失败: {e}")
    return _mongodb_instance


def get_minio():
    """获取MinIO单例实例"""
    global _minio_instance
    if _minio_instance is None:
        try:
            from src.storage.minio_storage import MinIOStorage
            _minio_instance = MinIOStorage(
                endpoint=MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_SECURE
            )
            # 确保必要的bucket存在
            _minio_instance.create_bucket(MINIO_BUCKET_MODELS)
            _minio_instance.create_bucket(MINIO_BUCKET_BACKUPS)
            _minio_instance.create_bucket(MINIO_BUCKET_DATA)
        except Exception as e:
            logger.warning(f"MinIO初始化失败: {e}")
    return _minio_instance


class RAGSystem:
    """RAG问答系统（支持Rerank、混合检索、HyDE、Redis缓存、MongoDB日志、MinIO存储）"""
    
    def __init__(self, use_rerank: bool = None, use_hybrid: bool = True, 
                 use_query_rewrite: bool = False, use_hyde: bool = None,
                 use_ensemble: bool = False, use_cache: bool = True):
        """
        初始化RAG系统
        
        Args:
            use_rerank: 是否使用Rerank，None则使用配置文件中的默认值
            use_hybrid: 是否使用混合检索(BM25+向量)
            use_query_rewrite: 是否使用查询改写
            use_hyde: 是否使用HyDE假设文档嵌入，None则使用配置文件中的默认值
            use_ensemble: 是否使用集成检索（RRF多路融合）
            use_cache: 是否使用Redis缓存
        """
        logger.info("初始化RAG系统...")
        
        # 配置功能开关
        self.use_rerank = use_rerank if use_rerank is not None else USE_RERANK
        self.use_hybrid = use_hybrid
        self.use_query_rewrite = use_query_rewrite
        self.use_hyde = use_hyde if use_hyde is not None else getattr(__import__('config.config', fromlist=['USE_HYDE']), 'USE_HYDE', False)
        self.use_ensemble = use_ensemble
        self.use_cache = use_cache
        self.reranker = None
        self.hybrid_searcher = None
        self.query_rewriter = None
        self.hyde = None
        self.ensemble_retriever = None
        self.redis_cache = None
        
        # 0. 初始化存储服务（MongoDB + MinIO + Redis）
        logger.info("初始化存储服务...")
        self._init_storage_services()
        
        # 1. 加载Embedding模型
        logger.info("加载Embedding模型...")
        self.embedder = TextEmbedder()
        
        # 2. 连接向量数据库
        logger.info("连接向量数据库...")
        self.milvus = MilvusManager()
        self.milvus.load_collection()
        
        # 3. 初始化LLM客户端（硅基流动）
        logger.info("初始化LLM客户端（硅基流动）...")
        self.llm_client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
        
        # 4. 初始化Reranker（如果启用）
        if self.use_rerank:
            logger.info("加载Rerank模型...")
            try:
                from src.retrieval.reranker import Reranker
                self.reranker = Reranker()
                logger.info("✅ Rerank模型加载成功")
            except Exception as e:
                logger.warning(f"Rerank模型加载失败，将不使用Rerank: {e}")
                self.use_rerank = False
        
        # 5. 初始化混合检索（如果启用）
        if self.use_hybrid:
            logger.info("初始化混合检索器...")
            try:
                from src.retrieval.hybrid_searcher import HybridSearcher
                self.hybrid_searcher = HybridSearcher()
                logger.info("✅ 混合检索器初始化成功")
            except Exception as e:
                logger.warning(f"混合检索器初始化失败: {e}")
                self.use_hybrid = False
        
        # 6. 初始化查询改写（如果启用）
        if self.use_query_rewrite:
            logger.info("初始化查询改写器...")
            try:
                from src.retrieval.query_rewriter import QueryRewriter
                self.query_rewriter = QueryRewriter()
                logger.info("✅ 查询改写器初始化成功")
            except Exception as e:
                logger.warning(f"查询改写器初始化失败: {e}")
                self.use_query_rewrite = False
        
        # 7. 初始化HyDE（如果启用）
        if self.use_hyde:
            logger.info("初始化HyDE模块...")
            try:
                from src.retrieval.hyde import HyDE
                self.hyde = HyDE()
                logger.info("✅ HyDE模块初始化成功")
            except Exception as e:
                logger.warning(f"HyDE模块初始化失败: {e}")
                self.use_hyde = False
        
        # 8. 初始化集成检索（如果启用）
        if self.use_ensemble:
            logger.info("初始化集成检索器（RRF融合）...")
            try:
                from src.retrieval.rrf_fusion import EnsembleRetriever
                self.ensemble_retriever = EnsembleRetriever(
                    embedder=self.embedder,
                    milvus_manager=self.milvus,
                    hybrid_searcher=self.hybrid_searcher if self.use_hybrid else None
                )
                logger.info("✅ 集成检索器初始化成功")
            except Exception as e:
                logger.warning(f"集成检索器初始化失败: {e}")
                self.use_ensemble = False
        
        logger.info("✅ RAG系统初始化完成")
    
    def _init_storage_services(self):
        """初始化所有存储服务：Redis缓存、MongoDB文档存储、MinIO对象存储"""
        # Redis缓存 + 向量缓存管理器
        if self.use_cache:
            logger.info("  → 初始化Redis缓存...")
            try:
                from src.caching.redis_cache import RedisCache, VectorCacheManager
                self.redis_cache = RedisCache()
                self.vector_cache = VectorCacheManager(self.redis_cache)
                logger.info("  ✅ Redis缓存 + 向量缓存初始化成功")
            except Exception as e:
                logger.warning(f"  ⚠️ Redis缓存初始化失败: {e}")
                self.use_cache = False
                self.vector_cache = None
        else:
            self.vector_cache = None
        
        # MongoDB文档存储
        logger.info("  → 初始化MongoDB文档存储...")
        self.mongodb = get_mongodb()
        if self.mongodb:
            logger.info("  ✅ MongoDB文档存储初始化成功")
        else:
            logger.warning("  ⚠️ MongoDB不可用，查询日志将不会保存")
        
        # MinIO对象存储
        logger.info("  → 初始化MinIO对象存储...")
        self.minio = get_minio()
        if self.minio:
            logger.info("  ✅ MinIO对象存储初始化成功")
        else:
            logger.warning("  ⚠️ MinIO不可用，备份功能将不可用")
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """生成缓存键"""
        config_str = f"{self.use_rerank}_{self.use_hybrid}_{self.use_hyde}_{top_k}"
        return hashlib.md5(f"{query}_{config_str}".encode()).hexdigest()
    
    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        """
        检索相关文档（支持查询改写、混合检索、HyDE、集成检索、Rerank重排序、Redis缓存）
        
        Args:
            query: 用户查询
            top_k: 返回Top-K结果
            
        Returns:
            检索结果列表
        """
        original_query = query
        
        # 0. 检查Redis缓存
        cache_key = self._get_cache_key(query, top_k)
        if self.use_cache and self.redis_cache:
            cached_result = self.redis_cache.get_query_cache(cache_key)
            if cached_result:
                logger.info(f"✅ Redis缓存命中: {query[:30]}...")
                return cached_result
        
        # 1. 如果启用集成检索，使用EnsembleRetriever
        if self.use_ensemble and self.ensemble_retriever:
            logger.info("使用集成检索（RRF多路融合）...")
            candidates = self.ensemble_retriever.retrieve_ensemble(
                original_query,
                top_k=top_k * 3 if self.use_rerank else top_k,
                use_hyde=self.use_hyde,
                use_hybrid=self.use_hybrid
            )
            # 如果启用Rerank，对集成结果重排序
            if self.use_rerank and self.reranker:
                logger.info(f"使用Rerank对 {len(candidates)} 个候选进行重排序...")
                candidates = self.reranker.rerank(original_query, candidates, top_k=top_k)
                logger.info(f"Rerank后保留Top-{len(candidates)}结果")
            
            final_results = candidates[:top_k]
            # 保存到Redis缓存
            if self.use_cache and self.redis_cache:
                self.redis_cache.set_query_cache(cache_key, final_results, ttl=3600)
            return final_results
        
        # 1. 查询改写（如果启用）
        if self.use_query_rewrite and self.query_rewriter:
            logger.info(f"原始查询: {query}")
            query = self.query_rewriter.expand_medical_query(query)
            logger.info(f"扩展查询: {query}")
        
        # 2. HyDE假设文档嵌入（如果启用）
        if self.use_hyde and self.hyde:
            logger.info("使用HyDE生成假设文档...")
            query = self.hyde.get_hyde_query(original_query)
            logger.info(f"HyDE假设文档: {query[:100]}...")
        
        # 3. Query向量化（优先使用向量缓存）
        if self.vector_cache:
            query_embedding = self.vector_cache.get_or_compute_vector(query)
        else:
            query_embedding = self.embedder.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # 4. 向量检索
        search_top_k = top_k * 3 if (self.use_rerank or self.use_hybrid) else top_k
        results = self.milvus.search(query_embedding, top_k=search_top_k)
        
        if not results or not results[0]:
            return []
        
        candidates = results[0]
        
        # 5. 混合检索（如果启用）
        if self.use_hybrid and self.hybrid_searcher:
            logger.info(f"使用混合检索融合BM25和向量结果...")
            candidates = self.hybrid_searcher.hybrid_search(
                original_query,  # 使用原始查询做BM25
                candidates,
                alpha=0.6,  # 向量检索权重60%，BM25权重40%
                top_k=search_top_k
            )
        
        # 6. Rerank重排序（如果启用）
        if self.use_rerank and self.reranker:
            logger.info(f"使用Rerank对 {len(candidates)} 个候选进行重排序...")
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
            logger.info(f"Rerank后保留Top-{len(candidates)}结果")
        
        final_results = candidates[:top_k]
        
        # 7. 保存到Redis缓存
        if self.use_cache and self.redis_cache:
            self.redis_cache.set_query_cache(cache_key, final_results, ttl=3600)
            logger.info(f"已缓存查询结果: {query[:30]}...")
        
        return final_results
    
    def generate(self, query: str, contexts: List[Dict]) -> Tuple[str, Dict]:
        """
        基于检索结果生成答案
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文
            
        Returns:
            (答案, 元信息)
        """
        if not contexts:
            return "抱歉，没有找到相关信息。", {}
        
        # 构建prompt
        context_text = "\n\n".join([
            f"[文档{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts[:RERANK_TOP_K])
        ])
        
        system_prompt = """You are an expert medical knowledge assistant with deep expertise in clinical medicine, diagnostics, and treatment protocols.

Your task: Provide accurate, comprehensive, evidence-based answers using ONLY the provided scientific literature.

Core Principles:
1. **Evidence-Based Medicine**: Base ALL statements on provided documents. Never speculate or add external knowledge
2. **Clinical Precision**: Use accurate medical terminology, cite specific mechanisms, dosages, and clinical findings
3. **Comprehensive Coverage**: Address ALL relevant aspects:
   - Definition & Pathophysiology
   - Risk factors & Epidemiology
   - Clinical manifestations & Symptoms
   - Diagnostic criteria & Tests
   - Treatment options & Management
   - Prognosis & Outcomes
4. **Structured Response**: Organize information clearly with distinct sections
5. **Source Attribution**: Reference specific documents [Document N] for key claims
6. **Honesty**: If literature is insufficient, explicitly state "The provided literature does not contain information about..."
7. **Language Matching**: Respond in the SAME language as the question

Response Format:
- Start with a direct, concise answer to the core question
- Follow with detailed explanation organized by topic
- Include relevant clinical details (values, percentages, mechanisms)
- End with practical implications or key takeaways"""
        
        user_prompt = f"""### REFERENCE LITERATURE:
{context_text}

### MEDICAL QUESTION:
{query}

### INSTRUCTIONS:
Analyze the above literature and provide a comprehensive, evidence-based answer. Include:
- Direct answer to the question
- Supporting evidence from the documents (with citations)
- Relevant clinical details and mechanisms
- Key points summary

Answer:"""
        
        # 调用LLM
        try:
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=SILICONFLOW_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P
            )
            
            generation_time = time.time() - start_time
            
            answer = response.choices[0].message.content
            
            # 元信息
            metadata = {
                "model": SILICONFLOW_MODEL,
                "generation_time": generation_time,
                "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "num_contexts": len(contexts)
            }
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"生成答案时出错: {str(e)}", {}
    
    def answer(self, query: str, return_contexts: bool = True) -> Dict:
        """
        完整的问答流程
        
        Args:
            query: 用户问题
            return_contexts: 是否返回检索的上下文
            
        Returns:
            问答结果字典
        """
        logger.info(f"用户提问: {query}")
        
        start_time = time.time()
        
        # 1. 检索
        retrieval_start = time.time()
        contexts = self.retrieve(query, top_k=RETRIEVAL_TOP_K)
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"检索到 {len(contexts)} 个相关文档，耗时 {retrieval_time:.3f}秒")
        
        # 2. 生成
        answer, gen_metadata = self.generate(query, contexts)
        
        total_time = time.time() - start_time
        
        # 构建结果
        result = {
            "query": query,
            "answer": answer,
            "retrieval_time": retrieval_time,
            "generation_time": gen_metadata.get("generation_time", 0),
            "total_time": total_time,
            "num_contexts": len(contexts)
        }
        
        if return_contexts:
            result["contexts"] = [
                {
                    "pmid": ctx["pmid"],
                    "text": ctx["text"],
                    "score": ctx["score"]
                }
                for ctx in contexts[:RERANK_TOP_K]
            ]
        
        # 3. 记录查询日志到MongoDB
        self._log_query_to_mongodb(query, contexts, result)
        
        logger.info(f"回答生成完成，总耗时 {total_time:.3f}秒")
        
        return result
    
    def _log_query_to_mongodb(self, query: str, contexts: List[Dict], result: Dict):
        """记录查询日志到MongoDB"""
        if not self.mongodb:
            return
        try:
            metrics = {
                "retrieval_time_ms": result["retrieval_time"] * 1000,
                "generation_time_ms": result["generation_time"] * 1000,
                "total_time_ms": result["total_time"] * 1000,
                "num_contexts": len(contexts),
                "use_hybrid": self.use_hybrid,
                "use_rerank": self.use_rerank,
                "cache_hit": False  # 如果走到这里说明没有命中缓存
            }
            self.mongodb.log_query(query, contexts, metrics)
        except Exception as e:
            logger.warning(f"MongoDB日志记录失败: {e}")
    
    def batch_answer(self, queries: List[str]) -> List[Dict]:
        """批量问答"""
        results = []
        for query in queries:
            result = self.answer(query)
            results.append(result)
        return results
    
    def save_evaluation_to_mongodb(self, eval_results: Dict) -> bool:
        """保存评估结果到MongoDB"""
        if not self.mongodb:
            logger.warning("MongoDB不可用，无法保存评估结果")
            return False
        try:
            self.mongodb.save_evaluation_results(eval_results)
            logger.info("✅ 评估结果已保存到MongoDB")
            return True
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
            return False
    
    def get_query_statistics(self) -> Dict:
        """从MongoDB获取查询统计"""
        if not self.mongodb:
            return {}
        try:
            return self.mongodb.get_query_statistics()
        except Exception as e:
            logger.error(f"获取查询统计失败: {e}")
            return {}
    
    def backup_to_minio(self, file_path: Path, object_name: str = None, 
                        bucket: str = None) -> bool:
        """备份文件到MinIO"""
        if not self.minio:
            logger.warning("MinIO不可用，无法备份")
            return False
        try:
            bucket = bucket or MINIO_BUCKET_BACKUPS
            object_name = object_name or file_path.name
            return self.minio.upload_file(bucket, object_name, file_path)
        except Exception as e:
            logger.error(f"备份到MinIO失败: {e}")
            return False
    
    def list_minio_backups(self, bucket: str = None) -> List[Dict]:
        """列出MinIO中的备份文件"""
        if not self.minio:
            return []
        try:
            bucket = bucket or MINIO_BUCKET_BACKUPS
            return self.minio.list_objects(bucket)
        except Exception as e:
            logger.error(f"列出MinIO备份失败: {e}")
            return []
    
    def get_storage_status(self) -> Dict:
        """获取所有存储服务状态"""
        status = {
            "redis": {"enabled": self.use_cache, "connected": self.redis_cache is not None},
            "mongodb": {"enabled": True, "connected": self.mongodb is not None},
            "minio": {"enabled": True, "connected": self.minio is not None},
            "milvus": {"enabled": True, "connected": True}
        }
        
        # 获取详细统计
        if self.mongodb:
            try:
                status["mongodb"]["query_logs"] = self.mongodb.get_collection_stats("query_logs")
                status["mongodb"]["evaluations"] = self.mongodb.get_collection_stats("evaluation_results")
            except:
                pass
        
        if self.minio:
            try:
                status["minio"]["backups"] = len(self.minio.list_objects(MINIO_BUCKET_BACKUPS))
            except:
                pass
        
        return status
    
    def generate_without_context(self, query: str) -> Tuple[str, Dict]:
        """
        不使用检索结果，直接用LLM回答（用于基线对比）
        
        Args:
            query: 用户查询
            
        Returns:
            (答案, 元信息)
        """
        system_prompt = """你是一个专业的医学知识问答助手。请基于你的医学知识回答用户问题。

要求：
1. 回答要专业、准确、简洁
2. 如果不确定，请明确告知
3. 使用中文回答"""
        
        user_prompt = f"用户问题：{query}\n\n请回答这个医学问题："
        
        try:
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=SILICONFLOW_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P
            )
            
            generation_time = time.time() - start_time
            
            answer = response.choices[0].message.content
            
            metadata = {
                "model": SILICONFLOW_MODEL,
                "generation_time": generation_time,
                "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "method": "baseline_no_context"
            }
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"基线LLM生成失败: {e}")
            return f"生成答案时出错: {str(e)}", {}
    
    def answer_baseline(self, query: str) -> Dict:
        """
        基线问答（不使用RAG检索）
        
        Args:
            query: 用户问题
            
        Returns:
            问答结果字典
        """
        logger.info(f"基线问答（无RAG）: {query}")
        
        start_time = time.time()
        
        # 直接生成
        answer, gen_metadata = self.generate_without_context(query)
        
        total_time = time.time() - start_time
        
        result = {
            "query": query,
            "answer": answer,
            "retrieval_time": 0,
            "generation_time": gen_metadata.get("generation_time", 0),
            "total_time": total_time,
            "method": "baseline_no_rag"
        }
        
        return result


def main():
    """测试RAG系统"""
    logger.info("="*50)
    logger.info("测试RAG系统")
    logger.info("="*50)
    
    # 初始化系统
    rag = RAGSystem()
    
    # 测试问题
    test_queries = [
        "什么是糖尿病？",
        "如何预防心血管疾病？",
        "癌症的常见治疗方法有哪些？"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"问题: {query}")
        logger.info(f"{'='*50}")
        
        result = rag.answer(query)
        
        logger.info(f"\n答案:\n{result['answer']}")
        logger.info(f"\n性能指标:")
        logger.info(f"  检索时间: {result['retrieval_time']:.3f}秒")
        logger.info(f"  生成时间: {result['generation_time']:.3f}秒")
        logger.info(f"  总时间: {result['total_time']:.3f}秒")
        logger.info(f"  参考文档数: {result['num_contexts']}")


if __name__ == "__main__":
    main()
