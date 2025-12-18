#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRF (Reciprocal Rank Fusion) 多路召回融合模块
融合多个检索结果列表，提升检索多样性和准确性
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("rrf_fusion", LOGS_DIR / "rrf_fusion.log")


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) 多路召回融合"""
    
    def __init__(self, k: int = 60):
        """
        初始化RRF融合器
        
        Args:
            k: RRF常数，通常设为60
        """
        self.k = k
        logger.info(f"初始化RRF融合器，k={k}")
    
    def fuse(self, result_lists: List[List[Dict]], weights: List[float] = None, top_k: int = 10) -> List[Dict]:
        """
        使用RRF算法融合多个检索结果列表
        
        RRF公式: score(d) = Σ (weight_i / (k + rank_i(d)))
        
        Args:
            result_lists: 多个检索结果列表，每个列表包含字典 {'id': ..., 'text': ..., ...}
            weights: 各列表的权重，默认等权重
            top_k: 返回Top-K结果
            
        Returns:
            融合后的结果列表
        """
        if not result_lists:
            return []
        
        # 默认等权重
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # 确保权重数量匹配
        if len(weights) != len(result_lists):
            weights = [1.0] * len(result_lists)
        
        # 计算RRF分数
        rrf_scores = defaultdict(float)
        doc_info = {}  # 存储文档详细信息
        
        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]
            
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get('id')
                if doc_id is None:
                    continue
                
                # RRF公式: weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                rrf_scores[doc_id] += rrf_score
                
                # 保存文档信息（优先保留最高排名的版本）
                if doc_id not in doc_info:
                    doc_info[doc_id] = doc.copy()
        
        # 按RRF分数排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 构建结果
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_info:
                result = doc_info[doc_id].copy()
                result['rrf_score'] = rrf_score
                fused_results.append(result)
        
        logger.info(f"RRF融合: {len(result_lists)}路召回 → {len(fused_results)}个结果")
        
        return fused_results
    
    def fuse_with_original_scores(self, 
                                   result_lists: List[List[Dict]], 
                                   score_keys: List[str],
                                   weights: List[float] = None,
                                   top_k: int = 10) -> List[Dict]:
        """
        融合结果并保留原始分数信息
        
        Args:
            result_lists: 检索结果列表
            score_keys: 每个列表对应的分数键名
            weights: 权重列表
            top_k: 返回数量
            
        Returns:
            带有多路分数的融合结果
        """
        if not result_lists:
            return []
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        rrf_scores = defaultdict(float)
        doc_info = {}
        doc_scores = defaultdict(dict)
        
        for list_idx, (results, score_key) in enumerate(zip(result_lists, score_keys)):
            weight = weights[list_idx]
            
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get('id')
                if doc_id is None:
                    continue
                
                # RRF分数
                rrf_score = weight / (self.k + rank)
                rrf_scores[doc_id] += rrf_score
                
                # 保存原始分数
                original_score = doc.get('score', doc.get('distance', doc.get('similarity', 0)))
                doc_scores[doc_id][score_key] = original_score
                doc_scores[doc_id][f'{score_key}_rank'] = rank
                
                # 文档信息
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        'id': doc_id,
                        'text': doc.get('text', ''),
                        'pmid': doc.get('pmid', ''),
                    }
        
        # 排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 构建结果
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_info:
                result = doc_info[doc_id].copy()
                result['rrf_score'] = round(rrf_score, 6)
                result.update(doc_scores[doc_id])
                fused_results.append(result)
        
        return fused_results


class EnsembleRetriever:
    """集成检索器：组合多种检索策略"""
    
    def __init__(self, embedder, milvus_manager, hybrid_searcher=None):
        """
        初始化集成检索器
        
        Args:
            embedder: 文本嵌入器
            milvus_manager: Milvus管理器
            hybrid_searcher: 混合检索器（可选）
        """
        self.embedder = embedder
        self.milvus = milvus_manager
        self.hybrid_searcher = hybrid_searcher
        self.rrf = RRFFusion(k=60)
        
        # 可选：HyDE
        self.hyde = None
        try:
            from src.retrieval.hyde import HyDE
            self.hyde = HyDE()
            logger.info("✅ HyDE模块已加载")
        except Exception as e:
            logger.warning(f"HyDE模块加载失败: {e}")
        
        logger.info("✅ 集成检索器初始化完成")
    
    def retrieve_ensemble(self, 
                          query: str, 
                          top_k: int = 10,
                          use_hyde: bool = True,
                          use_hybrid: bool = True,
                          weights: List[float] = None) -> List[Dict]:
        """
        多路召回集成检索
        
        策略:
        1. 原始查询向量检索
        2. HyDE假设文档向量检索
        3. BM25关键词检索（通过hybrid_searcher）
        
        Args:
            query: 用户查询
            top_k: 返回数量
            use_hyde: 是否使用HyDE
            use_hybrid: 是否使用混合检索
            weights: RRF权重 [原始向量, HyDE, BM25]
            
        Returns:
            融合后的检索结果
        """
        result_lists = []
        score_keys = []
        
        # 1. 原始查询向量检索
        query_embedding = self.embedder.encode_single(query).reshape(1, -1)
        vector_results = self.milvus.search(query_embedding, top_k=top_k * 3)
        if vector_results and vector_results[0]:
            result_lists.append(vector_results[0])
            score_keys.append('vector_score')
            logger.info(f"原始向量检索: {len(vector_results[0])} 个结果")
        
        # 2. HyDE假设文档检索
        if use_hyde and self.hyde:
            try:
                hypo_doc = self.hyde.get_hyde_query(query)
                hyde_embedding = self.embedder.encode_single(hypo_doc).reshape(1, -1)
                hyde_results = self.milvus.search(hyde_embedding, top_k=top_k * 3)
                if hyde_results and hyde_results[0]:
                    result_lists.append(hyde_results[0])
                    score_keys.append('hyde_score')
                    logger.info(f"HyDE检索: {len(hyde_results[0])} 个结果")
            except Exception as e:
                logger.warning(f"HyDE检索失败: {e}")
        
        # 3. BM25检索（通过hybrid_searcher）
        if use_hybrid and self.hybrid_searcher:
            try:
                bm25_results = self.hybrid_searcher.bm25_search(query, top_k=top_k * 3)
                # 转换为标准格式
                bm25_docs = []
                for idx, score in bm25_results:
                    chunk = self.hybrid_searcher.get_chunk_by_id(idx)
                    if chunk:
                        bm25_docs.append({
                            'id': idx,
                            'text': chunk['chunk_text'],
                            'pmid': chunk.get('pmid', ''),
                            'score': score
                        })
                if bm25_docs:
                    result_lists.append(bm25_docs)
                    score_keys.append('bm25_score')
                    logger.info(f"BM25检索: {len(bm25_docs)} 个结果")
            except Exception as e:
                logger.warning(f"BM25检索失败: {e}")
        
        # 4. RRF融合
        if len(result_lists) > 1:
            # 默认权重：原始向量0.4, HyDE 0.35, BM25 0.25
            if weights is None:
                weights = [0.4, 0.35, 0.25][:len(result_lists)]
            
            fused_results = self.rrf.fuse_with_original_scores(
                result_lists, score_keys, weights=weights, top_k=top_k
            )
            logger.info(f"RRF融合完成: {len(fused_results)} 个最终结果")
            return fused_results
        elif result_lists:
            return result_lists[0][:top_k]
        else:
            return []


if __name__ == "__main__":
    # 测试RRF融合
    print("=" * 70)
    print("🔀 RRF融合模块测试")
    print("=" * 70)
    
    rrf = RRFFusion(k=60)
    
    # 模拟多路召回结果
    list1 = [
        {'id': 1, 'text': 'doc1', 'score': 0.95},
        {'id': 2, 'text': 'doc2', 'score': 0.90},
        {'id': 3, 'text': 'doc3', 'score': 0.85},
    ]
    
    list2 = [
        {'id': 2, 'text': 'doc2', 'score': 0.92},
        {'id': 4, 'text': 'doc4', 'score': 0.88},
        {'id': 1, 'text': 'doc1', 'score': 0.80},
    ]
    
    list3 = [
        {'id': 3, 'text': 'doc3', 'score': 0.93},
        {'id': 1, 'text': 'doc1', 'score': 0.85},
        {'id': 5, 'text': 'doc5', 'score': 0.75},
    ]
    
    print("\n📥 输入:")
    print(f"  列表1: {[d['id'] for d in list1]}")
    print(f"  列表2: {[d['id'] for d in list2]}")
    print(f"  列表3: {[d['id'] for d in list3]}")
    
    fused = rrf.fuse([list1, list2, list3], weights=[0.4, 0.35, 0.25], top_k=5)
    
    print("\n📤 RRF融合结果:")
    for i, doc in enumerate(fused, 1):
        print(f"  [{i}] ID={doc['id']}, RRF Score={doc['rrf_score']:.6f}")
    
    print("\n✅ RRF融合测试完成!")
