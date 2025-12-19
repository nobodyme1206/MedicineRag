#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索模块 - BM25 + 向量检索
结合关键词匹配和语义相似度
"""

from __future__ import annotations

import re
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import PROCESSED_DATA_DIR, LOGS_DIR, EMBEDDING_DATA_DIR
from src.utils.logger import setup_logger
from src.utils.exceptions import RetrievalError, handle_errors

logger = setup_logger("hybrid_search", LOGS_DIR / "hybrid_search.log")

# 类型别名
SearchResult = Dict[str, any]
BM25Result = Tuple[int, float]

# BM25索引缓存路径
BM25_INDEX_CACHE = EMBEDDING_DATA_DIR / "bm25_index.pkl"


class HybridSearcher:
    """混合检索器：BM25关键词 + 向量语义"""
    
    def __init__(
        self, 
        chunks_file: Optional[Path] = None, 
        sample_size: int = 500000,
        use_cache: bool = True
    ) -> None:
        """
        初始化混合检索器
        
        Args:
            chunks_file: chunks数据文件路径
            sample_size: BM25索引采样大小（避免内存溢出）
            use_cache: 是否使用缓存的BM25索引
            
        Raises:
            RetrievalError: 数据文件不存在或加载失败
        """
        parquet_file = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
        chunks_file = chunks_file or parquet_file
        
        if not chunks_file.exists():
            raise RetrievalError(f"数据文件不存在: {chunks_file}")
        
        logger.info(f"初始化混合检索器: {chunks_file}")
        
        # 尝试从缓存加载
        if use_cache and self._load_from_cache():
            logger.info(f"✅ 从缓存加载BM25索引成功: {len(self.chunks):,} 文档")
            return
        
        try:
            # 只加载必要的列
            df = pd.read_parquet(chunks_file, columns=['chunk_text', 'pmid', 'topic'])
            self.total_count: int = len(df)
            
            # 采样避免内存溢出
            if self.total_count > sample_size:
                logger.info(f"数据量 {self.total_count:,} 过大，采样 {sample_size:,} 条")
                df = df.sample(n=sample_size, random_state=42)
            
            self.chunks: List[Dict] = df.to_dict('records')
            self._normalize_fields()
            
            logger.info(f"加载 {len(self.chunks):,} 个文本块")
            self._build_bm25_index()
            
            # 保存到缓存
            if use_cache:
                self._save_to_cache()
            
        except Exception as e:
            raise RetrievalError(f"加载数据失败: {chunks_file}", e)
    
    def _load_from_cache(self) -> bool:
        """从缓存加载BM25索引"""
        if not BM25_INDEX_CACHE.exists():
            return False
        
        try:
            logger.info(f"尝试从缓存加载BM25索引: {BM25_INDEX_CACHE}")
            with open(BM25_INDEX_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.bm25 = cache_data['bm25']
            self.chunks = cache_data['chunks']
            self.chunk_ids = cache_data['chunk_ids']
            self.total_count = cache_data.get('total_count', len(self.chunks))
            return True
        except Exception as e:
            logger.warning(f"加载BM25缓存失败: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """保存BM25索引到缓存"""
        try:
            BM25_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'bm25': self.bm25,
                'chunks': self.chunks,
                'chunk_ids': self.chunk_ids,
                'total_count': self.total_count
            }
            with open(BM25_INDEX_CACHE, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✅ BM25索引已缓存: {BM25_INDEX_CACHE}")
        except Exception as e:
            logger.warning(f"保存BM25缓存失败: {e}")
    
    def _normalize_fields(self) -> None:
        """统一字段名"""
        for chunk in self.chunks:
            if 'content' in chunk and 'chunk_text' not in chunk:
                chunk['chunk_text'] = chunk['content']
    
    def _build_bm25_index(self) -> None:
        """构建BM25索引"""
        logger.info("构建BM25索引...")
        
        corpus: List[List[str]] = []
        self.chunk_ids: List[int] = []
        
        for i, chunk in enumerate(self.chunks):
            text = chunk.get('chunk_text', '')
            tokens = self._tokenize(text)
            corpus.append(tokens)
            self.chunk_ids.append(i)
        
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"✅ BM25索引构建完成: {len(corpus):,} 文档")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词列表
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    @handle_errors(default_return=[], log_level="warning")
    def bm25_search(
        self, 
        query: str, 
        top_k: int = 100
    ) -> List[BM25Result]:
        """
        BM25关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            [(chunk_id, score), ...]
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[SearchResult],
        alpha: float = 0.6,
        top_k: int = 10,
        use_rrf: bool = True
    ) -> List[SearchResult]:
        """
        混合检索：融合向量检索和BM25结果
        
        Args:
            query: 查询文本
            vector_results: 向量检索结果
            alpha: 向量检索权重（0-1），BM25权重为1-alpha
            top_k: 最终返回数量
            use_rrf: 是否使用RRF融合（更稳定）
            
        Returns:
            融合后的结果列表
        """
        if not vector_results:
            return []
        
        # 计算BM25分数
        query_tokens = self._tokenize(query)
        bm25_scores = self._compute_bm25_scores(query_tokens, vector_results)
        
        # 归一化分数
        bm25_scores = self._normalize_scores(bm25_scores)
        vector_scores = self._extract_vector_scores(vector_results)
        vector_scores = self._normalize_scores(vector_scores)
        
        if use_rrf:
            # 使用RRF融合（基于排名，更稳定）
            hybrid_results = self._fuse_with_rrf(
                vector_results, vector_scores, bm25_scores, alpha
            )
        else:
            # 使用加权融合
            hybrid_results = self._fuse_results(
                vector_results, vector_scores, bm25_scores, alpha
            )
        
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        final_results = hybrid_results[:top_k]
        
        logger.debug(
            f"混合检索: BM25权重={1-alpha:.2f}, 向量权重={alpha:.2f}, "
            f"RRF={use_rrf}, 返回{len(final_results)}个结果"
        )
        
        return final_results
    
    def _fuse_with_rrf(
        self,
        results: List[SearchResult],
        vector_scores: List[float],
        bm25_scores: List[float],
        alpha: float,
        k: int = 60
    ) -> List[SearchResult]:
        """
        使用RRF（Reciprocal Rank Fusion）融合
        
        RRF公式: score = alpha/(k+rank_vec) + (1-alpha)/(k+rank_bm25)
        
        Args:
            results: 检索结果
            vector_scores: 向量分数
            bm25_scores: BM25分数
            alpha: 向量权重
            k: RRF常数
            
        Returns:
            融合后的结果
        """
        # 计算向量排名
        vec_ranks = self._scores_to_ranks(vector_scores)
        # 计算BM25排名
        bm25_ranks = self._scores_to_ranks(bm25_scores)
        
        hybrid_results = []
        for i, r in enumerate(results):
            vec_rank = vec_ranks[i]
            bm25_rank = bm25_ranks[i]
            
            # RRF分数
            rrf_score = alpha / (k + vec_rank) + (1 - alpha) / (k + bm25_rank)
            
            hybrid_results.append({
                'id': r.get('id'),
                'text': r.get('text', '') or r.get('content', ''),
                'pmid': r.get('pmid', ''),
                'topic': r.get('topic', ''),
                'score': rrf_score,
                'hybrid_score': rrf_score,
                'bm25_score': bm25_scores[i],
                'bm25_rank': bm25_rank,
                'vector_score': vector_scores[i],
                'vector_rank': vec_rank
            })
        
        return hybrid_results
    
    def _scores_to_ranks(self, scores: List[float]) -> List[int]:
        """将分数转换为排名（1-based）"""
        # 创建(index, score)对并按分数降序排序
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 分配排名
        ranks = [0] * len(scores)
        for rank, (idx, _) in enumerate(indexed_scores, 1):
            ranks[idx] = rank
        
        return ranks
    
    def _compute_bm25_scores(
        self, 
        query_tokens: List[str], 
        results: List[SearchResult]
    ) -> List[float]:
        """计算BM25分数"""
        scores = []
        for r in results:
            text = r.get('text', '') or r.get('content', '')
            if text:
                doc_tokens = self._tokenize(text)
                score = sum(1 for t in query_tokens if t in doc_tokens)
                score = score / (len(doc_tokens) + 1) * len(query_tokens)
            else:
                score = 0.0
            scores.append(score)
        return scores
    
    def _extract_vector_scores(self, results: List[SearchResult]) -> List[float]:
        """提取向量检索分数"""
        scores = []
        for r in results:
            score = r.get('score', 0)
            if score == 0:
                distance = r.get('distance', 1.0)
                score = 1 - distance if distance <= 1 else 1 / (1 + distance)
            scores.append(score)
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数到0-1"""
        if not scores:
            return scores
        
        max_score = max(scores) if max(scores) > 0 else 1
        min_score = min(scores)
        range_score = max_score - min_score if max_score > min_score else 1
        
        return [(s - min_score) / range_score for s in scores]
    
    def _fuse_results(
        self,
        results: List[SearchResult],
        vector_scores: List[float],
        bm25_scores: List[float],
        alpha: float
    ) -> List[SearchResult]:
        """融合结果"""
        hybrid_results = []
        
        for i, r in enumerate(results):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            vector_score = vector_scores[i] if i < len(vector_scores) else 0
            hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            hybrid_results.append({
                'id': r.get('id'),
                'text': r.get('text', '') or r.get('content', ''),
                'pmid': r.get('pmid', ''),
                'topic': r.get('topic', ''),
                'score': hybrid_score,
                'hybrid_score': hybrid_score,
                'bm25_score': bm25_score,
                'vector_score': vector_score
            })
        
        return hybrid_results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """根据ID获取chunk"""
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None


def main() -> None:
    """测试混合检索"""
    logger.info("=" * 60)
    logger.info("混合检索模块测试")
    logger.info("=" * 60)
    
    try:
        searcher = HybridSearcher()
        
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"测试查询: {test_query}")
        
        # 模拟向量检索结果
        mock_results = [
            {'id': 100, 'distance': 0.2, 'text': 'diabetes symptoms include...'},
            {'id': 200, 'distance': 0.3, 'text': 'type 2 diabetes...'},
            {'id': 300, 'distance': 0.4, 'text': 'hyperglycemia signs...'},
        ]
        
        results = searcher.hybrid_search(test_query, mock_results, alpha=0.6, top_k=10)
        
        logger.info(f"混合检索结果: {len(results)} 条")
        for i, r in enumerate(results[:3], 1):
            logger.info(
                f"  [{i}] hybrid={r['hybrid_score']:.3f} "
                f"bm25={r['bm25_score']:.3f} vec={r['vector_score']:.3f}"
            )
        
        logger.info("✅ 测试完成")
        
    except RetrievalError as e:
        logger.error(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()
