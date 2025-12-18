#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索模块 - BM25 + 向量检索
结合关键词匹配和语义相似度
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from rank_bm25 import BM25Okapi
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("hybrid_search", LOGS_DIR / "hybrid_search.log")


class HybridSearcher:
    """混合检索器：BM25关键词 + 向量语义"""
    
    def __init__(self, chunks_file: Path = None, sample_size: int = 100000):
        """
        初始化混合检索器
        
        Args:
            chunks_file: chunks数据文件路径（默认使用Parquet格式）
            sample_size: BM25索引采样大小（避免内存溢出）
        """
        import pandas as pd
        
        # 使用Parquet格式（更快、更省空间）
        parquet_file = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
        
        if chunks_file is None:
            chunks_file = parquet_file
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {chunks_file}，请先运行数据处理生成Parquet文件")
        
        logger.info(f"初始化混合检索器，加载Parquet数据: {chunks_file}")
        
        # 只加载必要的列，减少内存占用
        df = pd.read_parquet(chunks_file, columns=['chunk_text', 'pmid', 'topic'])
        total_count = len(df)
        
        # 采样以避免内存溢出（416万条太多）
        if total_count > sample_size:
            logger.info(f"数据量 {total_count:,} 过大，采样 {sample_size:,} 条用于BM25索引")
            df = df.sample(n=sample_size, random_state=42)
        
        self.chunks = df.to_dict('records')
        
        # 统一字段名
        for chunk in self.chunks:
            if 'content' in chunk and 'chunk_text' not in chunk:
                chunk['chunk_text'] = chunk['content']
        
        logger.info(f"加载 {len(self.chunks)} 个文本块用于BM25")
        
        # 构建BM25索引
        self._build_bm25_index()
        self.total_count = total_count
        
    def _build_bm25_index(self):
        """构建BM25索引"""
        logger.info("开始构建BM25索引...")
        
        # 提取所有文本并分词
        corpus = []
        self.chunk_ids = []
        
        for i, chunk in enumerate(self.chunks):
            text = chunk['chunk_text']
            # 简单分词（按空格和标点）
            tokens = self._tokenize(text)
            corpus.append(tokens)
            self.chunk_ids.append(i)
        
        # 构建BM25
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"✅ BM25索引构建完成，共 {len(corpus)} 个文档")
        
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词列表
        """
        # 转小写
        text = text.lower()
        # 按空格分词，移除标点
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def bm25_search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
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
        
        # 获取top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def hybrid_search(self, 
                     query: str,
                     vector_results: List[Dict],
                     alpha: float = 0.6,
                     top_k: int = 10) -> List[Dict]:
        """
        混合检索：对向量检索结果用BM25重新评分并融合
        
        Args:
            query: 查询文本
            vector_results: 向量检索结果 [{'id': ..., 'distance': ..., 'text': ...}, ...]
            alpha: 向量检索权重（0-1），BM25权重为1-alpha
            top_k: 最终返回数量
            
        Returns:
            融合后的结果列表
        """
        if not vector_results:
            return []
        
        # 1. 对向量检索结果计算BM25分数
        query_tokens = self._tokenize(query)
        
        bm25_scores = []
        for r in vector_results:
            text = r.get('text', '') or r.get('content', '')
            if text:
                doc_tokens = self._tokenize(text)
                # 计算BM25分数（简化版：词频匹配）
                score = sum(1 for t in query_tokens if t in doc_tokens)
                # 加上IDF权重
                score = score / (len(doc_tokens) + 1) * len(query_tokens)
            else:
                score = 0
            bm25_scores.append(score)
        
        # 2. 归一化BM25分数
        if bm25_scores:
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = [s / max_bm25 for s in bm25_scores]
        
        # 3. 归一化向量检索分数（score字段，越高越好）
        vector_scores = []
        for r in vector_results:
            # Milvus返回的score是相似度（COSINE），越高越好
            score = r.get('score', 0)
            if score == 0:
                # 如果没有score，用distance转换
                distance = r.get('distance', 1.0)
                score = 1 - distance if distance <= 1 else 1 / (1 + distance)
            vector_scores.append(score)
        
        if vector_scores:
            max_vec = max(vector_scores) if max(vector_scores) > 0 else 1
            min_vec = min(vector_scores)
            range_vec = max_vec - min_vec if max_vec > min_vec else 1
            vector_scores = [(s - min_vec) / range_vec for s in vector_scores]
        
        # 4. 融合分数并重排序
        hybrid_results = []
        for i, r in enumerate(vector_results):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            vector_score = vector_scores[i] if i < len(vector_scores) else 0
            hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            result = {
                'id': r.get('id'),
                'text': r.get('text', '') or r.get('content', ''),
                'pmid': r.get('pmid', ''),
                'topic': r.get('topic', ''),
                'score': hybrid_score,
                'hybrid_score': hybrid_score,
                'bm25_score': bm25_score,
                'vector_score': vector_score
            }
            hybrid_results.append(result)
        
        # 5. 按混合分数排序
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        final_results = hybrid_results[:top_k]
        
        logger.info(f"混合检索完成：BM25权重={1-alpha:.2f}, 向量权重={alpha:.2f}, 返回{len(final_results)}个结果")
        
        return final_results
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict:
        """根据ID获取chunk"""
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None


if __name__ == "__main__":
    # 测试混合检索
    print("=" * 70)
    print("🔍 混合检索模块测试")
    print("=" * 70)
    
    # 初始化
    searcher = HybridSearcher()
    
    # 测试查询
    test_query = "What are the symptoms of diabetes?"
    print(f"\n📝 测试查询: {test_query}")
    
    # 模拟向量检索结果
    mock_vector_results = [
        {'id': 100, 'distance': 0.2, 'text': 'diabetes symptoms...'},
        {'id': 200, 'distance': 0.3, 'text': 'type 2 diabetes...'},
        {'id': 300, 'distance': 0.4, 'text': 'hyperglycemia signs...'},
    ]
    
    # 混合检索
    results = searcher.hybrid_search(test_query, mock_vector_results, alpha=0.6, top_k=10)
    
    print(f"\n✅ 混合检索结果 (Top-10):")
    for i, r in enumerate(results[:5], 1):
        print(f"  [{i}] Hybrid={r['hybrid_score']:.3f} | BM25={r['bm25_score']:.3f} | Vector={r['vector_score']:.3f}")
        print(f"      Text: {r['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ 混合检索模块测试完成!")
    print("=" * 70)
