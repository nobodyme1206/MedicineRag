#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统评估模块
基于PubMedQA公开数据集，评估检索效果

评估内容:
- BM25基线
- 向量检索基线
- 混合RAG系统（Hybrid）

指标:
- Precision@K, Recall@K, F1@K
- MRR, MAP, NDCG@K
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from config.config import DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR
from src.utils.logger import setup_logger
from src.utils.exceptions import handle_errors

logger = setup_logger("rag_evaluator", LOGS_DIR / "rag_evaluation.log")

# 类型别名
SearchResult = Dict[str, Any]
Metrics = Dict[str, float]


@dataclass
class TestQuery:
    """测试查询"""
    id: str
    query: str
    relevant_doc_ids: List[str]
    relevance_grades: List[int]  # 0-3分级
    answer: Optional[str] = None
    source: str = "pubmedqa"


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        self.test_queries: List[TestQuery] = []
        self.cache_dir = DATA_DIR / "test_set"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    # ==================== 数据加载 ====================
    
    def load_pubmedqa(self, max_samples: int = 200) -> int:
        """
        加载PubMedQA数据集
        https://pubmedqa.github.io/
        """
        cache_file = self.cache_dir / "pubmedqa_test.json"
        
        if cache_file.exists():
            logger.info(f"从缓存加载PubMedQA: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.test_queries = [TestQuery(**item) for item in data[:max_samples]]
            return len(self.test_queries)
        
        logger.info("从HuggingFace下载PubMedQA...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
            
            queries = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                pubid = str(item.get("pubid", i))
                query = TestQuery(
                    id=f"pubmedqa_{pubid}",
                    query=item["question"],
                    relevant_doc_ids=[pubid],
                    relevance_grades=[3],
                    answer=item.get("long_answer", ""),
                    source="pubmedqa"
                )
                queries.append(query)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(q) for q in queries], f, ensure_ascii=False, indent=2)
            
            self.test_queries = queries
            logger.info(f"PubMedQA加载完成: {len(queries)} 条")
            return len(queries)
            
        except Exception as e:
            logger.warning(f"PubMedQA加载失败: {e}")
            return self._load_fallback_testset(max_samples)
    
    def _load_fallback_testset(self, max_samples: int = 100) -> int:
        """从语料库生成备用测试集"""
        parquet_path = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
        
        if not parquet_path.exists():
            logger.error("语料库不存在")
            return 0
        
        logger.info("从语料库生成测试集...")
        df = pd.read_parquet(parquet_path, columns=['id', 'title', 'topic'])
        
        queries = []
        for topic in df['topic'].unique():
            topic_docs = df[df['topic'] == topic]
            if len(topic_docs) < 10:
                continue
            
            sampled = topic_docs.sample(min(3, len(topic_docs)))
            for _, row in sampled.iterrows():
                title = row.get('title', '')
                if not title or len(str(title)) < 20:
                    continue
                
                relevant_ids = topic_docs['id'].tolist()[:10]
                queries.append(TestQuery(
                    id=f"synthetic_{row['id']}",
                    query=str(title),
                    relevant_doc_ids=relevant_ids,
                    relevance_grades=[2] * len(relevant_ids),
                    source="synthetic"
                ))
                
                if len(queries) >= max_samples:
                    break
            if len(queries) >= max_samples:
                break
        
        self.test_queries = queries
        logger.info(f"生成测试集: {len(queries)} 条")
        return len(queries)
    
    # ==================== 评估指标 ====================
    
    def _precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K"""
        top_k = set(retrieved[:k])
        return len(top_k & set(relevant)) / k if k > 0 else 0.0
    
    def _recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K"""
        if not relevant:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & set(relevant)) / len(relevant)
    
    def _mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """MRR"""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _ndcg_at_k(self, retrieved: List[str], relevance_map: Dict[str, int], k: int) -> float:
        """NDCG@K"""
        gains = [relevance_map.get(doc_id, 0) for doc_id in retrieved[:k]]
        dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0
    
    def _map_score(self, retrieved: List[str], relevant: List[str]) -> float:
        """MAP"""
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        precisions = []
        hit = 0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                hit += 1
                precisions.append(hit / (i + 1))
        return np.mean(precisions) if precisions else 0.0

    # ==================== 检索器评估 ====================
    
    def _evaluate_retriever(self, retriever_fn: Callable, name: str,
                            k_values: List[int] = [5, 10, 20]) -> Dict:
        """通用检索器评估"""
        if not self.test_queries:
            logger.error("测试集为空")
            return {"error": "测试集为空"}
        
        max_k = max(k_values)
        metrics = {f"P@{k}": [] for k in k_values}
        metrics.update({f"R@{k}": [] for k in k_values})
        metrics.update({f"NDCG@{k}": [] for k in k_values})
        metrics["MRR"] = []
        metrics["MAP"] = []
        metrics["latency_ms"] = []
        
        logger.info(f"\n评估 {name}，共 {len(self.test_queries)} 个查询...")
        
        for i, test in enumerate(self.test_queries):
            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i + 1}/{len(self.test_queries)}")
            
            try:
                start = time.time()
                results = retriever_fn(test.query, max_k)
                latency = (time.time() - start) * 1000
                
                # 提取检索结果的PMID（去掉chunk后缀，如 "12345_0" -> "12345"）
                retrieved_pmids = []
                for j, doc in enumerate(results):
                    pmid = doc.get("pmid") or doc.get("id") or doc.get("doc_id") or str(j)
                    pmid = str(pmid).split("_")[0]  # 去掉chunk后缀
                    retrieved_pmids.append(pmid)
                
                # 去重但保持顺序（同一PMID的多个chunk只算一次）
                seen = set()
                retrieved_ids = []
                for pmid in retrieved_pmids:
                    if pmid not in seen:
                        seen.add(pmid)
                        retrieved_ids.append(pmid)
                
                relevance_map = dict(zip(test.relevant_doc_ids, test.relevance_grades))
                
                for k in k_values:
                    metrics[f"P@{k}"].append(self._precision_at_k(retrieved_ids, test.relevant_doc_ids, k))
                    metrics[f"R@{k}"].append(self._recall_at_k(retrieved_ids, test.relevant_doc_ids, k))
                    metrics[f"NDCG@{k}"].append(self._ndcg_at_k(retrieved_ids, relevance_map, k))
                
                metrics["MRR"].append(self._mrr(retrieved_ids, test.relevant_doc_ids))
                metrics["MAP"].append(self._map_score(retrieved_ids, test.relevant_doc_ids))
                metrics["latency_ms"].append(latency)
                
            except Exception as e:
                logger.warning(f"查询失败 [{test.id}]: {e}")
        
        result = {"name": name, "num_queries": len(self.test_queries), "metrics": {}}
        for key, values in metrics.items():
            if values:
                result["metrics"][key] = round(np.mean(values), 4)
                result["metrics"][f"{key}_std"] = round(np.std(values), 4)
        
        for k in k_values:
            p = result["metrics"].get(f"P@{k}", 0)
            r = result["metrics"].get(f"R@{k}", 0)
            result["metrics"][f"F1@{k}"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0
        
        return result
    
    def evaluate_bm25(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """评估BM25基线"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 BM25 基线评估")
        logger.info("=" * 50)
        
        try:
            from src.retrieval.hybrid_searcher import HybridSearcher
            searcher = HybridSearcher()
            
            def bm25_retriever(query, k):
                # BM25返回 [(chunk_idx, score), ...]
                results = searcher.bm25_search(query, top_k=k)
                # 转换为包含pmid的字典格式
                docs = []
                for idx, score in results:
                    if idx < len(searcher.chunks):
                        chunk = searcher.chunks[idx]
                        pmid = chunk.get('pmid', str(idx))
                        docs.append({"pmid": pmid, "score": score})
                return docs
            
            return self._evaluate_retriever(
                retriever_fn=bm25_retriever,
                name="BM25", k_values=k_values
            )
        except Exception as e:
            logger.error(f"BM25评估失败: {e}")
            return {"name": "BM25", "error": str(e)}
    
    def evaluate_vector(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """评估向量检索基线"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 Vector 基线评估")
        logger.info("=" * 50)
        
        try:
            from src.rag.rag_system import RAGSystem
            rag = RAGSystem()
            return self._evaluate_retriever(
                retriever_fn=lambda q, k: rag.vector_search(q, top_k=k),
                name="Vector", k_values=k_values
            )
        except Exception as e:
            logger.error(f"Vector评估失败: {e}")
            return {"name": "Vector", "error": str(e)}
    
    def evaluate_hybrid(self, k_values: List[int] = [5, 10, 20], use_hyde: bool = False) -> Dict:
        """评估混合RAG系统"""
        logger.info("\n" + "=" * 50)
        logger.info(f"📊 Hybrid RAG 评估 (HyDE: {use_hyde})")
        logger.info("=" * 50)
        
        try:
            from src.rag.rag_system import RAGSystem
            # 关闭HyDE加速评估，保留混合检索和Rerank
            rag = RAGSystem(use_hyde=use_hyde)
            return self._evaluate_retriever(
                retriever_fn=lambda q, k: rag.retrieve(q, top_k=k),
                name="Hybrid_RAG", k_values=k_values
            )
        except Exception as e:
            logger.error(f"Hybrid评估失败: {e}")
            return {"name": "Hybrid_RAG", "error": str(e)}
    
    # ==================== 完整评估 ====================
    
    def run_evaluation(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """运行完整RAG评估"""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 RAG系统评估 (BM25 / Vector / Hybrid)")
        logger.info("=" * 60)
        
        if not self.test_queries:
            self.load_pubmedqa()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {"name": "PubMedQA", "num_queries": len(self.test_queries)},
            "methods": {}
        }
        
        self.results["methods"]["BM25"] = self.evaluate_bm25(k_values)
        self.results["methods"]["Vector"] = self.evaluate_vector(k_values)
        self.results["methods"]["Hybrid_RAG"] = self.evaluate_hybrid(k_values)
        
        self._print_comparison(k_values)
        self._save_results()
        
        return self.results
    
    def _print_comparison(self, k_values: List[int]):
        """打印对比表格"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 RAG评估结果对比")
        logger.info("=" * 80)
        
        header = f"{'方法':<15}"
        for k in k_values:
            header += f"{'P@'+str(k):<8}{'R@'+str(k):<8}{'F1@'+str(k):<8}"
        header += f"{'MRR':<8}{'MAP':<8}{'延迟ms':<10}"
        logger.info(header)
        logger.info("-" * 80)
        
        for name, data in self.results["methods"].items():
            if "error" in data:
                logger.info(f"{name:<15} 评估失败: {data['error']}")
                continue
            m = data.get("metrics", {})
            row = f"{name:<15}"
            for k in k_values:
                row += f"{m.get(f'P@{k}', 0):<8.3f}{m.get(f'R@{k}', 0):<8.3f}{m.get(f'F1@{k}', 0):<8.3f}"
            row += f"{m.get('MRR', 0):<8.3f}{m.get('MAP', 0):<8.3f}{m.get('latency_ms', 0):<10.1f}"
            logger.info(row)
        
        logger.info("=" * 80)
    
    def _save_results(self):
        """保存结果"""
        output_file = RESULTS_DIR / "evaluation" / "rag_evaluation.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存: {output_file}")


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG系统评估")
    parser.add_argument("--samples", type=int, default=200, help="测试样本数")
    args = parser.parse_args()
    
    evaluator = RAGEvaluator()
    evaluator.load_pubmedqa(args.samples)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
