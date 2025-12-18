# -*- coding: utf-8 -*-
"""
Rerank重排序模块
使用交叉编码器对检索结果进行精细排序
"""

import torch
from typing import List, Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("reranker", LOGS_DIR / "reranker.log")


class Reranker:
    """文档重排序器"""
    
    def __init__(self, model_name: str = None):
        """
        初始化重排序器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name or RERANK_MODEL_NAME
        self.model = None
        self.device = EMBEDDING_DEVICE
        
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        logger.info(f"加载Rerank模型: {self.model_name}")
        
        try:
            from FlagEmbedding import FlagReranker
            
            self.model = FlagReranker(
                self.model_name,
                use_fp16=True,  # 使用半精度加速
                device=self.device
            )
            
            logger.info(f"✅ Rerank模型加载成功")
            logger.info(f"   设备: {self.device}")
            
        except ImportError:
            logger.warning("FlagEmbedding未安装，尝试使用sentence-transformers")
            self._load_fallback_model()
        except Exception as e:
            logger.error(f"加载Rerank模型失败: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """加载备用模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            # 使用较小的交叉编码器作为备用
            fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"使用备用模型: {fallback_model}")
            
            self.model = CrossEncoder(fallback_model, device=self.device)
            self.model_type = "cross_encoder"
            
            logger.info("✅ 备用Rerank模型加载成功")
            
        except Exception as e:
            logger.error(f"备用模型也加载失败: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表，每个文档包含 'text' 字段
            top_k: 返回前K个结果
            
        Returns:
            重排序后的文档列表
        """
        if not self.model or not documents:
            return documents[:top_k] if top_k else documents
        
        top_k = top_k or RERANK_TOP_K
        
        try:
            # 提取文档文本
            doc_texts = [doc.get('text', '') for doc in documents]
            
            # 构建查询-文档对
            pairs = [[query, text] for text in doc_texts]
            
            # 计算相关性分数
            if hasattr(self.model, 'compute_score'):
                # FlagReranker
                scores = self.model.compute_score(pairs, normalize=True)
            else:
                # CrossEncoder
                scores = self.model.predict(pairs)
            
            # 如果scores是单个值，转为列表
            if not isinstance(scores, (list, tuple)):
                scores = [scores]
            
            # 添加重排序分数到文档
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i]) if i < len(scores) else 0.0
            
            # 按重排序分数排序
            reranked = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            logger.debug(f"重排序完成: {len(documents)} -> {top_k} 文档")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]
    
    def batch_rerank(self, queries: List[str], documents_list: List[List[Dict]], 
                     top_k: int = None) -> List[List[Dict]]:
        """
        批量重排序
        
        Args:
            queries: 查询列表
            documents_list: 每个查询对应的文档列表
            top_k: 每个查询返回的文档数
            
        Returns:
            重排序后的文档列表的列表
        """
        results = []
        for query, docs in zip(queries, documents_list):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        return results


def main():
    """测试Reranker"""
    logger.info("测试Reranker模块")
    
    reranker = Reranker()
    
    # 测试数据
    query = "What are the symptoms of diabetes?"
    documents = [
        {"text": "Diabetes is a chronic disease that affects blood sugar levels.", "pmid": "1", "score": 0.8},
        {"text": "The weather forecast shows sunny skies tomorrow.", "pmid": "2", "score": 0.85},
        {"text": "Common symptoms of diabetes include increased thirst, frequent urination, and fatigue.", "pmid": "3", "score": 0.75},
        {"text": "Cardiovascular disease is a leading cause of death.", "pmid": "4", "score": 0.7},
        {"text": "Type 2 diabetes symptoms may develop slowly over several years.", "pmid": "5", "score": 0.65},
    ]
    
    logger.info(f"原始排序:")
    for i, doc in enumerate(documents):
        logger.info(f"  {i+1}. score={doc['score']:.2f}: {doc['text'][:50]}...")
    
    reranked = reranker.rerank(query, documents, top_k=3)
    
    logger.info(f"\n重排序后:")
    for i, doc in enumerate(reranked):
        logger.info(f"  {i+1}. rerank_score={doc.get('rerank_score', 0):.4f}: {doc['text'][:50]}...")


if __name__ == "__main__":
    main()
