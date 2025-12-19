#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本向量化模块
提供TextEmbedder类，支持单文本和批量文本向量化
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIR, LOGS_DIR
from src.utils.logger import setup_logger
from src.utils.exceptions import EmbeddingError, handle_errors, retry

logger = setup_logger("embedder", LOGS_DIR / "embedding.log")

# 类型别名
Vector = NDArray[np.float32]
VectorBatch = NDArray[np.float32]


class TextEmbedder:
    """文本向量化器"""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ) -> None:
        """
        初始化向量化器
        
        Args:
            model_name: 模型名称，默认使用配置文件中的模型
            device: 设备 ("cuda" / "cpu")，默认自动检测
            cache_dir: 模型缓存目录
            
        Raises:
            EmbeddingError: 模型加载失败
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError("缺少依赖: pip install torch sentence-transformers", e)
        
        self.model_name: str = model_name or EMBEDDING_MODEL_NAME
        # 优先使用 HF_HOME 环境变量
        hf_home = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE'))
        self.cache_dir: Path = cache_dir or (Path(hf_home) if hf_home else EMBEDDING_MODEL_DIR)
        
        # 设备检测
        if device:
            self.device: str = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 加载模型
        logger.info(f"加载模型: {self.model_name}, 设备: {self.device}")
        
        try:
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=str(self.cache_dir), 
                device=self.device
            )
            self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成, 维度: {self.embedding_dim}")
        except Exception as e:
            raise EmbeddingError(f"模型加载失败: {self.model_name}", e)
    
    def encode_single(self, text: str) -> Vector:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            归一化的向量 (numpy array, shape: [dim])
            
        Raises:
            EmbeddingError: 编码失败
        """
        if not text or not text.strip():
            raise EmbeddingError("输入文本不能为空")
        
        try:
            embedding: Vector = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding
        except Exception as e:
            raise EmbeddingError(f"文本编码失败: {text[:50]}...", e)
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 64, 
        show_progress: bool = False
    ) -> VectorBatch:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            向量矩阵 (numpy array, shape: [N, dim])
            
        Raises:
            EmbeddingError: 编码失败
        """
        if not texts:
            raise EmbeddingError("输入文本列表不能为空")
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"过滤了 {len(texts) - len(valid_texts)} 个空文本")
        
        if not valid_texts:
            raise EmbeddingError("所有输入文本都为空")
        
        try:
            embeddings: VectorBatch = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"批量编码失败, 文本数: {len(valid_texts)}", e)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        **kwargs
    ) -> Union[Vector, VectorBatch]:
        """
        通用编码接口
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 传递给encode_batch的参数
            
        Returns:
            向量或向量矩阵
        """
        if isinstance(texts, str):
            return self.encode_single(texts)
        return self.encode_batch(texts, **kwargs)
    
    def get_embedding_dim(self) -> int:
        """获取向量维度"""
        return self.embedding_dim
    
    @handle_errors(default_return=0.0, log_level="warning")
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            余弦相似度 (0-1)
        """
        vec1 = self.encode_single(text1)
        vec2 = self.encode_single(text2)
        return float(np.dot(vec1, vec2))


def main() -> None:
    """测试向量化器"""
    logger.info("=" * 50)
    logger.info("测试向量化器")
    logger.info("=" * 50)
    
    try:
        embedder = TextEmbedder()
        
        # 单文本测试
        text = "What are the symptoms of diabetes?"
        emb = embedder.encode_single(text)
        logger.info(f"单文本向量: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        
        # 批量测试
        texts = [
            "diabetes symptoms", 
            "heart disease treatment", 
            "cancer prevention"
        ]
        embs = embedder.encode_batch(texts)
        logger.info(f"批量向量: shape={embs.shape}")
        
        # 相似度测试
        sim = embedder.compute_similarity(
            "diabetes treatment",
            "diabetic therapy options"
        )
        logger.info(f"相似度: {sim:.4f}")
        
        logger.info("✅ 测试通过")
        
    except EmbeddingError as e:
        logger.error(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()
