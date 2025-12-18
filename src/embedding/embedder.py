# -*- coding: utf-8 -*-
"""
文本向量化模块 - 兼容层
提供TextEmbedder类，封装unified_embedder的功能
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Union

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIR, LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("embedder", LOGS_DIR / "embedding.log")


class TextEmbedder:
    """文本向量化器 - 兼容接口"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化向量化器
        
        Args:
            model_name: 模型名称，默认使用配置文件中的模型
            device: 设备 ("cuda" / "cpu")，默认自动检测
        """
        import torch
        from sentence_transformers import SentenceTransformer
        
        # 设置离线模式
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        
        # 设备检测
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 加载模型
        cache_dir = str(EMBEDDING_MODEL_DIR)
        logger.info(f"加载模型: {self.model_name}, 设备: {self.device}")
        
        self.model = SentenceTransformer(
            self.model_name, 
            cache_folder=cache_dir, 
            device=self.device,
            local_files_only=True
        )
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"模型加载完成, 维度: {self.embedding_dim}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            向量 (numpy array)
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 64, 
                     show_progress: bool = False) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            向量矩阵 (N x dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
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


if __name__ == "__main__":
    # 测试
    embedder = TextEmbedder()
    
    # 单文本
    text = "What are the symptoms of diabetes?"
    emb = embedder.encode_single(text)
    print(f"单文本向量: shape={emb.shape}")
    
    # 批量
    texts = ["diabetes symptoms", "heart disease treatment", "cancer prevention"]
    embs = embedder.encode_batch(texts)
    print(f"批量向量: shape={embs.shape}")
