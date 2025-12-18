# -*- coding: utf-8 -*-
"""
Milvus向量数据库管理
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("milvus_manager", LOGS_DIR / "milvus.log")


class MilvusManager:
    """Milvus向量数据库管理器"""
    
    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        """
        初始化Milvus连接
        
        Args:
            host: Milvus服务地址
            port: Milvus服务端口
        """
        logger.info(f"连接Milvus: {host}:{port}")
        
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            logger.info("✅ Milvus连接成功")
        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            logger.info("请确保Milvus服务已启动 (docker-compose up -d)")
            raise
        
        self.collection = None
    
    def create_collection(self, collection_name: str = MILVUS_COLLECTION_NAME,
                         dimension: int = EMBEDDING_DIMENSION):
        """
        创建集合
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
        """
        # 如果集合已存在，删除它
        if utility.has_collection(collection_name):
            logger.warning(f"集合 {collection_name} 已存在，将删除重建")
            utility.drop_collection(collection_name)
        
        # 定义字段schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Medical Knowledge Base"
        )
        
        # 创建集合
        self.collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        logger.info(f"✅ 集合 {collection_name} 创建成功")
        logger.info(f"   向量维度: {dimension}")
    
    def create_index(self):
        """创建索引"""
        if not self.collection:
            logger.error("请先创建集合")
            return
        
        logger.info("创建索引...")
        
        index_params = {
            "index_type": MILVUS_INDEX_TYPE,
            "metric_type": MILVUS_METRIC_TYPE,
            "params": {"nlist": MILVUS_NLIST}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"✅ 索引创建成功")
        logger.info(f"   类型: {MILVUS_INDEX_TYPE}")
        logger.info(f"   度量: {MILVUS_METRIC_TYPE}")
    
    def insert_vectors(self, embeddings: np.ndarray, metadata: List[Dict],
                      batch_size: int = 1000):
        """
        插入向量
        
        Args:
            embeddings: 向量数组
            metadata: 元数据列表
            batch_size: 批量插入大小
        """
        if not self.collection:
            logger.error("请先创建集合")
            return
        
        total = len(embeddings)
        logger.info(f"开始插入 {total:,} 条向量，批量大小: {batch_size}")
        
        inserted = 0
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_embeddings = embeddings[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            
            # 准备数据
            entities = [
                [m.get("pmid", "") for m in batch_metadata],
                [m.get("chunk_text", "")[:2000] for m in batch_metadata],  # 截断到最大长度
                batch_embeddings.tolist()
            ]
            
            try:
                self.collection.insert(entities)
                inserted += len(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  已插入 {inserted:,}/{total:,} ({inserted/total*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"插入批次失败: {e}")
                continue
        
        # 刷新确保数据持久化
        self.collection.flush()
        logger.info(f"✅ 共插入 {inserted:,} 条向量")
        
        return inserted
    
    def load_collection(self, collection_name: str = MILVUS_COLLECTION_NAME):
        """加载集合到内存"""
        if not self.collection:
            self.collection = Collection(collection_name)
        
        self.collection.load()
        logger.info(f"✅ 集合 {collection_name} 已加载到内存")
    
    def search(self, query_vectors: np.ndarray, top_k: int = RETRIEVAL_TOP_K) -> List[List[Dict]]:
        """
        向量搜索
        
        Args:
            query_vectors: 查询向量
            top_k: 返回Top-K结果
            
        Returns:
            搜索结果列表
        """
        if not self.collection:
            logger.error("请先加载集合")
            return []
        
        search_params = {
            "metric_type": MILVUS_METRIC_TYPE,
            "params": {"nprobe": MILVUS_NPROBE}
        }
        
        results = self.collection.search(
            data=query_vectors.tolist() if isinstance(query_vectors, np.ndarray) else query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["pmid", "chunk_text"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            batch_results = []
            for hit in hits:
                batch_results.append({
                    "id": hit.id,
                    "pmid": hit.entity.get("pmid"),
                    "text": hit.entity.get("chunk_text"),
                    "score": hit.score
                })
            formatted_results.append(batch_results)
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.collection:
            return {}
        
        stats = {
            "name": self.collection.name,
            "num_entities": self.collection.num_entities,
            "description": self.collection.description
        }
        
        return stats


def rebuild_database(resume: bool = False, batch_size: int = 128):
    """
    重建向量数据库（支持断点续传）
    
    Args:
        resume: 是否断点续传
        batch_size: 批次大小
    """
    import time
    import pandas as pd
    
    logger.info("="*60)
    logger.info("🔄 重建向量数据库")
    logger.info(f"   模式: {'断点续传' if resume else '从头开始'}")
    logger.info(f"   批次大小: {batch_size}")
    logger.info("="*60)
    
    # 优先使用 Parquet 文件
    parquet_file = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
    chunks_file = PROCESSED_DATA_DIR / "medical_chunks.json"
    
    if parquet_file.exists():
        logger.info(f"加载Parquet数据: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        chunks = df.to_dict('records')
        logger.info(f"总chunks: {len(chunks):,}")
    elif chunks_file.exists():
        logger.info(f"加载JSON数据: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"总chunks: {len(chunks):,}")
    else:
        logger.error(f"数据文件不存在: {parquet_file} 或 {chunks_file}")
        return
    
    # 初始化
    from src.embedding.embedder import TextEmbedder
    embedder = TextEmbedder()
    milvus = MilvusManager()
    
    # 检查断点续传
    start_index = 0
    if resume:
        try:
            milvus.load_collection()
            start_index = milvus.collection.num_entities
            logger.info(f"检测到已有 {start_index:,} 个向量，从第 {start_index+1} 条继续")
        except:
            resume = False
    
    if not resume or start_index == 0:
        milvus.create_collection()
        start_index = 0
    
    chunks_to_process = chunks[start_index:]
    if len(chunks_to_process) == 0:
        logger.info(f"所有 {len(chunks):,} 个chunks已完成")
        return
    
    # 向量化和入库
    logger.info(f"待处理: {len(chunks_to_process):,} 个chunks")
    
    start_time = time.time()
    total_inserted = start_index
    buffer_embeddings = []
    buffer_metadata = []
    insert_interval = 50
    total_batches = (len(chunks_to_process) + batch_size - 1) // batch_size
    
    # 断点保存间隔
    checkpoint_file = EMBEDDING_DATA_DIR / "rebuild_checkpoint.json"
    save_checkpoint_interval = 10000  # 每1万条保存一次断点
    
    for i in range(0, len(chunks_to_process), batch_size):
        batch = chunks_to_process[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        texts = [c.get('chunk_text') or c.get('content', '') for c in batch]
        embeddings = embedder.encode_batch(texts)
        
        metadata_batch = [{'pmid': str(c.get('pmid', '')), 'chunk_text': t[:2000]} for c, t in zip(batch, texts)]
        
        buffer_embeddings.append(embeddings)
        buffer_metadata.extend(metadata_batch)
        
        if batch_num % insert_interval == 0 or batch_num == total_batches:
            batch_vectors = np.vstack(buffer_embeddings)
            milvus.insert_vectors(batch_vectors, buffer_metadata, batch_size=5000)
            total_inserted += len(batch_vectors)
            buffer_embeddings = []
            buffer_metadata = []
            
            progress = total_inserted / len(chunks) * 100
            elapsed = time.time() - start_time
            speed = (total_inserted - start_index) / elapsed if elapsed > 0 else 0
            eta_seconds = (len(chunks) - total_inserted) / speed if speed > 0 else 0
            eta_minutes = eta_seconds / 60
            
            logger.info(f"进度: {progress:.1f}% | 已入库: {total_inserted:,}/{len(chunks):,} | "
                       f"速度: {speed:.0f}条/秒 | 预计剩余: {eta_minutes:.1f}分钟")
            
            # 保存断点
            if total_inserted % save_checkpoint_interval < batch_size * insert_interval:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': total_inserted, 'total': len(chunks)}, f)
                logger.info(f"  💾 断点已保存: {total_inserted:,}")
    
    milvus.create_index()
    milvus.load_collection()
    
    total_time = time.time() - start_time
    logger.info(f"✅ 重建完成! 向量数: {total_inserted:,}, 耗时: {total_time/60:.1f}分钟")


def main():
    """主函数 - 构建向量数据库"""
    logger.info("="*50)
    logger.info("构建Milvus向量数据库")
    logger.info("="*50)
    
    # 1. 连接Milvus
    manager = MilvusManager()
    
    # 2. 创建集合
    manager.create_collection()
    
    # 3. 读取向量数据
    embedding_file = EMBEDDING_DATA_DIR / "medical_embeddings.npy"
    mapping_file = EMBEDDING_DATA_DIR / "medical_embeddings.mapping.json"
    
    if not embedding_file.exists():
        logger.error(f"向量文件不存在: {embedding_file}")
        logger.info("请先运行 embedder.py 生成向量")
        return
    
    logger.info(f"读取向量: {embedding_file}")
    embeddings = np.load(embedding_file)
    logger.info(f"  向量数量: {len(embeddings):,}")
    logger.info(f"  向量维度: {embeddings.shape[1]}")
    
    logger.info(f"读取元数据: {mapping_file}")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info(f"  元数据数量: {len(metadata):,}")
    
    # 4. 插入向量
    manager.insert_vectors(embeddings, metadata)
    
    # 5. 创建索引
    manager.create_index()
    
    # 6. 加载集合
    manager.load_collection()
    
    # 7. 统计信息
    stats = manager.get_stats()
    logger.info("\n向量数据库统计:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n✅ 向量数据库构建完成！")


if __name__ == "__main__":
    main()
