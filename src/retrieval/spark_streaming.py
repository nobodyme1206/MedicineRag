# -*- coding: utf-8 -*-
"""
方案C: Spark Streaming增量索引
实时监听新数据并自动更新Milvus向量索引
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import threading

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("spark_streaming", LOGS_DIR / "spark_streaming.log")


class IncrementalIndexer:
    """增量索引管理器 - 基于Spark Structured Streaming"""
    
    def __init__(self, watch_dir: Path = None, use_spark: bool = True):
        """
        初始化增量索引器
        
        Args:
            watch_dir: 监听的数据目录
            use_spark: 是否使用Spark Streaming（False则用简单文件监听）
        """
        self.watch_dir = watch_dir or RAW_DATA_DIR
        self.use_spark = use_spark
        self.spark = None
        self.milvus = None
        self.embedder = None
        self.is_running = False
        self.processed_files = set()
        self.stats = {"files_processed": 0, "vectors_added": 0, "errors": 0}
        
        logger.info(f"📡 增量索引器初始化")
        logger.info(f"   监听目录: {self.watch_dir}")
        logger.info(f"   使用Spark: {use_spark}")
    
    def _init_components(self):
        """延迟初始化组件"""
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        
        if self.milvus is None:
            from src.retrieval.milvus_manager import MilvusManager
            self.milvus = MilvusManager()
            try:
                self.milvus.load_collection()
            except:
                self.milvus.create_collection()

    def _init_spark_streaming(self):
        """初始化Spark Structured Streaming"""
        if self.spark is not None:
            return
        
        from pyspark.sql import SparkSession
        
        self.spark = SparkSession.builder \
            .appName("MedicalRAG-IncrementalIndex") \
            .master("local[2]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.streaming.schemaInference", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("✅ Spark Streaming初始化完成")
    
    def process_new_file(self, file_path: Path) -> Dict:
        """处理单个新文件并更新索引"""
        logger.info(f"📄 处理新文件: {file_path.name}")
        
        self._init_components()
        
        try:
            # 读取数据
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            if not articles:
                return {"status": "empty", "count": 0}
            
            # 提取文本
            texts = []
            metadata = []
            for article in articles:
                text = article.get('full_text') or article.get('abstract', '')
                if text and len(text) > 100:
                    texts.append(text[:2000])  # 截断
                    metadata.append({
                        'pmid': str(article.get('pmid', '')),
                        'chunk_text': text[:500]
                    })
            
            if not texts:
                return {"status": "no_valid_text", "count": 0}
            
            # 向量化
            logger.info(f"   向量化 {len(texts)} 条文本...")
            embeddings = self.embedder.encode_batch(texts, batch_size=64)
            
            # 插入Milvus
            logger.info(f"   插入Milvus...")
            self.milvus.insert_vectors(embeddings, metadata)
            
            self.stats["files_processed"] += 1
            self.stats["vectors_added"] += len(texts)
            
            logger.info(f"✅ 完成: 新增 {len(texts)} 个向量")
            
            return {"status": "success", "count": len(texts)}
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"❌ 处理失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def start_file_watcher(self, interval: int = 30):
        """启动文件监听（简单模式）"""
        logger.info(f"👀 启动文件监听，间隔: {interval}秒")
        self.is_running = True
        
        def watch_loop():
            while self.is_running:
                try:
                    # 扫描新文件
                    for file_path in self.watch_dir.glob("*.json"):
                        if file_path.name not in self.processed_files:
                            self.process_new_file(file_path)
                            self.processed_files.add(file_path.name)
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"监听错误: {e}")
                    time.sleep(interval)
        
        self.watch_thread = threading.Thread(target=watch_loop, daemon=True)
        self.watch_thread.start()
        logger.info("✅ 文件监听已启动")
    
    def start_spark_streaming(self):
        """启动Spark Structured Streaming"""
        self._init_spark_streaming()
        self._init_components()
        
        logger.info("🚀 启动Spark Structured Streaming...")
        
        # 创建流式读取
        schema = "pmid STRING, title STRING, abstract STRING, full_text STRING, topic STRING"
        
        stream_df = self.spark.readStream \
            .format("json") \
            .schema(schema) \
            .option("maxFilesPerTrigger", 1) \
            .load(str(self.watch_dir))
        
        # 处理函数
        def process_batch(batch_df, batch_id):
            if batch_df.count() == 0:
                return
            
            logger.info(f"📦 处理批次 {batch_id}: {batch_df.count()} 条")
            
            # 转为Pandas处理
            pdf = batch_df.toPandas()
            texts = pdf['full_text'].fillna(pdf['abstract']).tolist()
            texts = [t[:2000] for t in texts if t and len(str(t)) > 100]
            
            if texts:
                embeddings = self.embedder.encode_batch(texts, batch_size=64)
                metadata = [{'pmid': str(row['pmid']), 'chunk_text': texts[i][:500]} 
                           for i, row in pdf.iterrows() if i < len(texts)]
                self.milvus.insert_vectors(embeddings, metadata)
                self.stats["vectors_added"] += len(texts)
            
            logger.info(f"✅ 批次 {batch_id} 完成: {len(texts)} 向量")
        
        # 启动流
        query = stream_df.writeStream \
            .foreachBatch(process_batch) \
            .trigger(processingTime="30 seconds") \
            .start()
        
        self.is_running = True
        logger.info("✅ Spark Streaming已启动")
        
        return query
    
    def stop(self):
        """停止增量索引"""
        self.is_running = False
        if self.spark:
            self.spark.stop()
        logger.info("🛑 增量索引已停止")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "watch_dir": str(self.watch_dir),
            "is_running": self.is_running
        }


class BatchIncrementalUpdater:
    """批量增量更新器 - 用于定期批量更新索引"""
    
    def __init__(self):
        self.embedder = None
        self.milvus = None
    
    def update_from_new_data(self, data_file: Path, batch_size: int = 1000) -> Dict:
        """从新数据文件批量更新索引"""
        logger.info(f"📊 批量增量更新: {data_file}")
        
        # 初始化
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
        
        if self.milvus is None:
            from src.retrieval.milvus_manager import MilvusManager
            self.milvus = MilvusManager()
            self.milvus.load_collection()
        
        # 读取数据
        import pandas as pd
        if data_file.suffix == '.parquet':
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_json(data_file)
        
        logger.info(f"   数据量: {len(df):,} 条")
        
        # 获取当前索引数量
        current_count = self.milvus.collection.num_entities
        logger.info(f"   当前索引: {current_count:,} 向量")
        
        # 批量处理
        text_col = 'content' if 'content' in df.columns else 'chunk_text'
        total_added = 0
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            texts = batch_df[text_col].tolist()
            
            # 向量化
            embeddings = self.embedder.encode_batch(texts, batch_size=128)
            
            # 准备元数据
            metadata = []
            for _, row in batch_df.iterrows():
                metadata.append({
                    'pmid': str(row.get('pmid', '')),
                    'chunk_text': str(row.get(text_col, ''))[:500]
                })
            
            # 插入
            self.milvus.insert_vectors(embeddings, metadata)
            total_added += len(texts)
            
            logger.info(f"   进度: {total_added:,}/{len(df):,}")
        
        # 重建索引
        self.milvus.create_index()
        self.milvus.load_collection()
        
        new_count = self.milvus.collection.num_entities
        
        result = {
            "previous_count": current_count,
            "added": total_added,
            "new_count": new_count,
            "status": "success"
        }
        
        logger.info(f"✅ 增量更新完成: {current_count:,} -> {new_count:,} (+{total_added:,})")
        
        return result


def main():
    """测试入口"""
    print("=" * 60)
    print("🚀 Spark Streaming增量索引测试")
    print("=" * 60)
    
    # 简单文件监听模式
    indexer = IncrementalIndexer(use_spark=False)
    indexer.start_file_watcher(interval=10)
    
    print("\n监听中... 按Ctrl+C停止")
    try:
        while True:
            time.sleep(5)
            stats = indexer.get_stats()
            print(f"统计: 文件={stats['files_processed']}, 向量={stats['vectors_added']}")
    except KeyboardInterrupt:
        indexer.stop()
        print("\n已停止")


if __name__ == "__main__":
    main()
