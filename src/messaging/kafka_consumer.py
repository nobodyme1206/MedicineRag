# -*- coding: utf-8 -*-
"""
Kafka消费者 - 从Kafka消费文章并处理
支持多消费者并行处理，实现高吞吐量
"""

import json
import time
import threading
from typing import Dict, List, Callable, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("kafka_consumer", LOGS_DIR / "kafka_consumer.log")

# Kafka配置
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_RAW = "medical_raw_articles"
KAFKA_TOPIC_PROCESSED = "medical_processed"
KAFKA_TOPIC_EMBEDDINGS = "medical_embeddings"
KAFKA_CONSUMER_GROUP = "medical_rag_processors"


class KafkaArticleConsumer:
    """Kafka文章消费者 - 消费并处理文章"""
    
    def __init__(self, 
                 topic: str = KAFKA_TOPIC_RAW,
                 group_id: str = KAFKA_CONSUMER_GROUP,
                 bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
                 auto_offset_reset: str = 'earliest'):
        """
        初始化Kafka消费者
        
        Args:
            topic: 订阅的topic
            group_id: 消费者组ID（同组消费者自动负载均衡）
            bootstrap_servers: Kafka服务器地址
            auto_offset_reset: 偏移量重置策略
        """
        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.running = False
        
        # 统计
        self.stats = {
            "consumed": 0,
            "processed": 0,
            "failed": 0,
            "bytes_consumed": 0
        }
        
        # 批处理缓冲
        self.buffer: List[Dict] = []
        self.buffer_size = 100  # 每100条处理一次
        self.buffer_lock = threading.Lock()
        
        self._init_consumer(auto_offset_reset)
    
    def _init_consumer(self, auto_offset_reset: str):
        """初始化Kafka Consumer"""
        try:
            from kafka import KafkaConsumer
            
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                # 性能优化
                fetch_min_bytes=1024 * 100,    # 至少100KB才返回
                fetch_max_wait_ms=500,          # 最多等待500ms
                max_poll_records=500,           # 每次最多拉取500条
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"✅ Kafka Consumer已连接: {self.topic} (group: {self.group_id})")
            
        except ImportError:
            logger.warning("⚠️ kafka-python未安装")
            self.consumer = None
        except Exception as e:
            logger.warning(f"⚠️ Kafka连接失败: {e}")
            self.consumer = None
    
    def consume_and_process(self, 
                           processor: Callable[[List[Dict]], None],
                           batch_size: int = 100,
                           timeout_ms: int = 1000):
        """
        消费消息并批量处理
        
        Args:
            processor: 处理函数，接收文章列表
            batch_size: 批处理大小
            timeout_ms: 拉取超时时间
        """
        if not self.consumer:
            logger.error("Consumer未初始化")
            return
        
        self.running = True
        self.buffer_size = batch_size
        
        logger.info(f"🚀 开始消费 {self.topic}，批次大小: {batch_size}")
        
        try:
            while self.running:
                # 拉取消息
                messages = self.consumer.poll(timeout_ms=timeout_ms)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        article = record.value
                        self.stats["consumed"] += 1
                        
                        with self.buffer_lock:
                            self.buffer.append(article)
                            
                            # 达到批次大小，触发处理
                            if len(self.buffer) >= self.buffer_size:
                                self._process_buffer(processor)
                
                # 定期处理剩余数据
                if self.buffer and len(self.buffer) > 0:
                    with self.buffer_lock:
                        if len(self.buffer) > 0:
                            self._process_buffer(processor)
                            
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self._flush_buffer(processor)
            self.stop()
    
    def _process_buffer(self, processor: Callable[[List[Dict]], None]):
        """处理缓冲区中的数据"""
        if not self.buffer:
            return
        
        batch = self.buffer.copy()
        self.buffer.clear()
        
        try:
            start_time = time.time()
            processor(batch)
            elapsed = time.time() - start_time
            
            self.stats["processed"] += len(batch)
            logger.info(f"✅ 处理批次: {len(batch)} 条, 耗时: {elapsed:.2f}s, "
                       f"速率: {len(batch)/elapsed:.1f} 条/秒")
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            self.stats["failed"] += len(batch)
    
    def _flush_buffer(self, processor: Callable[[List[Dict]], None]):
        """刷新剩余缓冲区"""
        with self.buffer_lock:
            if self.buffer:
                self._process_buffer(processor)
    
    def stop(self):
        """停止消费"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka Consumer已关闭")
    
    def get_stats(self) -> Dict:
        """获取消费统计"""
        return self.stats


class DataProcessingConsumer(KafkaArticleConsumer):
    """数据处理消费者 - 消费原始文章，处理后发送到下一个Topic"""
    
    def __init__(self):
        super().__init__(
            topic=KAFKA_TOPIC_RAW,
            group_id="data_processors"
        )
        self.output_producer = None
        self._init_output_producer()
    
    def _init_output_producer(self):
        """初始化输出生产者"""
        from src.messaging.kafka_producer import KafkaArticleProducer
        self.output_producer = KafkaArticleProducer()
    
    def process_articles(self, articles: List[Dict]):
        """
        处理文章：清洗 + 切分 + 发送到下游
        """
        from src.data_processing.data_processor import DataProcessor
        
        # 简化处理：直接切分文本
        chunks = []
        for article in articles:
            text = article.get('full_text', '')
            if len(text) < 100:
                continue
            
            # 简单切分
            chunk_size = 512
            overlap = 50
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                
                if len(chunk_text) >= 100:
                    chunk = {
                        "pmid": article.get("pmid"),
                        "chunk_id": f"{article.get('pmid')}_{chunk_id}",
                        "chunk_text": chunk_text,
                        "title": article.get("title", ""),
                        "topic": article.get("topic", "")
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                
                start += (chunk_size - overlap)
        
        # 发送到处理后的Topic
        if self.output_producer and chunks:
            for chunk in chunks:
                self.output_producer.send_processing_request(chunk)
            self.output_producer.producer.flush() if self.output_producer.producer else None
        
        logger.info(f"处理完成: {len(articles)} 文章 → {len(chunks)} chunks")
    
    def start(self):
        """启动处理"""
        self.consume_and_process(
            processor=self.process_articles,
            batch_size=100
        )


class EmbeddingConsumer(KafkaArticleConsumer):
    """向量化消费者 - 消费处理后的chunks，生成向量"""
    
    def __init__(self):
        super().__init__(
            topic=KAFKA_TOPIC_PROCESSED,
            group_id="embedding_workers"
        )
        self.embedder = None
        self.milvus_manager = None
    
    def _init_embedder(self):
        """延迟初始化Embedder（GPU资源）"""
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
            logger.info("Embedder已初始化")
    
    def _init_milvus(self):
        """延迟初始化Milvus"""
        if self.milvus_manager is None:
            from src.retrieval.milvus_manager import MilvusManager
            self.milvus_manager = MilvusManager()
            logger.info("Milvus已连接")
    
    def process_chunks(self, chunks: List[Dict]):
        """
        处理chunks：向量化 + 入库
        """
        self._init_embedder()
        self._init_milvus()
        
        # 提取文本
        texts = [c.get("chunk_text", "") for c in chunks]
        
        # 批量向量化
        embeddings = self.embedder.encode_batch(texts, show_progress=False)
        
        # 准备入库数据
        entities = []
        for i, chunk in enumerate(chunks):
            entities.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "pmid": chunk.get("pmid", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "embedding": embeddings[i].tolist()
            })
        
        # 批量插入Milvus
        if self.milvus_manager:
            try:
                self.milvus_manager.insert_batch(entities)
                logger.info(f"向量入库: {len(entities)} 条")
            except Exception as e:
                logger.error(f"Milvus插入失败: {e}")
    
    def start(self):
        """启动向量化处理"""
        self.consume_and_process(
            processor=self.process_chunks,
            batch_size=64  # GPU批次小一些
        )


def run_data_processor():
    """运行数据处理消费者"""
    consumer = DataProcessingConsumer()
    consumer.start()


def run_embedding_worker():
    """运行向量化消费者"""
    consumer = EmbeddingConsumer()
    consumer.start()


def main():
    """测试消费者"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["processor", "embedder"], default="processor")
    args = parser.parse_args()
    
    if args.type == "processor":
        run_data_processor()
    else:
        run_embedding_worker()


if __name__ == "__main__":
    main()
