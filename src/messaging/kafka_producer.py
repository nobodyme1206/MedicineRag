# -*- coding: utf-8 -*-
"""
Kafka生产者 - 将爬取的文章发送到Kafka
实现数据采集与处理的解耦
"""

import json
import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("kafka_producer", LOGS_DIR / "kafka_producer.log")

# Kafka配置
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_RAW = "medical_raw_articles"      # 原始文章
KAFKA_TOPIC_PROCESSED = "medical_processed"    # 处理后的chunks
KAFKA_TOPIC_EMBEDDINGS = "medical_embeddings"  # 向量化请求


class KafkaArticleProducer:
    """Kafka文章生产者 - 将爬取的文章异步发送到Kafka"""
    
    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        """
        初始化Kafka生产者
        
        Args:
            bootstrap_servers: Kafka服务器地址
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self._init_producer()
        
        # 统计
        self.stats = {
            "sent": 0,
            "failed": 0,
            "bytes_sent": 0
        }
    
    def _init_producer(self):
        """初始化Kafka Producer"""
        try:
            from kafka import KafkaProducer
            from kafka.errors import NoBrokersAvailable
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # 性能优化配置
                batch_size=16384 * 4,      # 64KB批次
                linger_ms=10,               # 等待10ms凑批
                compression_type='gzip',    # 压缩
                acks='all',                 # 确保消息持久化
                retries=3,                  # 重试次数
                max_in_flight_requests_per_connection=5
            )
            logger.info(f"✅ Kafka Producer已连接: {self.bootstrap_servers}")
            
        except ImportError:
            logger.warning("⚠️ kafka-python未安装，使用模拟模式")
            self.producer = None
        except Exception as e:
            logger.warning(f"⚠️ Kafka连接失败: {e}，使用模拟模式")
            self.producer = None
    
    def send_article(self, article: Dict, topic: str = KAFKA_TOPIC_RAW) -> bool:
        """
        发送单篇文章到Kafka
        
        Args:
            article: 文章数据
            topic: Kafka topic
            
        Returns:
            是否发送成功
        """
        if not self.producer:
            # 模拟模式：直接返回成功
            self.stats["sent"] += 1
            return True
        
        try:
            # 使用PMID作为key，保证同一文章发送到同一partition
            key = article.get("pmid", "")
            
            future = self.producer.send(topic, key=key, value=article)
            # 异步发送，不等待确认（高吞吐）
            
            self.stats["sent"] += 1
            self.stats["bytes_sent"] += len(json.dumps(article, ensure_ascii=False).encode('utf-8'))
            
            return True
            
        except Exception as e:
            logger.error(f"发送失败: {e}")
            self.stats["failed"] += 1
            return False
    
    def send_batch(self, articles: List[Dict], topic: str = KAFKA_TOPIC_RAW) -> int:
        """
        批量发送文章
        
        Args:
            articles: 文章列表
            topic: Kafka topic
            
        Returns:
            成功发送的数量
        """
        success_count = 0
        
        for article in articles:
            if self.send_article(article, topic):
                success_count += 1
        
        # 确保所有消息都发送出去
        if self.producer:
            self.producer.flush()
        
        logger.info(f"批量发送完成: {success_count}/{len(articles)} 成功")
        return success_count
    
    def send_processing_request(self, chunk_data: Dict):
        """发送处理请求到处理队列"""
        return self.send_article(chunk_data, KAFKA_TOPIC_PROCESSED)
    
    def send_embedding_request(self, text_data: Dict):
        """发送向量化请求"""
        return self.send_article(text_data, KAFKA_TOPIC_EMBEDDINGS)
    
    def get_stats(self) -> Dict:
        """获取发送统计"""
        return {
            **self.stats,
            "bytes_sent_mb": round(self.stats["bytes_sent"] / (1024 * 1024), 2)
        }
    
    def close(self):
        """关闭生产者"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka Producer已关闭")


class KafkaTopicManager:
    """Kafka Topic管理器"""
    
    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = None
        self._init_admin()
    
    def _init_admin(self):
        """初始化Admin客户端"""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers
            )
            self.NewTopic = NewTopic
            logger.info("Kafka Admin已连接")
        except Exception as e:
            logger.warning(f"Kafka Admin连接失败: {e}")
            self.admin_client = None
    
    def create_topics(self):
        """创建所需的Topics"""
        if not self.admin_client:
            logger.warning("Admin客户端未初始化，跳过Topic创建")
            return
        
        topics = [
            self.NewTopic(
                name=KAFKA_TOPIC_RAW,
                num_partitions=6,        # 6个分区，支持6个消费者并行
                replication_factor=1     # 单节点部署
            ),
            self.NewTopic(
                name=KAFKA_TOPIC_PROCESSED,
                num_partitions=6,
                replication_factor=1
            ),
            self.NewTopic(
                name=KAFKA_TOPIC_EMBEDDINGS,
                num_partitions=3,        # 向量化是GPU密集型，分区少一些
                replication_factor=1
            )
        ]
        
        try:
            self.admin_client.create_topics(new_topics=topics, validate_only=False)
            logger.info(f"✅ Topics创建成功: {[t.name for t in topics]}")
        except Exception as e:
            if "TopicExistsException" in str(e) or "already exists" in str(e):
                logger.info("Topics已存在")
            else:
                logger.error(f"创建Topics失败: {e}")
    
    def list_topics(self) -> List[str]:
        """列出所有Topics"""
        if not self.admin_client:
            return []
        
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(bootstrap_servers=self.bootstrap_servers)
            topics = list(consumer.topics())
            consumer.close()
            return topics
        except Exception as e:
            logger.error(f"获取Topics失败: {e}")
            return []


def main():
    """测试Kafka生产者"""
    # 创建Topics
    manager = KafkaTopicManager()
    manager.create_topics()
    
    # 测试发送
    producer = KafkaArticleProducer()
    
    test_articles = [
        {
            "pmid": "12345678",
            "title": "Test Article 1",
            "abstract": "This is a test abstract for diabetes research.",
            "topic": "diabetes"
        },
        {
            "pmid": "12345679",
            "title": "Test Article 2",
            "abstract": "This is a test abstract for cancer research.",
            "topic": "cancer"
        }
    ]
    
    producer.send_batch(test_articles)
    print(f"发送统计: {producer.get_stats()}")
    
    producer.close()


if __name__ == "__main__":
    main()
