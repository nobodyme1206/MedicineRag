# -*- coding: utf-8 -*-
"""
Kafka集成版爬虫 - 将原有爬虫与Kafka集成
爬取的文章实时发送到Kafka，实现采集与处理解耦
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import RAW_DATA_DIR, LOGS_DIR
from src.utils.logger import setup_logger
from src.messaging.kafka_producer import KafkaArticleProducer, KAFKA_TOPIC_RAW

logger = setup_logger("kafka_crawler", LOGS_DIR / "kafka_crawler.log")


class KafkaIntegratedCrawler:
    """
    Kafka集成版爬虫
    
    与原有AsyncPubMedCrawler的区别：
    1. 爬取的文章实时发送到Kafka，不等待处理
    2. 支持更高的吞吐量（采集和处理并行）
    3. 消息持久化，支持重放
    """
    
    def __init__(self, use_kafka: bool = True):
        """
        初始化
        
        Args:
            use_kafka: 是否使用Kafka（False则回退到原有模式）
        """
        self.use_kafka = use_kafka
        self.producer = None
        self.crawler = None
        
        # 初始化Kafka生产者
        if use_kafka:
            self.producer = KafkaArticleProducer()
            if not self.producer.producer:
                logger.warning("Kafka不可用，回退到本地模式")
                self.use_kafka = False
        
        # 初始化原有爬虫
        from src.data_processing.pubmed_crawler import AsyncPubMedCrawler
        self.crawler = AsyncPubMedCrawler()
        
        # 统计
        self.stats = {
            "crawled": 0,
            "sent_to_kafka": 0,
            "saved_local": 0
        }
    
    def _on_article_crawled(self, article: Dict) -> bool:
        """
        文章爬取回调 - 发送到Kafka
        
        Args:
            article: 爬取的文章
            
        Returns:
            是否成功处理
        """
        self.stats["crawled"] += 1
        
        if self.use_kafka and self.producer:
            # 发送到Kafka
            success = self.producer.send_article(article, KAFKA_TOPIC_RAW)
            if success:
                self.stats["sent_to_kafka"] += 1
            return success
        else:
            # 本地模式
            self.stats["saved_local"] += 1
            return True
    
    def crawl_with_kafka(self, topics: List[str] = None, max_concurrent: int = 3):
        """
        使用Kafka的爬取模式
        
        爬取流程：
        1. 爬虫爬取文章
        2. 文章实时发送到Kafka
        3. 消费者异步处理（可以是另一个进程）
        """
        logger.info("=" * 60)
        logger.info("🚀 Kafka集成爬虫启动")
        logger.info(f"   Kafka模式: {'启用' if self.use_kafka else '禁用'}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 使用原有爬虫爬取
        articles = self.crawler.crawl_all_topics(topics, max_concurrent)
        
        # 批量发送到Kafka（如果之前没有实时发送）
        if self.use_kafka and self.producer:
            logger.info(f"📤 批量发送 {len(articles)} 篇文章到Kafka...")
            sent = self.producer.send_batch(articles)
            self.stats["sent_to_kafka"] = sent
        
        elapsed = time.time() - start_time
        
        # 打印统计
        logger.info("\n" + "=" * 60)
        logger.info("📊 爬取完成统计")
        logger.info(f"   总耗时: {elapsed:.1f} 秒")
        logger.info(f"   爬取文章: {len(articles)}")
        logger.info(f"   发送Kafka: {self.stats['sent_to_kafka']}")
        if self.producer:
            logger.info(f"   Kafka统计: {self.producer.get_stats()}")
        logger.info("=" * 60)
        
        return articles
    
    def close(self):
        """关闭资源"""
        if self.producer:
            self.producer.close()


def run_kafka_pipeline():
    """
    运行Kafka集成的完整Pipeline
    
    架构：
    [爬虫] → [Kafka: raw_articles] → [处理消费者] → [Kafka: processed] → [向量化消费者] → [Milvus]
    """
    import multiprocessing
    from src.messaging.kafka_consumer import DataProcessingConsumer, EmbeddingConsumer
    
    logger.info("=" * 60)
    logger.info("🚀 启动Kafka集成Pipeline")
    logger.info("=" * 60)
    
    # 启动消费者进程
    def run_processor():
        consumer = DataProcessingConsumer()
        consumer.start()
    
    def run_embedder():
        consumer = EmbeddingConsumer()
        consumer.start()
    
    # 启动处理消费者
    processor_process = multiprocessing.Process(target=run_processor, name="DataProcessor")
    processor_process.start()
    logger.info("✅ 数据处理消费者已启动")
    
    # 启动向量化消费者
    embedder_process = multiprocessing.Process(target=run_embedder, name="Embedder")
    embedder_process.start()
    logger.info("✅ 向量化消费者已启动")
    
    # 启动爬虫（生产者）
    crawler = KafkaIntegratedCrawler(use_kafka=True)
    
    try:
        crawler.crawl_with_kafka()
    finally:
        crawler.close()
        
        # 等待消费者处理完成
        logger.info("等待消费者处理完成...")
        time.sleep(30)  # 给消费者一些时间处理剩余消息
        
        processor_process.terminate()
        embedder_process.terminate()
        
        processor_process.join()
        embedder_process.join()
        
        logger.info("✅ Pipeline完成")


def main():
    """测试入口"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", action="store_true", help="使用Kafka模式")
    parser.add_argument("--pipeline", action="store_true", help="运行完整Pipeline")
    args = parser.parse_args()
    
    if args.pipeline:
        run_kafka_pipeline()
    else:
        crawler = KafkaIntegratedCrawler(use_kafka=args.kafka)
        crawler.crawl_with_kafka()
        crawler.close()


if __name__ == "__main__":
    main()
