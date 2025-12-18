# -*- coding: utf-8 -*-
"""
医学知识问答RAG系统 - 主执行脚本
支持完整Pipeline、向量数据库重建、系统评估、Web界面
"""

import argparse
import sys
import io
from pathlib import Path

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent))

from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("main", LOGS_DIR / "main.log")


def run_data_collection(max_per_topic: int = 20000, workers: int = 3, clear: bool = False):
    """步骤1: 数据采集（支持断点续传）"""
    logger.info("\n" + "="*60)
    logger.info("步骤1: 数据采集 - PubMed医学文献")
    logger.info("="*60)
    
    from src.data_processing.pubmed_crawler import PubMedCrawler
    crawler = PubMedCrawler(
        email=PUBMED_EMAIL,
        api_key=PUBMED_API_KEY,
        max_workers=workers
    )
    
    if clear:
        crawler.clear_checkpoints()
    
    crawler.crawl_all_topics(max_per_topic=max_per_topic)


def run_data_processing():
    """步骤2: 数据预处理"""
    logger.info("\n" + "="*60)
    logger.info("步骤2: 数据预处理 - Spark分布式处理")
    logger.info("="*60)
    
    from src.data_processing.data_processor import main as processor_main
    processor_main()


def run_embedding():
    """步骤3: 向量化"""
    logger.info("\n" + "="*60)
    logger.info("步骤3: 文本向量化")
    logger.info("="*60)
    
    from src.embedding.embedder import main as embedder_main
    embedder_main()


def run_vector_db_setup():
    """步骤4: 构建向量数据库"""
    logger.info("\n" + "="*60)
    logger.info("步骤4: 构建Milvus向量数据库")
    logger.info("="*60)
    
    from src.retrieval.milvus_manager import main as milvus_main
    milvus_main()


def run_evaluation(mode: str = "full", scale_factor: int = 10):
    """步骤5: 系统评估"""
    logger.info("\n" + "="*60)
    logger.info("步骤5: RAG系统评估")
    logger.info("="*60)
    
    from src.evaluation.unified_evaluator import UnifiedEvaluator
    evaluator = UnifiedEvaluator()
    
    if mode == "rag":
        results = evaluator.evaluate_rag_retrieval()
    elif mode == "storage":
        results = evaluator.evaluate_storage_performance()
    elif mode == "pyspark":
        results = evaluator.evaluate_pyspark_processing(scale_factor=scale_factor)
    else:
        results = evaluator.run_full_evaluation()
    
    if isinstance(results, dict) and 'overall_score' in results:
        logger.info(f"综合评分: {results['overall_score']}/100")
    
    return results


def run_expand_data(scale_factor: int = 10):
    """扩展数据集用于大数据测试"""
    logger.info("\n" + "="*60)
    logger.info(f"📊 扩展数据集 ({scale_factor}x)")
    logger.info("="*60)
    
    from src.evaluation.data_scaler import create_scaled_dataset
    path = create_scaled_dataset(scale_factor=scale_factor)
    
    logger.info(f"✅ 扩展数据集已创建: {path}")
    return path


def run_rebuild_database(resume: bool = False, batch_size: int = 128):
    """重建向量数据库（支持150万数据）"""
    logger.info("\n" + "="*60)
    logger.info("🔄 重建向量数据库")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"   断点续传: {'是' if resume else '否'}")
    logger.info("="*60)
    
    from src.retrieval.milvus_manager import rebuild_database
    rebuild_database(resume=resume, batch_size=batch_size)
    
    logger.info("\n✅ 向量数据库重建完成！")
    logger.info("   可以运行 python main.py --web 启动Web界面测试")


def run_web_interface():
    """步骤6: 启动Web界面"""
    logger.info("\n" + "="*60)
    logger.info("步骤6: 启动Web界面")
    logger.info("="*60)
    
    from web.app import main as web_main
    web_main()


def run_full_pipeline():
    """运行完整Pipeline"""
    logger.info("="*60)
    logger.info("🚀 开始执行完整Pipeline")
    logger.info("="*60)
    
    try:
        # 步骤1: 数据采集
        run_data_collection()
        
        # 步骤2: 数据处理
        run_data_processing()
        
        # 步骤3: 向量化
        run_embedding()
        
        # 步骤4: 构建向量数据库
        run_vector_db_setup()
        
        # 步骤5: 评估
        run_evaluation()
        
        logger.info("\n" + "="*60)
        logger.info("✅ 完整Pipeline执行成功！")
        logger.info("="*60)
        logger.info("\n现在可以启动Web界面进行测试:")
        logger.info("  python main.py --web")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline执行失败: {e}")
        raise


def run_spark_cluster():
    """启动Spark集群（Docker）"""
    logger.info("🚀 启动Spark集群...")
    import subprocess
    subprocess.run(["docker", "compose", "-f", "docker/docker-compose-spark.yml", "up", "-d"])
    logger.info("✅ Spark集群已启动")
    logger.info("   Master UI: http://localhost:8080")
    logger.info("   Master URL: spark://localhost:7077")


def run_spark_embed(use_cluster: bool = False):
    """使用Spark分布式向量化"""
    logger.info("⚡ Spark分布式向量化")
    from src.embedding.spark_embedder import SparkEmbedder
    
    embedder = SparkEmbedder(use_cluster=use_cluster)
    input_path = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
    output_path = EMBEDDING_DATA_DIR / "spark_embeddings"
    
    if input_path.exists():
        embedder.embed_with_pandas_udf(input_path, output_path)
    embedder.stop()


def run_incremental_index(use_spark: bool = False):
    """启动增量索引"""
    logger.info("📡 启动增量索引...")
    from src.retrieval.spark_streaming import IncrementalIndexer
    
    indexer = IncrementalIndexer(use_spark=use_spark)
    if use_spark:
        indexer.start_spark_streaming()
    else:
        indexer.start_file_watcher(interval=30)
    
    logger.info("监听中... 按Ctrl+C停止")
    try:
        import time
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        indexer.stop()


def run_cache_prewarm():
    """预热Redis缓存"""
    logger.info("🔥 预热Redis缓存...")
    from src.caching.redis_cache import RedisCache, VectorCacheManager
    
    cache = RedisCache()
    manager = VectorCacheManager(cache)
    
    # 预热常用医学查询
    common_queries = [
        "diabetes symptoms treatment",
        "cardiovascular disease prevention",
        "cancer chemotherapy side effects",
        "hypertension medication",
        "covid-19 vaccine effectiveness",
    ]
    manager.prewarm_cache(common_queries)
    logger.info(f"✅ 缓存预热完成: {manager.get_stats()}")


# ==================== Kafka + Airflow 功能 ====================

def run_kafka_services():
    """启动Kafka相关服务"""
    logger.info("🚀 启动Kafka服务...")
    import subprocess
    
    # 先创建网络（如果不存在）
    subprocess.run(["docker", "network", "create", "docker_rag-network"], 
                   capture_output=True)
    
    # 启动Kafka + Airflow
    result = subprocess.run([
        "docker", "compose", 
        "-f", "docker/docker-compose-kafka-airflow.yml", 
        "up", "-d"
    ])
    
    if result.returncode == 0:
        logger.info("✅ Kafka + Airflow 服务已启动")
        logger.info("   Kafka: localhost:9092")
        logger.info("   Kafka UI: http://localhost:8082")
        logger.info("   Airflow: http://localhost:8081 (admin/admin)")
    else:
        logger.error("❌ 服务启动失败")


def run_kafka_topics_setup():
    """创建Kafka Topics"""
    logger.info("📋 创建Kafka Topics...")
    from src.messaging.kafka_producer import KafkaTopicManager
    
    manager = KafkaTopicManager()
    manager.create_topics()
    
    topics = manager.list_topics()
    logger.info(f"✅ 当前Topics: {topics}")


def run_kafka_crawler(use_kafka: bool = True):
    """使用Kafka集成的爬虫"""
    logger.info("🕷️ 启动Kafka集成爬虫...")
    from src.messaging.kafka_integrated_crawler import KafkaIntegratedCrawler
    
    crawler = KafkaIntegratedCrawler(use_kafka=use_kafka)
    crawler.crawl_with_kafka()
    crawler.close()


def run_kafka_consumer(consumer_type: str = "processor"):
    """启动Kafka消费者"""
    logger.info(f"👂 启动Kafka消费者: {consumer_type}")
    
    if consumer_type == "processor":
        from src.messaging.kafka_consumer import DataProcessingConsumer
        consumer = DataProcessingConsumer()
        consumer.start()
    elif consumer_type == "embedder":
        from src.messaging.kafka_consumer import EmbeddingConsumer
        consumer = EmbeddingConsumer()
        consumer.start()
    else:
        logger.error(f"未知消费者类型: {consumer_type}")


def run_kafka_pipeline():
    """运行Kafka集成的完整Pipeline"""
    logger.info("🔄 启动Kafka Pipeline...")
    from src.messaging.kafka_integrated_crawler import run_kafka_pipeline as kafka_pipeline
    kafka_pipeline()


def show_kafka_stats():
    """显示Kafka统计信息"""
    logger.info("📊 Kafka统计信息")
    try:
        from kafka import KafkaConsumer
        from src.messaging.kafka_producer import (
            KAFKA_BOOTSTRAP_SERVERS, 
            KAFKA_TOPIC_RAW, 
            KAFKA_TOPIC_PROCESSED,
            KAFKA_TOPIC_EMBEDDINGS
        )
        
        consumer = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        topics = consumer.topics()
        
        logger.info(f"   可用Topics: {list(topics)}")
        
        # 获取各Topic的消息数量
        for topic in [KAFKA_TOPIC_RAW, KAFKA_TOPIC_PROCESSED, KAFKA_TOPIC_EMBEDDINGS]:
            if topic in topics:
                partitions = consumer.partitions_for_topic(topic)
                logger.info(f"   {topic}: {len(partitions) if partitions else 0} 分区")
        
        consumer.close()
    except Exception as e:
        logger.error(f"获取Kafka统计失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="医学知识问答RAG系统")
    
    # Pipeline步骤
    parser.add_argument("--collect", action="store_true", help="数据采集(支持断点续传)")
    parser.add_argument("--max-per-topic", type=int, default=20000, help="每主题最大文章数")
    parser.add_argument("--workers", type=int, default=3, help="爬虫并行线程数")
    parser.add_argument("--clear-checkpoint", action="store_true", help="清除checkpoint重新爬取")
    parser.add_argument("--process", action="store_true", help="数据处理")
    parser.add_argument("--embed", action="store_true", help="向量化")
    parser.add_argument("--setup-db", action="store_true", help="构建向量数据库")
    parser.add_argument("--full", action="store_true", help="运行完整Pipeline")
    
    # 数据库重建
    parser.add_argument("--rebuild", action="store_true", help="重建向量数据库")
    parser.add_argument("--resume", action="store_true", help="断点续传模式(配合--rebuild)")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小(默认128)")
    
    # 评估
    parser.add_argument("--eval", action="store_true", help="完整系统评估")
    parser.add_argument("--eval-rag", action="store_true", help="仅RAG检索评估")
    parser.add_argument("--eval-storage", action="store_true", help="仅存储性能评估")
    parser.add_argument("--eval-pyspark", action="store_true", help="PySpark大数据处理评估")
    
    # 数据扩展
    parser.add_argument("--expand-data", action="store_true", help="扩展数据集用于大数据测试")
    parser.add_argument("--scale", type=int, default=10, help="数据扩展倍数(默认10x)")
    
    # Spark增强功能
    parser.add_argument("--spark-cluster", action="store_true", help="启动Spark集群(Docker)")
    parser.add_argument("--spark-embed", action="store_true", help="Spark分布式向量化")
    parser.add_argument("--use-cluster", action="store_true", help="使用Spark集群模式")
    parser.add_argument("--incremental", action="store_true", help="启动增量索引")
    parser.add_argument("--cache-prewarm", action="store_true", help="预热Redis缓存")
    
    # Kafka + Airflow 功能
    parser.add_argument("--kafka-start", action="store_true", help="启动Kafka+Airflow服务")
    parser.add_argument("--kafka-topics", action="store_true", help="创建Kafka Topics")
    parser.add_argument("--kafka-crawl", action="store_true", help="Kafka集成爬虫")
    parser.add_argument("--kafka-consumer", type=str, choices=["processor", "embedder"], 
                        help="启动Kafka消费者")
    parser.add_argument("--kafka-pipeline", action="store_true", help="Kafka完整Pipeline")
    parser.add_argument("--kafka-stats", action="store_true", help="显示Kafka统计")
    
    # Web界面
    parser.add_argument("--web", action="store_true", help="启动Web界面")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        print("\n" + "="*50)
        print("常用命令:")
        print("  python main.py --full           # 运行完整Pipeline")
        print("  python main.py --rebuild        # 重建向量数据库")
        print("  python main.py --eval           # 完整系统评估")
        print("  python main.py --web            # 启动Web界面")
        print("\nSpark增强:")
        print("  python main.py --spark-cluster  # 启动Spark集群")
        print("  python main.py --spark-embed    # Spark分布式向量化")
        print("  python main.py --incremental    # 增量索引")
        print("  python main.py --cache-prewarm  # 预热缓存")
        print("\nKafka + Airflow:")
        print("  python main.py --kafka-start    # 启动Kafka+Airflow服务")
        print("  python main.py --kafka-topics   # 创建Kafka Topics")
        print("  python main.py --kafka-crawl    # Kafka集成爬虫")
        print("  python main.py --kafka-consumer processor  # 启动处理消费者")
        print("  python main.py --kafka-pipeline # Kafka完整Pipeline")
        return
    
    # 执行对应步骤
    # Kafka + Airflow 命令
    if args.kafka_start:
        run_kafka_services()
    elif args.kafka_topics:
        run_kafka_topics_setup()
    elif args.kafka_crawl:
        run_kafka_crawler(use_kafka=True)
    elif args.kafka_consumer:
        run_kafka_consumer(args.kafka_consumer)
    elif args.kafka_pipeline:
        run_kafka_pipeline()
    elif args.kafka_stats:
        show_kafka_stats()
    # Spark命令
    elif args.spark_cluster:
        run_spark_cluster()
    elif args.spark_embed:
        run_spark_embed(use_cluster=args.use_cluster)
    elif args.incremental:
        run_incremental_index(use_spark=args.use_cluster)
    elif args.cache_prewarm:
        run_cache_prewarm()
    elif args.full:
        run_full_pipeline()
    elif args.rebuild:
        run_rebuild_database(resume=args.resume, batch_size=args.batch_size)
    elif args.eval or args.eval_rag or args.eval_storage or args.eval_pyspark:
        if args.eval_pyspark:
            run_evaluation("pyspark", scale_factor=args.scale)
        else:
            mode = "rag" if args.eval_rag else "storage" if args.eval_storage else "full"
            run_evaluation(mode)
    else:
        if args.collect:
            run_data_collection(
                max_per_topic=args.max_per_topic,
                workers=args.workers,
                clear=args.clear_checkpoint
            )
        if args.process:
            run_data_processing()
        if args.embed:
            run_embedding()
        if args.setup_db:
            run_vector_db_setup()
        if args.web:
            run_web_interface()


if __name__ == "__main__":
    main()
