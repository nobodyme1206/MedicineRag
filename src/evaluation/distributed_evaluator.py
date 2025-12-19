#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式计算评估模块
评估所有数据密集型技术栈的性能

评估内容:
1. PySpark - 大数据处理 (vs Pandas)
2. Milvus - 向量数据库性能
3. Redis - 缓存性能
4. Kafka - 消息队列吞吐量
5. MongoDB - 文档存储性能
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from config.config import (
    PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR,
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, MILVUS_NPROBE,
    REDIS_HOST, REDIS_PORT,
    MONGODB_HOST, MONGODB_PORT
)
from src.utils.logger import setup_logger
from src.utils.exceptions import handle_errors

logger = setup_logger("distributed_evaluator", LOGS_DIR / "distributed_evaluation.log")

# 类型别名
EvalResult = Dict[str, Any]
Metrics = Dict[str, float]


class DistributedEvaluator:
    """分布式计算评估器"""
    
    def __init__(self):
        self.data_path = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "pyspark": {},
            "milvus": {},
            "redis": {},
            "kafka": {},
            "mongodb": {},
            "summary": {}
        }
    
    # ==================== 1. PySpark评估 ====================
    
    def evaluate_pyspark(self) -> Dict:
        """评估PySpark vs Pandas性能"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PySpark 分布式处理评估")
        logger.info("=" * 60)
        
        if not self.data_path.exists():
            return {"error": "数据文件不存在"}
        
        result = {"pandas": {}, "pyspark": {}, "comparison": {}}
        
        # Pandas测试
        logger.info("\n1️⃣ Pandas 性能测试...")
        start = time.time()
        df = pd.read_parquet(self.data_path)
        pandas_read = time.time() - start
        
        text_col = 'content' if 'content' in df.columns else df.columns[0]
        start = time.time()
        df['text_length'] = df[text_col].astype(str).str.len()
        _ = df.groupby('topic')['text_length'].agg(['mean', 'max', 'min']).reset_index()
        pandas_process = time.time() - start
        
        result["pandas"] = {
            "read_time_s": round(pandas_read, 3),
            "process_time_s": round(pandas_process, 3),
            "total_time_s": round(pandas_read + pandas_process, 3),
            "records": len(df),
            "throughput_rec_s": round(len(df) / (pandas_read + pandas_process), 0)
        }
        logger.info(f"   Pandas: {result['pandas']['total_time_s']:.3f}s, "
                   f"{result['pandas']['throughput_rec_s']:,.0f} rec/s")
        
        # PySpark测试
        logger.info("\n2️⃣ PySpark 性能测试...")
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import length, col, avg, max as spark_max, min as spark_min
            
            spark = SparkSession.builder \
                .appName("SparkEvaluation") \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            spark.sparkContext.setLogLevel("WARN")
            
            start = time.time()
            sdf = spark.read.parquet(str(self.data_path))
            count = sdf.count()
            spark_read = time.time() - start
            
            cols = sdf.columns
            text_col = 'content' if 'content' in cols else cols[0]
            start = time.time()
            sdf = sdf.withColumn("text_length", length(col(text_col)))
            _ = sdf.groupBy("topic").agg(
                avg("text_length"), spark_max("text_length"), spark_min("text_length")
            ).collect()
            spark_process = time.time() - start
            
            spark.stop()
            
            result["pyspark"] = {
                "read_time_s": round(spark_read, 3),
                "process_time_s": round(spark_process, 3),
                "total_time_s": round(spark_read + spark_process, 3),
                "records": count,
                "throughput_rec_s": round(count / (spark_read + spark_process), 0)
            }
            logger.info(f"   PySpark: {result['pyspark']['total_time_s']:.3f}s, "
                       f"{result['pyspark']['throughput_rec_s']:,.0f} rec/s")
            
            # 对比
            speedup = result["pandas"]["total_time_s"] / result["pyspark"]["total_time_s"]
            result["comparison"] = {
                "speedup": round(speedup, 2),
                "winner": "PySpark" if speedup > 1 else "Pandas",
                "note": f"PySpark {'快' if speedup > 1 else '慢'} {abs(speedup-1)*100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"PySpark评估失败: {e}")
            result["pyspark"] = {"error": str(e)}
        
        self.results["pyspark"] = result
        return result

    # ==================== 2. Milvus评估 ====================
    
    def evaluate_milvus(self, num_queries: int = 100) -> Dict:
        """评估Milvus向量数据库性能"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Milvus 向量数据库评估")
        logger.info("=" * 60)
        
        result = {
            "connection": False,
            "collection_stats": {},
            "search_performance": {},
            "insert_performance": {}
        }
        
        try:
            from pymilvus import connections, Collection, utility
            
            # 连接测试
            start = time.time()
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            connect_time = time.time() - start
            result["connection"] = True
            result["connect_time_ms"] = round(connect_time * 1000, 2)
            logger.info(f"   连接成功: {connect_time*1000:.2f}ms")
            
            # 集合统计
            if utility.has_collection(MILVUS_COLLECTION_NAME):
                collection = Collection(MILVUS_COLLECTION_NAME)
                collection.load()
                
                result["collection_stats"] = {
                    "name": MILVUS_COLLECTION_NAME,
                    "num_entities": collection.num_entities,
                    "index_type": MILVUS_INDEX_TYPE
                }
                logger.info(f"   集合: {MILVUS_COLLECTION_NAME}, "
                           f"向量数: {collection.num_entities:,}")
                
                # 搜索性能测试
                logger.info(f"\n   搜索性能测试 ({num_queries} 次查询)...")
                from src.embedding.embedder import TextEmbedder
                embedder = TextEmbedder()
                
                test_queries = [
                    "diabetes treatment", "cancer therapy", "heart disease",
                    "covid vaccine", "mental health", "obesity prevention"
                ] * (num_queries // 6 + 1)
                
                latencies = []
                for query in test_queries[:num_queries]:
                    vector = embedder.encode_single(query)
                    
                    start = time.time()
                    _ = collection.search(
                        data=[vector.tolist()],
                        anns_field="embedding",
                        param={"metric_type": MILVUS_METRIC_TYPE, "params": {"nprobe": MILVUS_NPROBE}},
                        limit=10
                    )
                    latencies.append((time.time() - start) * 1000)
                
                result["search_performance"] = {
                    "num_queries": num_queries,
                    "avg_latency_ms": round(np.mean(latencies), 2),
                    "p50_latency_ms": round(np.percentile(latencies, 50), 2),
                    "p95_latency_ms": round(np.percentile(latencies, 95), 2),
                    "p99_latency_ms": round(np.percentile(latencies, 99), 2),
                    "qps": round(1000 / np.mean(latencies), 1)
                }
                logger.info(f"   平均延迟: {result['search_performance']['avg_latency_ms']:.2f}ms, "
                           f"QPS: {result['search_performance']['qps']:.1f}")
            else:
                logger.warning(f"   集合 {MILVUS_COLLECTION_NAME} 不存在")
            
            connections.disconnect("default")
            
        except Exception as e:
            logger.error(f"Milvus评估失败: {e}")
            result["error"] = str(e)
        
        self.results["milvus"] = result
        return result
    
    # ==================== 3. Redis评估 ====================
    
    def evaluate_redis(self, num_ops: int = 1000) -> Dict:
        """评估Redis缓存性能"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Redis 缓存评估")
        logger.info("=" * 60)
        
        result = {
            "connection": False,
            "write_performance": {},
            "read_performance": {},
            "vector_cache_performance": {}
        }
        
        try:
            import redis
            
            # 连接测试
            start = time.time()
            client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=15)  # 测试用db
            client.ping()
            connect_time = time.time() - start
            result["connection"] = True
            result["connect_time_ms"] = round(connect_time * 1000, 2)
            logger.info(f"   连接成功: {connect_time*1000:.2f}ms")
            
            # 写入性能
            logger.info(f"\n   写入性能测试 ({num_ops} 次)...")
            start = time.time()
            for i in range(num_ops):
                client.set(f"test_key_{i}", f"test_value_{i}" * 100)
            write_time = time.time() - start
            
            result["write_performance"] = {
                "num_ops": num_ops,
                "total_time_s": round(write_time, 3),
                "ops_per_sec": round(num_ops / write_time, 0),
                "avg_latency_ms": round(write_time / num_ops * 1000, 3)
            }
            logger.info(f"   写入: {result['write_performance']['ops_per_sec']:,.0f} ops/s")
            
            # 读取性能
            logger.info(f"\n   读取性能测试 ({num_ops} 次)...")
            start = time.time()
            for i in range(num_ops):
                _ = client.get(f"test_key_{i}")
            read_time = time.time() - start
            
            result["read_performance"] = {
                "num_ops": num_ops,
                "total_time_s": round(read_time, 3),
                "ops_per_sec": round(num_ops / read_time, 0),
                "avg_latency_ms": round(read_time / num_ops * 1000, 3)
            }
            logger.info(f"   读取: {result['read_performance']['ops_per_sec']:,.0f} ops/s")
            
            # 向量缓存性能（512维float32）
            logger.info(f"\n   向量缓存测试...")
            vector_ops = num_ops // 10
            vectors = [np.random.rand(512).astype(np.float32).tobytes() for _ in range(vector_ops)]
            
            start = time.time()
            for i, vec in enumerate(vectors):
                client.set(f"vec_{i}", vec)
            vec_write_time = time.time() - start
            
            start = time.time()
            for i in range(vector_ops):
                _ = client.get(f"vec_{i}")
            vec_read_time = time.time() - start
            
            result["vector_cache_performance"] = {
                "num_vectors": vector_ops,
                "vector_dim": 512,
                "write_ops_per_sec": round(vector_ops / vec_write_time, 0),
                "read_ops_per_sec": round(vector_ops / vec_read_time, 0)
            }
            logger.info(f"   向量写入: {result['vector_cache_performance']['write_ops_per_sec']:,.0f} ops/s")
            logger.info(f"   向量读取: {result['vector_cache_performance']['read_ops_per_sec']:,.0f} ops/s")
            
            # 清理测试数据
            for i in range(num_ops):
                client.delete(f"test_key_{i}")
            for i in range(vector_ops):
                client.delete(f"vec_{i}")
            
            client.close()
            
        except Exception as e:
            logger.error(f"Redis评估失败: {e}")
            result["error"] = str(e)
        
        self.results["redis"] = result
        return result

    # ==================== 4. Kafka评估 ====================
    
    def evaluate_kafka(self, num_messages: int = 1000) -> Dict:
        """评估Kafka消息队列性能"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Kafka 消息队列评估")
        logger.info("=" * 60)
        
        result = {
            "connection": False,
            "producer_performance": {},
            "consumer_performance": {},
            "throughput": {}
        }
        
        try:
            from kafka import KafkaProducer, KafkaConsumer
            from kafka.admin import KafkaAdminClient, NewTopic
            
            bootstrap_servers = "localhost:9092"
            test_topic = "eval_test_topic"
            
            # 连接测试
            start = time.time()
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
            connect_time = time.time() - start
            result["connection"] = True
            result["connect_time_ms"] = round(connect_time * 1000, 2)
            logger.info(f"   连接成功: {connect_time*1000:.2f}ms")
            
            # 创建测试topic
            try:
                admin.create_topics([NewTopic(test_topic, num_partitions=3, replication_factor=1)])
            except Exception:
                pass  # topic可能已存在
            
            # 生产者性能
            logger.info(f"\n   生产者性能测试 ({num_messages} 条消息)...")
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            test_message = {"id": 0, "text": "test message " * 50, "timestamp": ""}
            message_size = len(json.dumps(test_message).encode('utf-8'))
            
            start = time.time()
            for i in range(num_messages):
                test_message["id"] = i
                test_message["timestamp"] = datetime.now().isoformat()
                producer.send(test_topic, test_message)
            producer.flush()
            produce_time = time.time() - start
            
            result["producer_performance"] = {
                "num_messages": num_messages,
                "message_size_bytes": message_size,
                "total_time_s": round(produce_time, 3),
                "messages_per_sec": round(num_messages / produce_time, 0),
                "throughput_mb_s": round(num_messages * message_size / produce_time / 1024 / 1024, 2)
            }
            logger.info(f"   生产: {result['producer_performance']['messages_per_sec']:,.0f} msg/s, "
                       f"{result['producer_performance']['throughput_mb_s']:.2f} MB/s")
            
            producer.close()
            
            # 消费者性能
            logger.info(f"\n   消费者性能测试...")
            consumer = KafkaConsumer(
                test_topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                consumer_timeout_ms=5000,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            
            start = time.time()
            consumed = 0
            for msg in consumer:
                consumed += 1
                if consumed >= num_messages:
                    break
            consume_time = time.time() - start
            
            result["consumer_performance"] = {
                "num_messages": consumed,
                "total_time_s": round(consume_time, 3),
                "messages_per_sec": round(consumed / consume_time, 0) if consume_time > 0 else 0,
                "throughput_mb_s": round(consumed * message_size / consume_time / 1024 / 1024, 2) if consume_time > 0 else 0
            }
            logger.info(f"   消费: {result['consumer_performance']['messages_per_sec']:,.0f} msg/s")
            
            consumer.close()
            
            # 删除测试topic
            try:
                admin.delete_topics([test_topic])
            except Exception:
                pass
            
            admin.close()
            
        except Exception as e:
            logger.error(f"Kafka评估失败: {e}")
            result["error"] = str(e)
        
        self.results["kafka"] = result
        return result
    
    # ==================== 5. MongoDB评估 ====================
    
    def evaluate_mongodb(self, num_docs: int = 1000) -> Dict:
        """评估MongoDB文档存储性能"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MongoDB 文档存储评估")
        logger.info("=" * 60)
        
        result = {
            "connection": False,
            "insert_performance": {},
            "query_performance": {},
            "aggregate_performance": {}
        }
        
        try:
            from pymongo import MongoClient
            
            # 连接测试
            start = time.time()
            client = MongoClient(MONGODB_HOST, MONGODB_PORT, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            connect_time = time.time() - start
            result["connection"] = True
            result["connect_time_ms"] = round(connect_time * 1000, 2)
            logger.info(f"   连接成功: {connect_time*1000:.2f}ms")
            
            db = client["eval_test_db"]
            collection = db["eval_test_collection"]
            collection.drop()  # 清理
            
            # 插入性能
            logger.info(f"\n   插入性能测试 ({num_docs} 条文档)...")
            test_docs = [
                {
                    "id": i,
                    "title": f"Test Document {i}",
                    "content": "This is test content. " * 50,
                    "topic": f"topic_{i % 10}",
                    "timestamp": datetime.now()
                }
                for i in range(num_docs)
            ]
            
            start = time.time()
            collection.insert_many(test_docs)
            insert_time = time.time() - start
            
            result["insert_performance"] = {
                "num_docs": num_docs,
                "total_time_s": round(insert_time, 3),
                "docs_per_sec": round(num_docs / insert_time, 0)
            }
            logger.info(f"   插入: {result['insert_performance']['docs_per_sec']:,.0f} docs/s")
            
            # 查询性能
            logger.info(f"\n   查询性能测试...")
            num_queries = 100
            
            start = time.time()
            for i in range(num_queries):
                _ = list(collection.find({"topic": f"topic_{i % 10}"}).limit(10))
            query_time = time.time() - start
            
            result["query_performance"] = {
                "num_queries": num_queries,
                "total_time_s": round(query_time, 3),
                "queries_per_sec": round(num_queries / query_time, 0),
                "avg_latency_ms": round(query_time / num_queries * 1000, 2)
            }
            logger.info(f"   查询: {result['query_performance']['queries_per_sec']:,.0f} qps, "
                       f"延迟: {result['query_performance']['avg_latency_ms']:.2f}ms")
            
            # 聚合性能
            logger.info(f"\n   聚合性能测试...")
            start = time.time()
            _ = list(collection.aggregate([
                {"$group": {"_id": "$topic", "count": {"$sum": 1}, "avg_len": {"$avg": {"$strLenCP": "$content"}}}},
                {"$sort": {"count": -1}}
            ]))
            agg_time = time.time() - start
            
            result["aggregate_performance"] = {
                "time_s": round(agg_time, 3),
                "docs_processed": num_docs
            }
            logger.info(f"   聚合: {agg_time:.3f}s")
            
            # 清理
            collection.drop()
            client.close()
            
        except Exception as e:
            logger.error(f"MongoDB评估失败: {e}")
            result["error"] = str(e)
        
        self.results["mongodb"] = result
        return result

    # ==================== 完整评估 ====================
    
    def run_evaluation(self) -> Dict:
        """运行完整分布式计算评估"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 分布式计算技术栈评估")
        logger.info("   PySpark | Milvus | Redis | Kafka | MongoDB")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # 依次评估各组件
        self.evaluate_pyspark()
        self.evaluate_milvus()
        self.evaluate_redis()
        self.evaluate_kafka()
        self.evaluate_mongodb()
        
        # 生成汇总
        self.results["summary"] = self._generate_summary()
        self.results["total_time_s"] = round(time.time() - start_time, 2)
        
        # 打印汇总
        self._print_summary()
        
        # 保存结果
        self._save_results()
        
        return self.results
    
    def _generate_summary(self) -> Dict:
        """生成评估汇总"""
        summary = {
            "components_tested": 0,
            "components_passed": 0,
            "components_failed": 0,
            "highlights": []
        }
        
        components = ["pyspark", "milvus", "redis", "kafka", "mongodb"]
        
        for comp in components:
            data = self.results.get(comp, {})
            if data:
                summary["components_tested"] += 1
                if "error" not in data and data.get("connection", True):
                    summary["components_passed"] += 1
                else:
                    summary["components_failed"] += 1
        
        # 提取亮点
        if self.results.get("pyspark", {}).get("comparison", {}).get("winner"):
            winner = self.results["pyspark"]["comparison"]["winner"]
            speedup = self.results["pyspark"]["comparison"].get("speedup", 1)
            summary["highlights"].append(f"数据处理: {winner} (加速比 {speedup}x)")
        
        if self.results.get("milvus", {}).get("search_performance", {}).get("qps"):
            qps = self.results["milvus"]["search_performance"]["qps"]
            summary["highlights"].append(f"向量检索: {qps:.1f} QPS")
        
        if self.results.get("redis", {}).get("read_performance", {}).get("ops_per_sec"):
            ops = self.results["redis"]["read_performance"]["ops_per_sec"]
            summary["highlights"].append(f"Redis缓存: {ops:,.0f} ops/s")
        
        if self.results.get("kafka", {}).get("producer_performance", {}).get("messages_per_sec"):
            msg_s = self.results["kafka"]["producer_performance"]["messages_per_sec"]
            summary["highlights"].append(f"Kafka吞吐: {msg_s:,.0f} msg/s")
        
        if self.results.get("mongodb", {}).get("query_performance", {}).get("queries_per_sec"):
            qps = self.results["mongodb"]["query_performance"]["queries_per_sec"]
            summary["highlights"].append(f"MongoDB查询: {qps:,.0f} qps")
        
        return summary
    
    def _print_summary(self):
        """打印评估汇总"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 分布式计算评估汇总")
        logger.info("=" * 70)
        
        summary = self.results.get("summary", {})
        
        logger.info(f"\n组件状态: {summary.get('components_passed', 0)}/{summary.get('components_tested', 0)} 通过")
        
        logger.info("\n性能亮点:")
        for highlight in summary.get("highlights", []):
            logger.info(f"   ✅ {highlight}")
        
        # 详细表格
        logger.info("\n" + "-" * 70)
        logger.info(f"{'组件':<12}{'状态':<10}{'关键指标':<40}")
        logger.info("-" * 70)
        
        # PySpark
        ps = self.results.get("pyspark", {})
        if "error" not in ps:
            throughput = ps.get("pyspark", {}).get("throughput_rec_s", 0)
            logger.info(f"{'PySpark':<12}{'✅ 正常':<10}{f'吞吐量: {throughput:,.0f} rec/s':<40}")
        else:
            logger.info(f"{'PySpark':<12}{'❌ 失败':<10}{ps.get('error', '')[:40]:<40}")
        
        # Milvus
        mv = self.results.get("milvus", {})
        if mv.get("connection"):
            qps = mv.get("search_performance", {}).get("qps", 0)
            latency = mv.get("search_performance", {}).get("avg_latency_ms", 0)
            logger.info(f"{'Milvus':<12}{'✅ 正常':<10}{f'QPS: {qps:.1f}, 延迟: {latency:.1f}ms':<40}")
        else:
            logger.info(f"{'Milvus':<12}{'❌ 失败':<10}{mv.get('error', '未连接')[:40]:<40}")
        
        # Redis
        rd = self.results.get("redis", {})
        if rd.get("connection"):
            read_ops = rd.get("read_performance", {}).get("ops_per_sec", 0)
            write_ops = rd.get("write_performance", {}).get("ops_per_sec", 0)
            logger.info(f"{'Redis':<12}{'✅ 正常':<10}{f'读: {read_ops:,.0f}, 写: {write_ops:,.0f} ops/s':<40}")
        else:
            logger.info(f"{'Redis':<12}{'❌ 失败':<10}{rd.get('error', '未连接')[:40]:<40}")
        
        # Kafka
        kf = self.results.get("kafka", {})
        if kf.get("connection"):
            prod = kf.get("producer_performance", {}).get("messages_per_sec", 0)
            cons = kf.get("consumer_performance", {}).get("messages_per_sec", 0)
            logger.info(f"{'Kafka':<12}{'✅ 正常':<10}{f'生产: {prod:,.0f}, 消费: {cons:,.0f} msg/s':<40}")
        else:
            logger.info(f"{'Kafka':<12}{'❌ 失败':<10}{kf.get('error', '未连接')[:40]:<40}")
        
        # MongoDB
        mg = self.results.get("mongodb", {})
        if mg.get("connection"):
            insert = mg.get("insert_performance", {}).get("docs_per_sec", 0)
            query = mg.get("query_performance", {}).get("queries_per_sec", 0)
            logger.info(f"{'MongoDB':<12}{'✅ 正常':<10}{f'插入: {insert:,.0f}, 查询: {query:,.0f} ops/s':<40}")
        else:
            logger.info(f"{'MongoDB':<12}{'❌ 失败':<10}{mg.get('error', '未连接')[:40]:<40}")
        
        logger.info("=" * 70)
    
    def _save_results(self):
        """保存结果"""
        output_file = RESULTS_DIR / "evaluation" / "distributed_evaluation.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"\n结果已保存: {output_file}")


def main():
    """命令行入口"""
    evaluator = DistributedEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
