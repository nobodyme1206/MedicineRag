#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB元数据存储模块
用于存储文档元数据、查询日志、系统统计等
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("mongodb_storage", LOGS_DIR / "mongodb_storage.log")


class MongoDBStorage:
    """MongoDB文档数据库管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "medical_rag"
    ):
        """
        初始化MongoDB连接
        
        Args:
            host: MongoDB主机地址
            port: 端口
            database: 数据库名称
        """
        try:
            self.client = MongoClient(
                host=host,
                port=port,
                serverSelectionTimeoutMS=5000
            )
            # 测试连接
            self.client.admin.command('ping')
            
            self.db = self.client[database]
            logger.info(f"✅ 连接MongoDB成功: {host}:{port}/{database}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ 连接MongoDB失败: {e}")
            raise
    
    def save_chunks_metadata(self, chunks: List[Dict], collection_name: str = "chunks_metadata"):
        """
        保存文档chunks元数据到MongoDB
        
        Args:
            chunks: 文档chunks列表
            collection_name: 集合名称
            
        Returns:
            插入的文档数量
        """
        collection = self.db[collection_name]
        
        logger.info(f"开始保存 {len(chunks)} 个chunks元数据...")
        
        # 添加时间戳
        for chunk in chunks:
            chunk['created_at'] = datetime.now()
            chunk['updated_at'] = datetime.now()
        
        # 批量插入
        if chunks:
            result = collection.insert_many(chunks)
            logger.info(f"✅ 保存成功: {len(result.inserted_ids)} 个文档")
            return len(result.inserted_ids)
        
        return 0
    
    def query_chunks_by_category(
        self,
        category: str,
        limit: int = 100,
        collection_name: str = "chunks_metadata"
    ) -> List[Dict]:
        """
        按类别查询chunks元数据
        
        Args:
            category: 疾病类别
            limit: 返回数量限制
            collection_name: 集合名称
            
        Returns:
            符合条件的文档列表
        """
        collection = self.db[collection_name]
        
        logger.info(f"查询类别: {category}, 限制: {limit}")
        
        results = list(collection.find(
            {"category": category},
            {"_id": 0}  # 排除_id字段
        ).limit(limit))
        
        logger.info(f"✅ 查询到 {len(results)} 条结果")
        
        return results
    
    def log_query(
        self,
        query_text: str,
        results: List[Dict],
        metrics: Dict,
        collection_name: str = "query_logs"
    ):
        """
        记录查询日志
        
        Args:
            query_text: 查询文本
            results: 检索结果
            metrics: 性能指标 (召回率、延迟等)
            collection_name: 集合名称
        """
        collection = self.db[collection_name]
        
        log_entry = {
            'timestamp': datetime.now(),
            'query': query_text,
            'num_results': len(results),
            'metrics': metrics,
            'result_ids': [r.get('id', '') for r in results[:10]]  # 只记录前10个
        }
        
        collection.insert_one(log_entry)
        logger.debug(f"查询日志已记录: {query_text[:50]}...")
    
    def get_query_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        collection_name: str = "query_logs"
    ) -> Dict:
        """
        获取查询统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            collection_name: 集合名称
            
        Returns:
            统计数据字典
        """
        collection = self.db[collection_name]
        
        # 构建查询条件
        query = {}
        if start_date or end_date:
            query['timestamp'] = {}
            if start_date:
                query['timestamp']['$gte'] = start_date
            if end_date:
                query['timestamp']['$lte'] = end_date
        
        # 聚合统计
        pipeline = [
            {'$match': query},
            {'$group': {
                '_id': None,
                'total_queries': {'$sum': 1},
                'avg_results': {'$avg': '$num_results'},
                'avg_recall': {'$avg': '$metrics.recall'},
                'avg_latency': {'$avg': '$metrics.latency_ms'}
            }}
        ]
        
        result = list(collection.aggregate(pipeline))
        
        if result:
            stats = result[0]
            stats.pop('_id')
            logger.info(f"查询统计: {stats}")
            return stats
        
        return {}
    
    def save_evaluation_results(
        self,
        results: Dict,
        collection_name: str = "evaluation_results"
    ):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            collection_name: 集合名称
        """
        collection = self.db[collection_name]
        
        results['timestamp'] = datetime.now()
        
        collection.insert_one(results)
        logger.info(f"✅ 评估结果已保存")
    
    def get_latest_evaluation(self, collection_name: str = "evaluation_results") -> Optional[Dict]:
        """
        获取最新的评估结果
        
        Args:
            collection_name: 集合名称
            
        Returns:
            最新评估结果
        """
        collection = self.db[collection_name]
        
        result = collection.find_one(
            {},
            {"_id": 0},
            sort=[("timestamp", -1)]
        )
        
        if result:
            logger.info(f"获取最新评估: {result.get('timestamp')}")
        
        return result
    
    def create_indexes(self):
        """创建常用索引以优化查询性能"""
        
        # chunks_metadata索引
        self.db.chunks_metadata.create_index("category")
        self.db.chunks_metadata.create_index("created_at")
        
        # query_logs索引
        self.db.query_logs.create_index("timestamp")
        self.db.query_logs.create_index([("timestamp", -1)])  # 降序索引
        
        # evaluation_results索引
        self.db.evaluation_results.create_index([("timestamp", -1)])
        
        logger.info("✅ 索引创建完成")
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息
        """
        collection = self.db[collection_name]
        
        stats = {
            'count': collection.count_documents({}),
            'indexes': [idx['name'] for idx in collection.list_indexes()],
            'size_mb': self.db.command("collStats", collection_name).get('size', 0) / (1024**2)
        }
        
        return stats
    
    def close(self):
        """关闭MongoDB连接"""
        self.client.close()
        logger.info("MongoDB连接已关闭")


if __name__ == "__main__":
    # 测试MongoDB存储
    print("=" * 70)
    print("📊 MongoDB存储测试")
    print("=" * 70)
    
    print("\n⚠️ 请先启动MongoDB服务:")
    print("docker run -d -p 27017:27017 --name mongodb \\")
    print("  -e MONGO_INITDB_ROOT_USERNAME=admin \\")
    print("  -e MONGO_INITDB_ROOT_PASSWORD=admin123 \\")
    print("  mongo:latest")
    
    try:
        storage = MongoDBStorage()
        
        # 创建索引
        print("\n📑 创建索引...")
        storage.create_indexes()
        
        # 测试保存元数据
        test_chunks = [
            {
                'id': 'test_001',
                'text': 'Type 2 diabetes is a chronic condition...',
                'category': 'diabetes',
                'pmid': '12345678'
            },
            {
                'id': 'test_002',
                'text': 'Cardiovascular disease affects millions...',
                'category': 'cardiovascular',
                'pmid': '87654321'
            }
        ]
        
        print(f"\n💾 保存测试数据: {len(test_chunks)} 条")
        storage.save_chunks_metadata(test_chunks, "test_chunks")
        
        # 测试查询
        print("\n🔍 按类别查询:")
        results = storage.query_chunks_by_category("diabetes", collection_name="test_chunks")
        for r in results:
            print(f"  {r['id']}: {r['text'][:50]}...")
        
        # 测试查询日志
        print("\n📝 记录查询日志:")
        storage.log_query(
            "What is diabetes?",
            results,
            {'recall': 0.85, 'latency_ms': 120.5}
        )
        
        # 获取统计
        print("\n📊 查询统计:")
        stats = storage.get_query_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 获取集合统计
        print("\n📈 集合统计:")
        for coll in ['test_chunks', 'query_logs']:
            coll_stats = storage.get_collection_stats(coll)
            print(f"  {coll}:")
            for key, value in coll_stats.items():
                print(f"    {key}: {value}")
        
        storage.close()
        
        print("\n✅ MongoDB存储测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示: 请确保MongoDB服务已启动")
