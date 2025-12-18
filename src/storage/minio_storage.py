#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinIO对象存储模块
用于存储模型、向量数据库备份等大文件
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from minio import Minio
from minio.error import S3Error
from typing import Optional
import os
import pandas as pd
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("minio_storage", LOGS_DIR / "minio_storage.log")


class MinIOStorage:
    """MinIO对象存储管理器"""
    
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False
    ):
        """
        初始化MinIO客户端
        
        Args:
            endpoint: MinIO服务地址
            access_key: 访问密钥
            secret_key: 密钥
            secure: 是否使用HTTPS
        """
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            logger.info(f"✅ 连接MinIO成功: {endpoint}")
        except Exception as e:
            logger.error(f"❌ 连接MinIO失败: {e}")
            raise
    
    def create_bucket(self, bucket_name: str):
        """
        创建存储桶
        
        Args:
            bucket_name: 桶名称
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"✅ 创建bucket: {bucket_name}")
            else:
                logger.info(f"ℹ️ Bucket已存在: {bucket_name}")
        except S3Error as e:
            logger.error(f"❌ 创建bucket失败: {e}")
            raise
    
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
        content_type: Optional[str] = None
    ) -> bool:
        """
        上传文件到MinIO
        
        Args:
            bucket_name: 桶名称
            object_name: 对象名称 (存储路径)
            file_path: 本地文件路径
            content_type: 文件类型
            
        Returns:
            是否上传成功
        """
        try:
            file_size = file_path.stat().st_size / (1024**2)  # MB
            logger.info(f"开始上传: {file_path.name} ({file_size:.2f} MB)")
            
            self.client.fput_object(
                bucket_name,
                object_name,
                str(file_path),
                content_type=content_type
            )
            
            logger.info(f"✅ 上传成功: {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"❌ 上传失败: {e}")
            return False
    
    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path
    ) -> bool:
        """
        从MinIO下载文件
        
        Args:
            bucket_name: 桶名称
            object_name: 对象名称
            file_path: 本地保存路径
            
        Returns:
            是否下载成功
        """
        try:
            logger.info(f"开始下载: {bucket_name}/{object_name}")
            
            self.client.fget_object(
                bucket_name,
                object_name,
                str(file_path)
            )
            
            file_size = file_path.stat().st_size / (1024**2)
            logger.info(f"✅ 下载成功: {file_path} ({file_size:.2f} MB)")
            return True
            
        except S3Error as e:
            logger.error(f"❌ 下载失败: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> list:
        """
        列出bucket中的对象
        
        Args:
            bucket_name: 桶名称
            prefix: 对象前缀过滤
            
        Returns:
            对象列表
        """
        try:
            objects = self.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            result = []
            for obj in objects:
                result.append({
                    'name': obj.object_name,
                    'size_mb': obj.size / (1024**2),
                    'last_modified': obj.last_modified
                })
            
            logger.info(f"列出对象: {bucket_name}/{prefix}* - 共 {len(result)} 个")
            return result
            
        except S3Error as e:
            logger.error(f"❌ 列出对象失败: {e}")
            return []
    
    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """
        删除对象
        
        Args:
            bucket_name: 桶名称
            object_name: 对象名称
            
        Returns:
            是否删除成功
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"✅ 删除成功: {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"❌ 删除失败: {e}")
            return False
    
    def backup_models(self, bucket_name: str = "rag-models"):
        """
        备份模型文件到MinIO
        
        Args:
            bucket_name: 桶名称
        """
        self.create_bucket(bucket_name)
        
        # 备份嵌入模型
        models_dir = MODELS_DIR / "embedding"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    logger.info(f"备份模型: {model_dir.name}")
                    # 这里可以打包后上传
                    # 实际实现需要tar打包
        
        logger.info("模型备份完成")
    
    def backup_database(self, backup_path: Path, bucket_name: str = "rag-backups"):
        """
        备份向量数据库到MinIO
        
        Args:
            backup_path: 备份文件路径
            bucket_name: 桶名称
        """
        self.create_bucket(bucket_name)
        
        object_name = f"milvus_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        
        self.upload_file(
            bucket_name,
            object_name,
            backup_path,
            content_type="application/gzip"
        )


if __name__ == "__main__":
    # 测试MinIO存储
    print("=" * 70)
    print("☁️ MinIO对象存储测试")
    print("=" * 70)
    
    print("\n⚠️ 请先启动MinIO服务:")
    print("docker run -d -p 9000:9000 -p 9001:9001 \\")
    print("  -e MINIO_ROOT_USER=minioadmin \\")
    print("  -e MINIO_ROOT_PASSWORD=minioadmin \\")
    print("  minio/minio server /data --console-address ':9001'")
    
    try:
        storage = MinIOStorage()
        
        # 创建测试bucket
        storage.create_bucket("rag-test")
        
        # 测试上传
        test_file = PROCESSED_DATA_DIR / "medical_chunks.json"
        if test_file.exists():
            print(f"\n📤 测试上传: {test_file.name}")
            storage.upload_file(
                "rag-test",
                "test/medical_chunks.json",
                test_file,
                "application/json"
            )
            
            # 列出对象
            print("\n📋 列出对象:")
            objects = storage.list_objects("rag-test", "test/")
            for obj in objects:
                print(f"  {obj['name']} - {obj['size_mb']:.2f} MB")
        
        print("\n✅ MinIO存储测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示: 请确保MinIO服务已启动")
