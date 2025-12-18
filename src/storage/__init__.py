# -*- coding: utf-8 -*-
"""
存储模块
- mongodb_storage: MongoDB文档存储
- minio_storage: MinIO对象存储
"""

from .mongodb_storage import MongoDBStorage
from .minio_storage import MinIOStorage

__all__ = ['MongoDBStorage', 'MinIOStorage']
