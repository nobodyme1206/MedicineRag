# -*- coding: utf-8 -*-
"""
医学知识问答RAG系统 - 配置文件
支持环境变量和.env文件
"""

import os
from pathlib import Path

# 加载.env文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv未安装时跳过


def get_env(key: str, default=None, cast_type=str):
    """获取环境变量，支持类型转换"""
    value = os.getenv(key, default)
    if value is None:
        return default
    if cast_type == bool:
        return str(value).lower() in ('true', '1', 'yes')
    if cast_type == int:
        return int(value) if value else default
    return value


# ====================
# 基础路径配置
# ====================
BASE_DIR = Path(get_env("BASE_DIR", "d:/桌面/data-course"))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDING_DATA_DIR = DATA_DIR / "embeddings"
TEST_SET_DIR = DATA_DIR / "test_set"

MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = MODEL_DIR / "cache"
EMBEDDING_MODEL_DIR = MODEL_DIR / "embedding"

LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "evaluation"
PERFORMANCE_RESULTS_DIR = RESULTS_DIR / "performance"

# ====================
# LLM配置 - 硅基流动（从环境变量读取）
# ====================
SILICONFLOW_API_KEY = get_env("SILICONFLOW_API_KEY", "sk-ilrtyfwuqgseeoiigwzpbjbgxbfwmmvshmoogltfyqcuvbeq")
SILICONFLOW_BASE_URL = get_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = get_env("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# LLM参数
LLM_TEMPERATURE = float(get_env("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = get_env("LLM_MAX_TOKENS", 2048, int)
LLM_TOP_P = float(get_env("LLM_TOP_P", "0.9"))

# ====================
# Embedding模型配置
# ====================
EMBEDDING_MODEL_NAME = get_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIMENSION = 512
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_DEVICE = get_env("EMBEDDING_DEVICE", "cuda")

# Rerank模型配置
RERANK_MODEL_NAME = get_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = 20
USE_RERANK = get_env("USE_RERANK", True, bool)

# ====================
# 向量数据库配置 - Milvus
# ====================
MILVUS_HOST = get_env("MILVUS_HOST", "localhost")
MILVUS_PORT = get_env("MILVUS_PORT", 19530, int)
MILVUS_COLLECTION_NAME = "medical_knowledge_base"
MILVUS_INDEX_TYPE = "IVF_FLAT"
MILVUS_METRIC_TYPE = "COSINE"
MILVUS_NLIST = 1024
MILVUS_NPROBE = 16

# ====================
# 数据处理配置
# ====================
PUBMED_EMAIL = get_env("PUBMED_EMAIL", "data_course_project@example.com")
PUBMED_API_KEY = get_env("PUBMED_API_KEY", None)
PUBMED_MAX_RESULTS = 320000
PUBMED_BATCH_SIZE = 500
PUBMED_SEARCH_TERMS = [
    "diabetes", "cardiovascular disease", "cancer", "hypertension",
    "obesity", "covid-19", "mental health", "alzheimer",
    "stroke", "pneumonia", "asthma", "arthritis"
]

# 文档切分配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 100

# Spark配置
SPARK_MASTER = get_env("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = get_env("SPARK_DRIVER_MEMORY", "4g")
SPARK_EXECUTOR_MEMORY = get_env("SPARK_EXECUTOR_MEMORY", "4g")

# ====================
# RAG检索配置
# ====================
RETRIEVAL_TOP_K = 15
RERANK_TOP_K = 20
SIMILARITY_THRESHOLD = 0.4
VECTOR_SEARCH_WEIGHT = 0.6
BM25_SEARCH_WEIGHT = 0.4
USE_HYDE = get_env("USE_HYDE", True, bool)

# ====================
# Web界面配置
# ====================
GRADIO_PORT = get_env("GRADIO_PORT", 7860, int)
GRADIO_SHARE = get_env("GRADIO_SHARE", False, bool)
GRADIO_SERVER_NAME = "0.0.0.0"

# ====================
# MongoDB配置
# ====================
MONGODB_HOST = get_env("MONGODB_HOST", "localhost")
MONGODB_PORT = get_env("MONGODB_PORT", 27017, int)
MONGODB_DATABASE = get_env("MONGODB_DATABASE", "medical_rag")

# ====================
# MinIO配置
# ====================
MINIO_ENDPOINT = get_env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = get_env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = get_env("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = get_env("MINIO_SECURE", False, bool)
MINIO_BUCKET_MODELS = "rag-models"
MINIO_BUCKET_BACKUPS = "rag-backups"
MINIO_BUCKET_DATA = "rag-data"

# ====================
# Redis配置
# ====================
REDIS_HOST = get_env("REDIS_HOST", "localhost")
REDIS_PORT = get_env("REDIS_PORT", 6379, int)
REDIS_DB = get_env("REDIS_DB", 0, int)

# ====================
# 日志配置
# ====================
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5  # 保留5个备份

# ====================
# 实验配置
# ====================
EXPERIMENTS = {
    "baseline": {"name": "纯LLM（无检索）", "enable_retrieval": False},
    "rag_v1": {"name": "基础向量检索", "enable_retrieval": True, "enable_rerank": False, "enable_hybrid": False},
    "rag_v2": {"name": "混合检索", "enable_retrieval": True, "enable_rerank": False, "enable_hybrid": True},
    "rag_v3": {"name": "混合检索+重排序", "enable_retrieval": True, "enable_rerank": True, "enable_hybrid": True}
}

# ====================
# 数据统计目标
# ====================
TARGET_DATA_SIZE_GB = 15
TARGET_DOCUMENT_COUNT = 100000
TARGET_VECTOR_COUNT = 500000


def ensure_dirs():
    """确保所有必要的目录存在"""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDING_DATA_DIR,
        MODEL_DIR, EMBEDDING_MODEL_DIR, LOGS_DIR, RESULTS_DIR, EVAL_RESULTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ 所有目录已创建")
