# -*- coding: utf-8 -*-
"""
统一高性能向量化模块
整合：大批次 + 多GPU + 多进程并行 + PySpark分布式 + 断点续传
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("unified_embedder", LOGS_DIR / "embedding.log")


@dataclass
class EmbeddingConfig:
    """向量化配置"""
    model_name: str = EMBEDDING_MODEL_NAME
    batch_size: int = 128
    checkpoint_interval: int = 50000
    num_workers: int = 4
    use_spark: bool = False
    spark_partitions: int = 16


@dataclass 
class EmbeddingResult:
    """向量化结果"""
    total_count: int
    processed_count: int
    elapsed_seconds: float
    throughput: float
    output_file: str
    resumed_from: int = 0


class CheckpointManager:
    """断点管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "embedding_progress.json"
        self.checkpoint_file = output_dir / "embedding_checkpoint.npy"
    
    def load_checkpoint(self) -> Tuple[int, Optional[np.ndarray]]:
        """加载断点，返回 (已处理数量, 已有向量)"""
        if not self.progress_file.exists():
            return 0, None
        
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        
        processed = progress.get('processed_count', 0)
        
        if self.checkpoint_file.exists() and processed > 0:
            embeddings = np.load(self.checkpoint_file)
            logger.info(f"从断点恢复: 已处理 {processed:,} 条, 向量形状 {embeddings.shape}")
            return processed, embeddings
        
        return 0, None
    
    def save_checkpoint(self, processed: int, total: int, embeddings: np.ndarray):
        """保存断点"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(self.checkpoint_file, embeddings)
        with open(self.progress_file, 'w') as f:
            json.dump({'processed_count': processed, 'total': total}, f)
        
        logger.info(f"检查点已保存: {processed:,}/{total:,}")
    
    def clear_checkpoint(self):
        """清理断点文件"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.progress_file.exists():
            self.progress_file.unlink()


class GPUEmbedder:
    """GPU向量化器 - 支持多GPU和大批次"""
    
    def __init__(self, config: EmbeddingConfig, device: str = None):
        import os
        import torch
        from sentence_transformers import SentenceTransformer
        
        # 设置离线模式，避免网络请求
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        self.config = config
        
        # 设备检测
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"检测到 {self.gpu_count} 个GPU")
        else:
            self.device = "cpu"
            self.gpu_count = 0
        
        # 加载模型 - 使用本地缓存
        cache_dir = str(EMBEDDING_MODEL_DIR)
        logger.info(f"从本地加载模型: {config.model_name}, 缓存目录: {cache_dir}")
        self.model = SentenceTransformer(config.model_name, cache_folder=cache_dir, device=self.device, local_files_only=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"模型: {config.model_name}, 维度: {self.embedding_dim}, 设备: {self.device}")
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """编码文本"""
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )


# ============== 多进程工作函数 ==============

_worker_model = None
_worker_config = None

def _init_worker(model_name: str, model_dir: str, device: str, batch_size: int):
    """初始化工作进程"""
    global _worker_model, _worker_config
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from sentence_transformers import SentenceTransformer
    _worker_model = SentenceTransformer(model_name, cache_folder=model_dir, device=device)
    _worker_config = {'batch_size': batch_size}


def _encode_chunk(texts: List[str]) -> np.ndarray:
    """工作进程编码"""
    global _worker_model, _worker_config
    return _worker_model.encode(
        texts,
        batch_size=_worker_config['batch_size'],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


class MultiProcessEngine:
    """多进程处理引擎"""
    
    def __init__(self, config: EmbeddingConfig):
        import torch
        
        self.config = config
        self.model_dir = str(EMBEDDING_MODEL_DIR)
        
        # GPU配置
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.num_workers = min(config.num_workers, self.gpu_count) if self.gpu_count > 1 else config.num_workers
        else:
            self.gpu_count = 0
            self.num_workers = config.num_workers
        
        logger.info(f"多进程引擎: {self.num_workers} 工作进程, {self.gpu_count} GPU")
    
    def process(self, texts: List[str], start_idx: int = 0, 
                existing_embeddings: np.ndarray = None,
                output_dir: Path = None) -> Tuple[np.ndarray, int]:
        """
        多进程处理
        
        Returns:
            (embeddings, processed_count)
        """
        total = len(texts)
        texts_to_process = texts[start_idx:]
        
        if not texts_to_process:
            return existing_embeddings, start_idx
        
        logger.info(f"多进程处理: {len(texts_to_process):,} 条 (从 {start_idx:,} 开始)")
        
        # 分块
        chunk_size = max(10000, len(texts_to_process) // (self.num_workers * 4))
        chunks = []
        for i in range(0, len(texts_to_process), chunk_size):
            chunks.append(texts_to_process[i:i + chunk_size])
        
        logger.info(f"分割为 {len(chunks)} 个块, 每块约 {chunk_size:,} 条")
        
        # 确定设备
        device = f"cuda:0" if self.gpu_count > 0 else "cpu"
        
        # 多进程执行
        results = [None] * len(chunks)
        checkpoint_mgr = CheckpointManager(output_dir) if output_dir else None
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_worker,
            initargs=(self.config.model_name, self.model_dir, device, self.config.batch_size)
        ) as executor:
            
            futures = {executor.submit(_encode_chunk, chunk): idx for idx, chunk in enumerate(chunks)}
            
            processed_chunks = 0
            with tqdm(total=len(chunks), desc="多进程向量化") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"块 {idx} 失败: {e}")
                        # 获取embedding维度
                        dim = 512  # 默认
                        if existing_embeddings is not None:
                            dim = existing_embeddings.shape[1]
                        results[idx] = np.zeros((len(chunks[idx]), dim))
                    
                    processed_chunks += 1
                    pbar.update(1)
                    
                    # 定期保存检查点
                    if checkpoint_mgr and processed_chunks % 10 == 0:
                        completed_results = [r for r in results if r is not None]
                        if completed_results:
                            partial = np.vstack(completed_results)
                            if existing_embeddings is not None:
                                partial = np.vstack([existing_embeddings, partial])
                            processed = start_idx + sum(len(chunks[i]) for i in range(len(chunks)) if results[i] is not None)
                            checkpoint_mgr.save_checkpoint(processed, total, partial)
        
        # 合并结果
        new_embeddings = np.vstack(results)
        
        if existing_embeddings is not None:
            final_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            final_embeddings = new_embeddings
        
        return final_embeddings, total


class SparkEngine:
    """PySpark分布式处理引擎"""
    
    def __init__(self, config: EmbeddingConfig, spark_master: str = None):
        from pyspark.sql import SparkSession
        
        self.config = config
        master = spark_master or "local[*]"
        
        self.spark = (SparkSession.builder
            .appName("UnifiedEmbedder-Spark")
            .master(master)
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark引擎初始化: {master}")
    
    def process_parquet(self, input_file: Path, output_dir: Path,
                       start_idx: int = 0) -> Tuple[str, int]:
        """
        Spark处理Parquet文件
        
        Returns:
            (output_parquet_path, total_count)
        """
        from pyspark.sql.functions import pandas_udf, col, monotonically_increasing_id, row_number
        from pyspark.sql.types import ArrayType, FloatType
        from pyspark.sql.window import Window
        
        model_name = self.config.model_name
        model_dir = str(EMBEDDING_MODEL_DIR)
        batch_size = self.config.batch_size
        
        logger.info("=" * 60)
        logger.info("Spark分布式向量化")
        logger.info(f"输入: {input_file}")
        logger.info("=" * 60)
        
        # 读取数据
        df = self.spark.read.parquet(str(input_file))
        total_count = df.count()
        logger.info(f"总数据量: {total_count:,}")
        
        # 添加行号用于断点续传
        df = df.withColumn("_row_id", monotonically_increasing_id())
        
        # 从断点继续
        if start_idx > 0:
            df = df.filter(col("_row_id") >= start_idx)
            remaining = df.count()
            logger.info(f"从断点 {start_idx:,} 继续, 剩余 {remaining:,} 条")
        
        # 确定文本列
        text_col = 'chunk_text' if 'chunk_text' in df.columns else 'content'
        
        # 分区优化
        num_partitions = max(self.config.spark_partitions, total_count // 100000)
        df = df.repartition(num_partitions)
        logger.info(f"分区数: {num_partitions}")
        
        # Pandas UDF
        @pandas_udf(ArrayType(FloatType()))
        def embed_udf(texts: pd.Series) -> pd.Series:
            from sentence_transformers import SentenceTransformer
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            model = SentenceTransformer(model_name, cache_folder=model_dir, local_files_only=True)
            embeddings = model.encode(
                texts.tolist(),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return pd.Series([emb.tolist() for emb in embeddings])
        
        # 执行向量化
        logger.info("开始Spark向量化...")
        df_embedded = df.withColumn("embedding", embed_udf(col(text_col)))
        
        # 保存
        output_dir.mkdir(parents=True, exist_ok=True)
        output_parquet = output_dir / "embeddings_spark.parquet"
        
        columns_to_save = ["_row_id", "pmid", "chunk_id", "embedding"]
        if "topic" in df.columns:
            columns_to_save.append("topic")
        
        df_embedded.select(*columns_to_save).write.mode("overwrite").parquet(str(output_parquet))
        
        logger.info(f"Spark输出: {output_parquet}")
        return str(output_parquet), total_count
    
    def stop(self):
        if self.spark:
            self.spark.stop()
            logger.info("Spark已停止")


class SingleGPUEngine:
    """单GPU处理引擎 - 带断点续传"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedder = GPUEmbedder(config)
    
    def process(self, texts: List[str], output_dir: Path,
               start_idx: int = 0, existing_embeddings: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """
        单GPU处理，支持断点续传
        """
        total = len(texts)
        checkpoint_mgr = CheckpointManager(output_dir)
        
        logger.info(f"单GPU处理: {total:,} 条, 从 {start_idx:,} 开始")
        
        all_embeddings = [existing_embeddings] if existing_embeddings is not None else []
        batch_size = self.config.batch_size
        
        for i in tqdm(range(start_idx, total, batch_size), desc="向量化进度"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.embedder.encode(batch_texts)
            all_embeddings.append(embeddings)
            
            # 保存检查点
            processed = i + len(batch_texts)
            if processed % self.config.checkpoint_interval == 0 or processed >= total:
                merged = np.vstack(all_embeddings)
                checkpoint_mgr.save_checkpoint(processed, total, merged)
        
        return np.vstack(all_embeddings), total


class UnifiedEmbedder:
    """
    统一向量化器
    自动选择最优策略：单GPU / 多进程 / Spark
    支持断点续传
    """
    
    def __init__(self, config: EmbeddingConfig = None, strategy: str = "auto"):
        """
        初始化
        
        Args:
            config: 向量化配置
            strategy: 策略 ("auto", "single", "multiprocess", "spark")
        """
        self.config = config or EmbeddingConfig()
        self.strategy = strategy
        
        logger.info("=" * 60)
        logger.info("统一向量化器初始化")
        logger.info(f"策略: {strategy}")
        logger.info(f"配置: batch_size={self.config.batch_size}, workers={self.config.num_workers}")
        logger.info("=" * 60)
    
    def process_parquet(self, input_file: Path, output_dir: Path,
                       resume: bool = True) -> EmbeddingResult:
        """
        处理Parquet文件
        
        Args:
            input_file: 输入Parquet文件
            output_dir: 输出目录
            resume: 是否从断点恢复
        """
        import torch
        
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_mgr = CheckpointManager(output_dir)
        
        # 加载断点
        start_idx = 0
        existing_embeddings = None
        if resume:
            start_idx, existing_embeddings = checkpoint_mgr.load_checkpoint()
        
        # 加载数据
        logger.info(f"加载数据: {input_file}")
        df = pd.read_parquet(input_file)
        total = len(df)
        logger.info(f"总数据量: {total:,}, 已处理: {start_idx:,}")
        
        if start_idx >= total:
            logger.info("数据已全部处理完成")
            return EmbeddingResult(
                total_count=total,
                processed_count=total,
                elapsed_seconds=0,
                throughput=0,
                output_file=str(output_dir / "embeddings_final.npy"),
                resumed_from=start_idx
            )
        
        text_col = 'chunk_text' if 'chunk_text' in df.columns else 'content'
        texts = df[text_col].fillna('').tolist()
        
        # 选择策略
        strategy = self._select_strategy(total, start_idx)
        logger.info(f"使用策略: {strategy}")
        
        start_time = time.time()
        
        # 执行向量化
        if strategy == "spark":
            engine = SparkEngine(self.config)
            try:
                output_path, processed = engine.process_parquet(input_file, output_dir, start_idx)
                # Spark输出为Parquet，需要转换
                result_file = output_path
            finally:
                engine.stop()
        
        elif strategy == "multiprocess":
            engine = MultiProcessEngine(self.config)
            embeddings, processed = engine.process(texts, start_idx, existing_embeddings, output_dir)
            
            # 保存最终结果
            result_file = output_dir / "embeddings_final.npy"
            np.save(result_file, embeddings)
            checkpoint_mgr.clear_checkpoint()
        
        else:  # single
            engine = SingleGPUEngine(self.config)
            embeddings, processed = engine.process(texts, output_dir, start_idx, existing_embeddings)
            
            # 保存最终结果
            result_file = output_dir / "embeddings_final.npy"
            np.save(result_file, embeddings)
            checkpoint_mgr.clear_checkpoint()
        
        elapsed = time.time() - start_time
        new_processed = total - start_idx
        throughput = new_processed / elapsed if elapsed > 0 else 0
        
        # 保存元数据
        self._save_metadata(output_dir, total, elapsed, throughput, str(result_file))
        
        logger.info("=" * 60)
        logger.info("向量化完成!")
        logger.info(f"  总数: {total:,}")
        logger.info(f"  本次处理: {new_processed:,}")
        logger.info(f"  耗时: {elapsed:.1f}s")
        logger.info(f"  速度: {throughput:.0f}/s")
        logger.info(f"  输出: {result_file}")
        logger.info("=" * 60)
        
        return EmbeddingResult(
            total_count=total,
            processed_count=total,
            elapsed_seconds=elapsed,
            throughput=throughput,
            output_file=str(result_file),
            resumed_from=start_idx
        )
    
    def _select_strategy(self, total: int, start_idx: int) -> str:
        """自动选择策略"""
        import torch
        
        if self.strategy != "auto":
            return self.strategy
        
        remaining = total - start_idx
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # 小数据集或接近完成时用单GPU
        if remaining < 500000:
            return "single"
        
        # 多GPU用多进程
        if gpu_count > 1:
            return "multiprocess"
        
        # 大数据集且配置了Spark
        if remaining > 2000000 and self.config.use_spark:
            return "spark"
        
        return "single"
    
    def _save_metadata(self, output_dir: Path, total: int, elapsed: float, 
                      throughput: float, output_file: str):
        """保存元数据"""
        meta = {
            'total_count': total,
            'elapsed_seconds': elapsed,
            'throughput': throughput,
            'output_file': output_file,
            'model_name': self.config.model_name,
            'batch_size': self.config.batch_size,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        meta_file = output_dir / "embedding_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一高性能向量化")
    parser.add_argument("--strategy", choices=["auto", "single", "multiprocess", "spark"],
                       default="auto", help="处理策略")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小")
    parser.add_argument("--workers", type=int, default=4, help="工作进程数")
    parser.add_argument("--no-resume", action="store_true", help="不从断点恢复")
    parser.add_argument("--use-spark", action="store_true", help="启用Spark")
    args = parser.parse_args()
    
    config = EmbeddingConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_spark=args.use_spark
    )
    
    embedder = UnifiedEmbedder(config=config, strategy=args.strategy)
    
    input_file = PROCESSED_DATA_DIR / "parquet" / "medical_chunks.parquet"
    output_dir = EMBEDDING_DATA_DIR
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    result = embedder.process_parquet(input_file, output_dir, resume=not args.no_resume)
    
    logger.info(f"\n最终结果: {asdict(result)}")


if __name__ == "__main__":
    main()
