# -*- coding: utf-8 -*-
"""
数据预处理模块 - 支持PySpark分布式处理和Pandas本地处理
功能：JSON/JSONL数据加载、清洗、文本切分、Parquet存储
"""

import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, PERFORMANCE_RESULTS_DIR
)
from src.utils.logger import setup_logger

logger = setup_logger("data_processor", LOGS_DIR / "data_processing.log")


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    total_articles: int
    total_chunks: int
    output_file: str
    elapsed_seconds: float


class TextChunker:
    """文本切分器"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, 
                 overlap: int = CHUNK_OVERLAP,
                 min_length: int = MIN_CHUNK_LENGTH):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_length = min_length
    
    def chunk(self, text: str) -> List[str]:
        """滑动窗口切分文本"""
        if not text or len(text) < self.min_length:
            return []
        
        chunks = []
        pos = 0
        step = self.chunk_size - self.overlap
        
        while pos < len(text):
            chunk = text[pos:pos + self.chunk_size].strip()
            if len(chunk) >= self.min_length:
                chunks.append(chunk)
            pos += step
        
        return chunks


class SparkProcessor:
    """PySpark分布式数据处理器"""
    
    def __init__(self, use_cluster: bool = False):
        """
        初始化Spark环境
        
        Args:
            use_cluster: 是否使用Docker Spark集群
        """
        from pyspark.sql import SparkSession
        
        logger.info("初始化Spark环境...")
        master = "spark://localhost:7077" if use_cluster else SPARK_MASTER
        
        builder = (SparkSession.builder
            .appName("MedicalRAG-DataProcessing")
            .master(master)
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer"))
        
        if use_cluster:
            builder = builder.config("spark.executor.instances", "2").config("spark.executor.cores", "4")
        
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.chunker = TextChunker()
        
        logger.info(f"Spark版本: {self.spark.version}, Master: {master}")
    
    def load_data(self, input_file: Path):
        """加载JSON/JSONL数据"""
        logger.info(f"加载数据: {input_file}")
        df = self.spark.read.json(str(input_file))
        logger.info(f"原始数据量: {df.count():,} 条")
        return df
    
    def clean_data(self, df):
        """数据清洗：过滤缺失值、短文本、去重"""
        from pyspark.sql.functions import col, length
        
        logger.info("开始数据清洗...")
        
        df_clean = (df
            .filter(col("title").isNotNull() & col("abstract").isNotNull() & col("full_text").isNotNull())
            .filter(length(col("full_text")) >= MIN_CHUNK_LENGTH)
            .dropDuplicates(["pmid"]))
        
        logger.info(f"清洗后: {df_clean.count():,} 条")
        return df_clean
    
    def process_and_chunk(self, df):
        """处理并切分文档"""
        from pyspark.sql.functions import col, explode, monotonically_increasing_id, udf
        from pyspark.sql.types import ArrayType, StringType
        
        logger.info("开始文档切分...")
        
        chunk_udf = udf(lambda text: self.chunker.chunk(text), ArrayType(StringType()))
        
        df_chunked = (df
            .withColumn("chunks", chunk_udf(col("full_text")))
            .select(
                col("pmid"), col("title"), col("authors"), col("pub_date"),
                col("keywords"), col("mesh_terms"), col("topic"),
                explode(col("chunks")).alias("chunk_text"))
            .withColumn("chunk_id", monotonically_increasing_id()))
        
        logger.info(f"切分后chunk数量: {df_chunked.count():,}")
        return df_chunked
    
    def save_data(self, df, output_path: Path):
        """保存处理后的数据为Parquet和JSON样例"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        parquet_path = output_path.with_suffix('.parquet')
        df.write.mode("overwrite").parquet(str(parquet_path))
        logger.info(f"已保存Parquet: {parquet_path}")
        
        # 保存JSON样例
        sample_data = df.limit(1000).toPandas().to_dict('records')
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存JSON样例: {json_path}")
    
    def process_full_dataset(self, input_file: Path, output_file: Path):
        """处理完整数据集"""
        start_time = time.time()
        
        df = self.load_data(input_file)
        df_clean = self.clean_data(df)
        df_chunked = self.process_and_chunk(df_clean)
        self.save_data(df_chunked, output_file)
        
        logger.info(f"处理完成！总耗时: {time.time() - start_time:.2f} 秒")
    
    def stop(self):
        """停止Spark"""
        self.spark.stop()
        logger.info("Spark已停止")


class PandasProcessor:
    """Pandas本地数据处理器（备选方案）"""
    
    def __init__(self):
        self.chunker = TextChunker()
    
    def _process_article(self, article: Dict) -> List[Dict]:
        """处理单篇文章，返回chunks列表"""
        pmid = article.get('pmid', '')
        title = article.get('title', '')
        abstract = article.get('abstract', '')
        full_text = article.get('full_text', '')
        topic = article.get('topic', '')
        
        text = full_text if full_text else f"{title}\n\n{abstract}"
        if len(text) < MIN_CHUNK_LENGTH:
            return []
        
        chunks = self.chunker.chunk(text)
        return [{
            'pmid': pmid,
            'chunk_id': f"{pmid}_{i}",
            'chunk_text': chunk,
            'title': title,
            'topic': topic,
            'content': chunk
        } for i, chunk in enumerate(chunks)]
    
    def process_json_file(self, input_file: Path, output_dir: Path) -> ProcessingResult:
        """处理JSON文件"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_size_gb = input_file.stat().st_size / (1024**3)
        logger.info(f"输入文件: {input_file.name} ({file_size_gb:.2f} GB)")
        
        start_time = time.time()
        
        # 加载数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_articles = len(data)
        logger.info(f"加载完成: {total_articles:,} 篇文章")
        
        # 处理成chunks
        all_chunks = []
        for article in tqdm(data, desc="处理进度", unit="篇"):
            all_chunks.extend(self._process_article(article))
        
        del data
        gc.collect()
        
        # 保存结果
        return self._save_results(all_chunks, total_articles, output_dir, start_time)
    
    def process_checkpoints(self, checkpoint_dir: Path, output_dir: Path = None, 
                           num_workers: int = 8) -> Optional[ProcessingResult]:
        """多线程并行处理checkpoint文件"""
        data_files = sorted(checkpoint_dir.glob("checkpoint_*.data.json"))
        if not data_files:
            logger.error(f"未找到checkpoint文件: {checkpoint_dir}")
            return None
        
        logger.info(f"找到 {len(data_files)} 个checkpoint文件")
        
        start_time = time.time()
        all_chunks = []
        total_articles = 0
        
        # 准备任务
        tasks = [(f, f.stem.replace("checkpoint_", "").replace(".data", "")) for f in data_files]
        
        # 多线程并行处理
        workers = min(num_workers, len(data_files))
        logger.info(f"使用 {workers} 个线程并行处理...")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._process_checkpoint_file, *task): task for task in tasks}
            
            with tqdm(total=len(futures), desc="处理进度", unit="文件") as pbar:
                for future in as_completed(futures):
                    chunks, count = future.result()
                    all_chunks.extend(chunks)
                    total_articles += count
                    pbar.update(1)
                    pbar.set_postfix({'chunks': f'{len(all_chunks):,}'})
        
        gc.collect()
        
        # 保存结果
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR / "parquet"
        
        return self._save_results(all_chunks, total_articles, output_dir, start_time)
    
    def _process_checkpoint_file(self, data_file: Path, topic_name: str) -> Tuple[List[Dict], int]:
        """处理单个checkpoint文件"""
        chunks = []
        article_count = 0
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            article_count = len(articles)
            for article in articles:
                article['topic'] = article.get('topic', topic_name)
                chunks.extend(self._process_article(article))
            
            del articles
        except Exception as e:
            logger.warning(f"处理文件失败 {data_file.name}: {e}")
        
        return chunks, article_count
    
    def _save_results(self, chunks: List[Dict], total_articles: int, 
                      output_dir: Path, start_time: float) -> ProcessingResult:
        """保存处理结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("保存Parquet文件...")
        df = pd.DataFrame(chunks)
        parquet_file = output_dir / "medical_chunks.parquet"
        df.to_parquet(parquet_file, index=False, compression='snappy')
        
        # 保存样例JSON
        sample_file = output_dir.parent / "medical_chunks.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(chunks[:1000], f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        file_size_mb = parquet_file.stat().st_size / (1024**2)
        
        logger.info("=" * 60)
        logger.info("数据处理完成!")
        logger.info(f"  总文章数: {total_articles:,}")
        logger.info(f"  总chunks数: {len(chunks):,}")
        logger.info(f"  输出文件: {parquet_file}")
        logger.info(f"  文件大小: {file_size_mb:.2f} MB")
        logger.info(f"  总耗时: {elapsed:.1f} 秒")
        logger.info(f"  处理速度: {total_articles/elapsed:.0f} 文章/秒")
        
        return ProcessingResult(
            total_articles=total_articles,
            total_chunks=len(chunks),
            output_file=str(parquet_file),
            elapsed_seconds=elapsed
        )


# ============== 工具函数 ==============

def convert_json_to_jsonl(json_file: Path, jsonl_file: Path) -> int:
    """将大JSON文件转换为JSONL格式（Spark需要）"""
    logger.info(f"转换JSON → JSONL: {json_file.name}")
    file_size_gb = json_file.stat().st_size / (1024**3)
    logger.info(f"文件大小: {file_size_gb:.2f} GB")
    
    start_time = time.time()
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    logger.info(f"加载完成: {total:,} 条记录 (耗时: {time.time() - start_time:.1f}秒)")
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="转换进度", unit="条"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    del data
    gc.collect()
    
    logger.info(f"转换完成: {jsonl_file.name}")
    return total


def compare_performance(input_file: Path) -> Dict:
    """性能对比：单机 vs Spark"""
    logger.info("=" * 50)
    logger.info("性能对比实验")
    logger.info("=" * 50)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_size = min(10000, len(data))
    sample_data = data[:sample_size]
    chunker = TextChunker()
    
    # 单机处理
    logger.info(f"\n1. 单机处理 {sample_size} 条数据...")
    start_time = time.time()
    single_chunks = []
    for article in sample_data:
        single_chunks.extend(chunker.chunk(article.get('full_text', '')))
    single_time = time.time() - start_time
    logger.info(f"   处理时间: {single_time:.2f} 秒, chunks: {len(single_chunks):,}")
    
    # Spark分布式处理
    logger.info(f"\n2. Spark分布式处理 {sample_size} 条数据...")
    temp_file = PROCESSED_DATA_DIR / "temp_sample.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False)
    
    start_time = time.time()
    processor = SparkProcessor()
    df = processor.load_data(temp_file)
    df_clean = processor.clean_data(df)
    df_chunked = processor.process_and_chunk(df_clean)
    spark_chunk_count = df_chunked.count()
    spark_time = time.time() - start_time
    processor.stop()
    temp_file.unlink()
    
    logger.info(f"   处理时间: {spark_time:.2f} 秒, chunks: {spark_chunk_count:,}")
    
    # 对比结果
    speedup = single_time / spark_time
    logger.info(f"\n加速比: {speedup:.2f}x")
    
    result = {
        "sample_size": sample_size,
        "single_machine": {"time_seconds": single_time, "chunks": len(single_chunks)},
        "spark_distributed": {"time_seconds": spark_time, "chunks": spark_chunk_count},
        "speedup": speedup
    }
    
    result_file = PERFORMANCE_RESULTS_DIR / "processing_comparison.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    return result


def main():
    """主函数"""
    checkpoint_dir = RAW_DATA_DIR / "checkpoints"
    processor = PandasProcessor()
    
    # 优先使用checkpoints目录
    if checkpoint_dir.exists():
        data_files = list(checkpoint_dir.glob("checkpoint_*.data.json"))
        if data_files:
            logger.info(f"使用checkpoints目录: {len(data_files)} 个文件")
            return processor.process_checkpoints(checkpoint_dir)
    
    # 备选：使用大JSON文件
    input_file = RAW_DATA_DIR / "pubmed_articles_all.json"
    output_dir = PROCESSED_DATA_DIR / "parquet"
    
    if not input_file.exists():
        logger.error("数据文件不存在，请先运行 pubmed_crawler.py 采集数据")
        return None
    
    return processor.process_json_file(input_file, output_dir)


if __name__ == "__main__":
    main()
