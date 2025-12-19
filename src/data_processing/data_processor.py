#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块 - 支持PySpark分布式处理和Pandas本地处理
功能：JSON/JSONL数据加载、清洗、文本切分、Parquet存储
"""

from __future__ import annotations

import gc
import json
import re
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, PERFORMANCE_RESULTS_DIR
)
from src.utils.logger import setup_logger
from src.utils.exceptions import DataProcessingError, handle_errors

logger = setup_logger("data_processor", LOGS_DIR / "data_processing.log")


@dataclass
class ProcessingResult:
    """处理结果"""
    total_articles: int
    total_chunks: int
    output_file: str
    elapsed_seconds: float
    
    def __str__(self) -> str:
        return (
            f"ProcessingResult(articles={self.total_articles:,}, "
            f"chunks={self.total_chunks:,}, time={self.elapsed_seconds:.1f}s)"
        )


@dataclass
class ChunkInfo:
    """Chunk信息（支持父子关联）"""
    text: str
    chunk_id: int
    parent_id: Optional[int] = None  # 父chunk ID
    children_ids: Optional[List[int]] = None  # 子chunk IDs
    section_type: str = "body"  # title, abstract, body, conclusion
    start_pos: int = 0
    end_pos: int = 0


class TextChunker:
    """文本切分器 - 增强版：支持语义切分和父子chunk关联"""
    
    # 段落分隔符模式
    PARAGRAPH_SEPARATORS = [
        r'\n\n+',  # 多个换行
        r'\n(?=[A-Z])',  # 换行后大写字母开头
        r'(?<=[.!?])\s+(?=[A-Z])',  # 句号后空格+大写
    ]
    
    # 章节标题模式
    SECTION_PATTERNS = [
        r'^(?:Introduction|Background|Methods?|Results?|Discussion|Conclusion|Abstract|Summary)',
        r'^(?:\d+\.?\s*)?(?:Introduction|Background|Methods?|Results?|Discussion|Conclusion)',
        r'^[A-Z][A-Z\s]+:',  # 全大写标题
    ]
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        overlap: int = CHUNK_OVERLAP,
        min_length: int = MIN_CHUNK_LENGTH,
        use_semantic: bool = True
    ) -> None:
        """
        初始化文本切分器
        
        Args:
            chunk_size: 切分块大小
            overlap: 重叠大小
            min_length: 最小块长度
            use_semantic: 是否使用语义切分
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_length = min_length
        self.use_semantic = use_semantic
        
        # 编译正则
        self._paragraph_re = re.compile('|'.join(self.PARAGRAPH_SEPARATORS))
        self._section_re = re.compile('|'.join(self.SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 先按双换行分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 简单的句子分割（保留句号）
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_section_type(self, text: str) -> str:
        """检测文本所属章节类型"""
        text_lower = text.lower()[:200]
        
        if 'abstract' in text_lower or text_lower.startswith('background'):
            return 'abstract'
        elif 'introduction' in text_lower:
            return 'introduction'
        elif 'method' in text_lower or 'material' in text_lower:
            return 'methods'
        elif 'result' in text_lower:
            return 'results'
        elif 'discussion' in text_lower:
            return 'discussion'
        elif 'conclusion' in text_lower or 'summary' in text_lower:
            return 'conclusion'
        else:
            return 'body'
    
    def chunk_semantic(self, text: str) -> List[ChunkInfo]:
        """
        语义切分：按段落和句子边界切分，保持语义完整性
        
        Args:
            text: 输入文本
            
        Returns:
            ChunkInfo列表
        """
        if not text or len(text) < self.min_length:
            return []
        
        chunks: List[ChunkInfo] = []
        chunk_id = 0
        
        # 1. 按段落分割
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            # 如果当前段落本身就超过chunk_size，需要进一步切分
            if len(para) > self.chunk_size:
                # 先保存当前累积的chunk
                if current_chunk and len(current_chunk) >= self.min_length:
                    section_type = self._detect_section_type(current_chunk)
                    chunks.append(ChunkInfo(
                        text=current_chunk.strip(),
                        chunk_id=chunk_id,
                        section_type=section_type,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    ))
                    chunk_id += 1
                    current_chunk = ""
                
                # 按句子切分长段落
                sentences = self._split_into_sentences(para)
                sent_chunk = ""
                sent_start = text.find(para)
                
                for sent in sentences:
                    if len(sent_chunk) + len(sent) + 1 <= self.chunk_size:
                        sent_chunk = f"{sent_chunk} {sent}".strip() if sent_chunk else sent
                    else:
                        if sent_chunk and len(sent_chunk) >= self.min_length:
                            section_type = self._detect_section_type(sent_chunk)
                            chunks.append(ChunkInfo(
                                text=sent_chunk,
                                chunk_id=chunk_id,
                                section_type=section_type,
                                start_pos=sent_start,
                                end_pos=sent_start + len(sent_chunk)
                            ))
                            chunk_id += 1
                        sent_chunk = sent
                
                # 保存最后的句子chunk
                if sent_chunk and len(sent_chunk) >= self.min_length:
                    section_type = self._detect_section_type(sent_chunk)
                    chunks.append(ChunkInfo(
                        text=sent_chunk,
                        chunk_id=chunk_id,
                        section_type=section_type,
                        start_pos=sent_start,
                        end_pos=sent_start + len(sent_chunk)
                    ))
                    chunk_id += 1
                
                current_start = text.find(para) + len(para)
            
            # 段落可以合并到当前chunk
            elif len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk = f"{current_chunk}\n\n{para}"
                else:
                    current_chunk = para
                    current_start = text.find(para)
            
            # 需要开始新chunk
            else:
                if current_chunk and len(current_chunk) >= self.min_length:
                    section_type = self._detect_section_type(current_chunk)
                    chunks.append(ChunkInfo(
                        text=current_chunk.strip(),
                        chunk_id=chunk_id,
                        section_type=section_type,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    ))
                    chunk_id += 1
                
                current_chunk = para
                current_start = text.find(para)
        
        # 保存最后的chunk
        if current_chunk and len(current_chunk) >= self.min_length:
            section_type = self._detect_section_type(current_chunk)
            chunks.append(ChunkInfo(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                section_type=section_type,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            ))
        
        return chunks
    
    def chunk_with_context(self, text: str, context_size: int = 100) -> List[Dict]:
        """
        带上下文的切分：每个chunk包含前后文摘要
        
        Args:
            text: 输入文本
            context_size: 上下文大小
            
        Returns:
            包含上下文的chunk字典列表
        """
        chunks = self.chunk_semantic(text) if self.use_semantic else self._chunk_sliding(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text if isinstance(chunk, ChunkInfo) else chunk
            
            # 获取前文
            prev_context = ""
            if i > 0:
                prev_chunk = chunks[i-1].text if isinstance(chunks[i-1], ChunkInfo) else chunks[i-1]
                prev_context = prev_chunk[-context_size:] if len(prev_chunk) > context_size else prev_chunk
            
            # 获取后文
            next_context = ""
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1].text if isinstance(chunks[i+1], ChunkInfo) else chunks[i+1]
                next_context = next_chunk[:context_size] if len(next_chunk) > context_size else next_chunk
            
            result.append({
                'text': chunk_text,
                'chunk_id': i,
                'prev_context': prev_context,
                'next_context': next_context,
                'section_type': chunk.section_type if isinstance(chunk, ChunkInfo) else 'body',
                'has_prev': i > 0,
                'has_next': i < len(chunks) - 1
            })
        
        return result
    
    def _chunk_sliding(self, text: str) -> List[str]:
        """滑动窗口切分（原始方法）"""
        if not text or len(text) < self.min_length:
            return []
        
        chunks: List[str] = []
        pos = 0
        step = self.chunk_size - self.overlap
        
        while pos < len(text):
            chunk = text[pos:pos + self.chunk_size].strip()
            if len(chunk) >= self.min_length:
                chunks.append(chunk)
            pos += step
        
        return chunks
    
    def chunk(self, text: str) -> List[str]:
        """
        切分文本（主接口）
        
        Args:
            text: 输入文本
            
        Returns:
            切分后的文本块列表
        """
        if self.use_semantic:
            chunk_infos = self.chunk_semantic(text)
            return [c.text for c in chunk_infos]
        else:
            return self._chunk_sliding(text)
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: Optional[int] = None, 
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        切分文本（兼容旧接口）
        
        Args:
            text: 输入文本
            chunk_size: 切分块大小（可选）
            overlap: 重叠大小（可选）
            
        Returns:
            切分后的文本块列表
        """
        original_size = self.chunk_size
        original_overlap = self.overlap
        
        if chunk_size:
            self.chunk_size = chunk_size
        if overlap:
            self.overlap = overlap
        
        result = self.chunk(text)
        
        self.chunk_size = original_size
        self.overlap = original_overlap
        
        return result


class SparkProcessor:
    """PySpark分布式数据处理器"""
    
    def __init__(self, use_cluster: bool = False) -> None:
        """
        初始化Spark环境
        
        Args:
            use_cluster: 是否使用Docker Spark集群
            
        Raises:
            DataProcessingError: Spark初始化失败
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            raise DataProcessingError("缺少PySpark: pip install pyspark", e)
        
        logger.info("初始化Spark环境...")
        master = "spark://localhost:7077" if use_cluster else SPARK_MASTER
        
        try:
            builder = (SparkSession.builder
                .appName("MedicalRAG-DataProcessing")
                .master(master)
                .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
                .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer"))
            
            if use_cluster:
                builder = (builder
                    .config("spark.executor.instances", "2")
                    .config("spark.executor.cores", "4"))
            
            self.spark = builder.getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            self.chunker = TextChunker()
            
            logger.info(f"Spark版本: {self.spark.version}, Master: {master}")
            
        except Exception as e:
            raise DataProcessingError("Spark初始化失败", e)
    
    @handle_errors(default_return=None, log_level="error")
    def load_data(self, input_file: Path):
        """加载JSON/JSONL数据"""
        logger.info(f"加载数据: {input_file}")
        df = self.spark.read.json(str(input_file))
        count = df.count()
        logger.info(f"原始数据量: {count:,} 条")
        return df
    
    def clean_data(self, df):
        """数据清洗：过滤缺失值、短文本、去重"""
        from pyspark.sql.functions import col, length
        
        logger.info("开始数据清洗...")
        
        df_clean = (df
            .filter(col("title").isNotNull())
            .filter(col("abstract").isNotNull())
            .filter(col("full_text").isNotNull())
            .filter(length(col("full_text")) >= MIN_CHUNK_LENGTH)
            .dropDuplicates(["pmid"]))
        
        count = df_clean.count()
        logger.info(f"清洗后: {count:,} 条")
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
        
        count = df_chunked.count()
        logger.info(f"切分后chunk数量: {count:,}")
        return df_chunked
    
    def save_data(self, df, output_path: Path) -> None:
        """保存处理后的数据"""
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
    
    def process_full_dataset(
        self, 
        input_file: Path, 
        output_file: Path
    ) -> None:
        """处理完整数据集"""
        start_time = time.time()
        
        df = self.load_data(input_file)
        if df is None:
            raise DataProcessingError(f"无法加载数据: {input_file}")
        
        df_clean = self.clean_data(df)
        df_chunked = self.process_and_chunk(df_clean)
        self.save_data(df_chunked, output_file)
        
        elapsed = time.time() - start_time
        logger.info(f"处理完成！总耗时: {elapsed:.2f} 秒")
    
    def stop(self) -> None:
        """停止Spark"""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            logger.info("Spark已停止")


class PandasProcessor:
    """Pandas本地数据处理器"""
    
    def __init__(self) -> None:
        self.chunker = TextChunker()
    
    def _process_article(self, article: Dict) -> List[Dict]:
        """
        处理单篇文章
        
        Args:
            article: 文章数据
            
        Returns:
            chunks列表
        """
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
            'id': f"{pmid}_{i}",
            'pmid': pmid,
            'chunk_id': f"{pmid}_{i}",
            'chunk_text': chunk,
            'title': title,
            'topic': topic,
            'content': chunk
        } for i, chunk in enumerate(chunks)]
    
    def process_json_file(
        self, 
        input_file: Path, 
        output_dir: Path
    ) -> ProcessingResult:
        """
        处理JSON文件
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            处理结果
            
        Raises:
            DataProcessingError: 处理失败
        """
        if not input_file.exists():
            raise DataProcessingError(f"输入文件不存在: {input_file}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_size_gb = input_file.stat().st_size / (1024**3)
        logger.info(f"输入文件: {input_file.name} ({file_size_gb:.2f} GB)")
        
        start_time = time.time()
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"JSON解析失败: {input_file}", e)
        
        total_articles = len(data)
        logger.info(f"加载完成: {total_articles:,} 篇文章")
        
        # 处理成chunks
        all_chunks: List[Dict] = []
        for article in tqdm(data, desc="处理进度", unit="篇"):
            all_chunks.extend(self._process_article(article))
        
        del data
        gc.collect()
        
        return self._save_results(all_chunks, total_articles, output_dir, start_time)
    
    def process_checkpoints(
        self, 
        checkpoint_dir: Path, 
        output_dir: Optional[Path] = None, 
        num_workers: int = 8
    ) -> Optional[ProcessingResult]:
        """
        多线程并行处理checkpoint文件
        
        Args:
            checkpoint_dir: checkpoint目录
            output_dir: 输出目录
            num_workers: 工作线程数
            
        Returns:
            处理结果
        """
        data_files = sorted(checkpoint_dir.glob("checkpoint_*.data.json"))
        if not data_files:
            logger.error(f"未找到checkpoint文件: {checkpoint_dir}")
            return None
        
        logger.info(f"找到 {len(data_files)} 个checkpoint文件")
        
        start_time = time.time()
        all_chunks: List[Dict] = []
        total_articles = 0
        
        tasks = [
            (f, f.stem.replace("checkpoint_", "").replace(".data", "")) 
            for f in data_files
        ]
        
        workers = min(num_workers, len(data_files))
        logger.info(f"使用 {workers} 个线程并行处理...")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_checkpoint_file, *task): task 
                for task in tasks
            }
            
            with tqdm(total=len(futures), desc="处理进度", unit="文件") as pbar:
                for future in as_completed(futures):
                    chunks, count = future.result()
                    all_chunks.extend(chunks)
                    total_articles += count
                    pbar.update(1)
                    pbar.set_postfix({'chunks': f'{len(all_chunks):,}'})
        
        gc.collect()
        
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR / "parquet"
        
        return self._save_results(all_chunks, total_articles, output_dir, start_time)
    
    @handle_errors(default_return=([], 0), log_level="warning")
    def _process_checkpoint_file(
        self, 
        data_file: Path, 
        topic_name: str
    ) -> Tuple[List[Dict], int]:
        """处理单个checkpoint文件"""
        chunks: List[Dict] = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        article_count = len(articles)
        for article in articles:
            article['topic'] = article.get('topic', topic_name)
            chunks.extend(self._process_article(article))
        
        del articles
        return chunks, article_count
    
    def _save_results(
        self, 
        chunks: List[Dict], 
        total_articles: int, 
        output_dir: Path, 
        start_time: float
    ) -> ProcessingResult:
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


# 为了兼容旧代码，保留DataProcessor别名
class DataProcessor(PandasProcessor):
    """DataProcessor别名（兼容旧代码）"""
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = CHUNK_SIZE, 
        overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """切分文本（兼容旧接口）"""
        return self.chunker.chunk_text(text, chunk_size, overlap)


def main() -> Optional[ProcessingResult]:
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
