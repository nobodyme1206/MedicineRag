# -*- coding: utf-8 -*-
"""
Airflow DAG - 医学RAG系统Pipeline编排
实现任务调度、依赖管理、失败重试、监控告警
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
import os
import sys

# 添加项目路径
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/opt/airflow/medical-rag')
sys.path.insert(0, PROJECT_DIR)


# ==================== 默认参数 ====================
default_args = {
    'owner': 'medical-rag',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=6),
}


# ==================== 任务函数 ====================

def collect_pubmed_data(**context):
    """
    任务1: PubMed数据采集
    - 支持断点续传
    - 采集结果发送到Kafka
    """
    from src.data_processing.pubmed_crawler import AsyncPubMedCrawler
    from src.messaging.kafka_producer import KafkaArticleProducer
    
    # 获取配置参数
    max_per_topic = int(Variable.get("pubmed_max_per_topic", default_var=5000))
    
    crawler = AsyncPubMedCrawler()
    producer = KafkaArticleProducer()
    
    # 爬取数据
    articles = crawler.crawl_all_topics()
    
    # 发送到Kafka（如果Kafka可用）
    if producer.producer:
        sent = producer.send_batch(articles)
        context['ti'].xcom_push(key='kafka_sent', value=sent)
    
    # 返回统计信息
    stats = {
        'total_articles': len(articles),
        'topics_completed': len(crawler.completed_topics),
        'kafka_sent': producer.get_stats()['sent']
    }
    
    producer.close()
    
    return stats


def process_data_spark(**context):
    """
    任务2: Spark数据处理
    - 数据清洗
    - 文本切分
    - 保存为Parquet
    """
    from src.data_processing.data_processor import DataProcessor
    from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    processor = DataProcessor(use_cluster=False)
    
    try:
        # 查找输入文件
        input_file = RAW_DATA_DIR / "pubmed_articles_all.json"
        if not input_file.exists():
            input_file = RAW_DATA_DIR / "pubmed_expanded.jsonl"
        
        if not input_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {input_file}")
        
        output_file = PROCESSED_DATA_DIR / "medical_chunks"
        
        # 处理数据
        processor.process_full_dataset(input_file, output_file)
        
        # 获取处理结果统计
        import pandas as pd
        df = pd.read_parquet(output_file.with_suffix('.parquet'))
        
        stats = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'chunk_count': len(df),
            'columns': list(df.columns)
        }
        
        return stats
        
    finally:
        processor.stop()


def generate_embeddings(**context):
    """
    任务3: 向量化
    - GPU加速
    - 批量处理
    """
    from src.embedding.embedder import TextEmbedder
    from config.config import PROCESSED_DATA_DIR, EMBEDDING_DATA_DIR
    
    embedder = TextEmbedder()
    
    input_file = PROCESSED_DATA_DIR / "medical_chunks.parquet"
    output_file = EMBEDDING_DATA_DIR / "medical_embeddings"
    
    embeddings, metadata = embedder.embed_dataset(input_file, output_file)
    
    return {
        'total_vectors': metadata['total_count'],
        'dimension': metadata['dimension'],
        'throughput': metadata['throughput']
    }


def update_milvus_index(**context):
    """
    任务4: 更新Milvus向量索引
    - 批量插入
    - 索引构建
    """
    from src.retrieval.milvus_manager import rebuild_database
    
    # 执行重建（支持断点续传）
    rebuild_database(resume=True, batch_size=256)
    
    return {'status': 'completed'}


def run_evaluation(**context):
    """
    任务5: 系统评估
    - RAG检索评估
    - 性能评估
    """
    from src.evaluation.unified_evaluator import UnifiedEvaluator
    
    evaluator = UnifiedEvaluator()
    results = evaluator.run_full_evaluation()
    
    # 提取关键指标
    metrics = {
        'overall_score': results.get('overall_score', 0),
        'precision_at_5': results.get('retrieval', {}).get('precision_at_5', 0),
        'mrr': results.get('retrieval', {}).get('mrr', 0),
        'hit_rate': results.get('retrieval', {}).get('hit_rate', 0)
    }
    
    # 检查是否达标
    if metrics['overall_score'] < 60:
        raise ValueError(f"评估分数过低: {metrics['overall_score']}/100")
    
    return metrics


def send_notification(**context):
    """
    任务6: 发送通知
    - 汇总Pipeline结果
    - 发送告警/报告
    """
    ti = context['ti']
    
    # 收集各任务结果
    collect_stats = ti.xcom_pull(task_ids='collect_pubmed')
    process_stats = ti.xcom_pull(task_ids='process_data')
    embed_stats = ti.xcom_pull(task_ids='generate_embeddings')
    eval_stats = ti.xcom_pull(task_ids='run_evaluation')
    
    report = f"""
    ========== Medical RAG Pipeline 完成报告 ==========
    
    📊 数据采集:
       - 文章数: {collect_stats.get('total_articles', 'N/A') if collect_stats else 'N/A'}
       - 主题数: {collect_stats.get('topics_completed', 'N/A') if collect_stats else 'N/A'}
    
    ⚙️ 数据处理:
       - Chunks数: {process_stats.get('chunk_count', 'N/A') if process_stats else 'N/A'}
    
    🔢 向量化:
       - 向量数: {embed_stats.get('total_vectors', 'N/A') if embed_stats else 'N/A'}
       - 吞吐量: {embed_stats.get('throughput', 'N/A'):.1f} 条/秒 if embed_stats else 'N/A'
    
    📈 评估结果:
       - 综合评分: {eval_stats.get('overall_score', 'N/A') if eval_stats else 'N/A'}/100
       - Precision@5: {eval_stats.get('precision_at_5', 'N/A') if eval_stats else 'N/A'}
       - MRR: {eval_stats.get('mrr', 'N/A') if eval_stats else 'N/A'}
    
    ================================================
    """
    
    print(report)
    
    # 这里可以集成钉钉/企业微信/邮件通知
    # send_dingtalk_message(report)
    # send_email(report)
    
    return {'report': report}


# ==================== DAG定义 ====================

# DAG 1: 每日增量更新
with DAG(
    dag_id='medical_rag_daily',
    default_args=default_args,
    description='医学RAG系统每日增量更新',
    schedule_interval='0 2 * * *',  # 每天凌晨2点
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['medical-rag', 'daily'],
    max_active_runs=1,
) as dag_daily:
    
    start = DummyOperator(task_id='start')
    
    collect_task = PythonOperator(
        task_id='collect_pubmed',
        python_callable=collect_pubmed_data,
        provide_context=True,
    )
    
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data_spark,
        provide_context=True,
    )
    
    embed_task = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        provide_context=True,
        # GPU任务可能需要更多资源
        pool='gpu_pool',
    )
    
    index_task = PythonOperator(
        task_id='update_milvus',
        python_callable=update_milvus_index,
        provide_context=True,
    )
    
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,  # 无论成功失败都执行
    )
    
    end = DummyOperator(task_id='end')
    
    # 定义依赖关系
    start >> collect_task >> process_task >> embed_task >> index_task >> notify_task >> end


# DAG 2: 每周完整评估
with DAG(
    dag_id='medical_rag_weekly_eval',
    default_args=default_args,
    description='医学RAG系统每周评估',
    schedule_interval='0 6 * * 0',  # 每周日早上6点
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['medical-rag', 'evaluation'],
    max_active_runs=1,
) as dag_weekly:
    
    eval_task = PythonOperator(
        task_id='run_evaluation',
        python_callable=run_evaluation,
        provide_context=True,
    )
    
    notify_eval = PythonOperator(
        task_id='notify_evaluation',
        python_callable=send_notification,
        provide_context=True,
    )
    
    eval_task >> notify_eval


# DAG 3: 手动触发的完整Pipeline
with DAG(
    dag_id='medical_rag_full_pipeline',
    default_args=default_args,
    description='医学RAG系统完整Pipeline（手动触发）',
    schedule_interval=None,  # 手动触发
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['medical-rag', 'manual'],
) as dag_full:
    
    start = DummyOperator(task_id='start')
    
    collect = PythonOperator(
        task_id='collect_pubmed',
        python_callable=collect_pubmed_data,
        provide_context=True,
    )
    
    process = PythonOperator(
        task_id='process_data',
        python_callable=process_data_spark,
        provide_context=True,
    )
    
    embed = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        provide_context=True,
    )
    
    index = PythonOperator(
        task_id='update_milvus',
        python_callable=update_milvus_index,
        provide_context=True,
    )
    
    evaluate = PythonOperator(
        task_id='run_evaluation',
        python_callable=run_evaluation,
        provide_context=True,
    )
    
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    
    end = DummyOperator(task_id='end')
    
    start >> collect >> process >> embed >> index >> evaluate >> notify >> end
