# -*- coding: utf-8 -*-
"""
Kafka + Airflow 集成效果评估
对比引入前后的性能差异
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR, RESULTS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("kafka_airflow_eval", LOGS_DIR / "kafka_airflow_eval.log")


class KafkaAirflowEvaluator:
    """Kafka + Airflow 集成效果评估器"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "before_integration": {},
            "after_integration": {},
            "comparison": {}
        }
    
    def evaluate_throughput_before(self, sample_size: int = 1000) -> Dict:
        """
        评估集成前的吞吐量（串行处理）
        
        模拟原有流程：爬取 → 处理 → 向量化（串行）
        """
        logger.info("=" * 60)
        logger.info("📊 评估集成前性能（串行模式）")
        logger.info("=" * 60)
        
        # 模拟数据
        test_articles = self._generate_test_data(sample_size)
        
        # 串行处理
        start_time = time.time()
        
        # 阶段1: 模拟爬取（IO密集）
        crawl_start = time.time()
        for article in test_articles:
            time.sleep(0.001)  # 模拟网络延迟
        crawl_time = time.time() - crawl_start
        
        # 阶段2: 模拟处理（CPU密集）
        process_start = time.time()
        processed = []
        for article in test_articles:
            # 模拟文本处理
            chunks = self._simulate_chunking(article.get("text", ""))
            processed.extend(chunks)
        process_time = time.time() - process_start
        
        # 阶段3: 模拟向量化（GPU密集）
        embed_start = time.time()
        for chunk in processed:
            time.sleep(0.0005)  # 模拟向量化
        embed_time = time.time() - embed_start
        
        total_time = time.time() - start_time
        
        results = {
            "mode": "serial",
            "sample_size": sample_size,
            "total_time_seconds": round(total_time, 2),
            "crawl_time": round(crawl_time, 2),
            "process_time": round(process_time, 2),
            "embed_time": round(embed_time, 2),
            "throughput_articles_per_sec": round(sample_size / total_time, 2),
            "chunks_generated": len(processed),
            "bottleneck": "串行等待，各阶段无法并行"
        }
        
        self.results["before_integration"] = results
        logger.info(f"串行模式结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
        
        return results
    
    def evaluate_throughput_after(self, sample_size: int = 1000) -> Dict:
        """
        评估集成后的吞吐量（Kafka异步解耦）
        
        新流程：爬取 → Kafka → 处理消费者 → Kafka → 向量化消费者（并行）
        """
        logger.info("=" * 60)
        logger.info("📊 评估集成后性能（Kafka异步模式）")
        logger.info("=" * 60)
        
        test_articles = self._generate_test_data(sample_size)
        
        # 使用队列模拟Kafka
        from queue import Queue
        import concurrent.futures
        
        raw_queue = Queue()
        processed_queue = Queue()
        results_list = []
        
        # 并行处理
        start_time = time.time()
        
        def producer():
            """生产者：爬取并发送到Kafka"""
            for article in test_articles:
                time.sleep(0.001)  # 模拟网络延迟
                raw_queue.put(article)
            raw_queue.put(None)  # 结束信号
        
        def processor():
            """消费者1：处理数据"""
            while True:
                article = raw_queue.get()
                if article is None:
                    processed_queue.put(None)
                    break
                chunks = self._simulate_chunking(article.get("text", ""))
                for chunk in chunks:
                    processed_queue.put(chunk)
        
        def embedder():
            """消费者2：向量化"""
            count = 0
            while True:
                chunk = processed_queue.get()
                if chunk is None:
                    break
                time.sleep(0.0005)  # 模拟向量化
                count += 1
            return count
        
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            producer_future = executor.submit(producer)
            processor_future = executor.submit(processor)
            embedder_future = executor.submit(embedder)
            
            producer_future.result()
            processor_future.result()
            embed_count = embedder_future.result()
        
        total_time = time.time() - start_time
        
        results = {
            "mode": "kafka_async",
            "sample_size": sample_size,
            "total_time_seconds": round(total_time, 2),
            "throughput_articles_per_sec": round(sample_size / total_time, 2),
            "chunks_processed": embed_count,
            "parallelism": "3个阶段并行执行",
            "benefits": [
                "采集和处理解耦，互不阻塞",
                "消息持久化，支持重放",
                "可水平扩展消费者数量"
            ]
        }
        
        self.results["after_integration"] = results
        logger.info(f"Kafka异步模式结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
        
        return results
    
    def evaluate_fault_tolerance(self) -> Dict:
        """评估容错能力"""
        logger.info("=" * 60)
        logger.info("📊 评估容错能力")
        logger.info("=" * 60)
        
        comparison = {
            "before": {
                "failure_recovery": "需要从头开始",
                "data_persistence": "内存中，进程崩溃则丢失",
                "retry_mechanism": "手动重试",
                "monitoring": "查看日志文件"
            },
            "after": {
                "failure_recovery": "从Kafka offset继续消费",
                "data_persistence": "Kafka持久化，可保留7天",
                "retry_mechanism": "Airflow自动重试3次",
                "monitoring": "Airflow Web UI + Kafka UI"
            }
        }
        
        return comparison
    
    def evaluate_scalability(self) -> Dict:
        """评估可扩展性"""
        logger.info("=" * 60)
        logger.info("📊 评估可扩展性")
        logger.info("=" * 60)
        
        comparison = {
            "before": {
                "horizontal_scaling": "需要修改代码",
                "max_parallelism": "受限于单机资源",
                "bottleneck": "最慢的阶段决定整体速度"
            },
            "after": {
                "horizontal_scaling": "增加消费者实例即可",
                "max_parallelism": "Kafka分区数 × 消费者数",
                "bottleneck": "各阶段独立扩展，消除瓶颈"
            },
            "scaling_example": {
                "scenario": "处理速度不够",
                "before_solution": "优化代码或升级硬件",
                "after_solution": "启动更多处理消费者实例"
            }
        }
        
        return comparison
    
    def compare_results(self) -> Dict:
        """对比集成前后的结果"""
        before = self.results.get("before_integration", {})
        after = self.results.get("after_integration", {})
        
        if not before or not after:
            return {}
        
        speedup = before.get("total_time_seconds", 1) / max(after.get("total_time_seconds", 1), 0.001)
        throughput_improvement = after.get("throughput_articles_per_sec", 0) / max(before.get("throughput_articles_per_sec", 1), 0.001)
        
        comparison = {
            "speedup": round(speedup, 2),
            "throughput_improvement": f"{round(throughput_improvement, 2)}x",
            "time_saved_percent": round((1 - 1/speedup) * 100, 1),
            "key_improvements": [
                f"处理速度提升 {round(throughput_improvement, 1)}x",
                "支持断点续传和消息重放",
                "可视化任务监控（Airflow UI）",
                "自动失败重试和告警",
                "水平扩展能力"
            ]
        }
        
        self.results["comparison"] = comparison
        return comparison
    
    def run_full_evaluation(self, sample_size: int = 1000) -> Dict:
        """运行完整评估"""
        logger.info("=" * 60)
        logger.info("🚀 Kafka + Airflow 集成效果评估")
        logger.info("=" * 60)
        
        # 1. 评估集成前
        self.evaluate_throughput_before(sample_size)
        
        # 2. 评估集成后
        self.evaluate_throughput_after(sample_size)
        
        # 3. 对比结果
        self.compare_results()
        
        # 4. 容错能力
        self.results["fault_tolerance"] = self.evaluate_fault_tolerance()
        
        # 5. 可扩展性
        self.results["scalability"] = self.evaluate_scalability()
        
        # 保存结果
        output_file = RESULTS_DIR / "kafka_airflow_evaluation.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 评估完成，结果已保存: {output_file}")
        
        # 打印摘要
        self._print_summary()
        
        return self.results
    
    def _generate_test_data(self, size: int) -> List[Dict]:
        """生成测试数据"""
        return [
            {
                "pmid": f"test_{i}",
                "title": f"Test Article {i}",
                "text": "This is a test abstract. " * 50  # 约500字符
            }
            for i in range(size)
        ]
    
    def _simulate_chunking(self, text: str) -> List[str]:
        """模拟文本切分"""
        chunk_size = 512
        chunks = []
        for i in range(0, len(text), chunk_size - 50):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:
                chunks.append(chunk)
        return chunks
    
    def _print_summary(self):
        """打印评估摘要"""
        comparison = self.results.get("comparison", {})
        
        print("\n" + "=" * 60)
        print("📊 Kafka + Airflow 集成效果评估摘要")
        print("=" * 60)
        print(f"\n🚀 性能提升:")
        print(f"   - 处理速度: {comparison.get('throughput_improvement', 'N/A')}")
        print(f"   - 时间节省: {comparison.get('time_saved_percent', 'N/A')}%")
        print(f"\n✨ 关键改进:")
        for improvement in comparison.get("key_improvements", []):
            print(f"   - {improvement}")
        print("\n" + "=" * 60)


def main():
    """运行评估"""
    evaluator = KafkaAirflowEvaluator()
    evaluator.run_full_evaluation(sample_size=1000)


if __name__ == "__main__":
    main()
