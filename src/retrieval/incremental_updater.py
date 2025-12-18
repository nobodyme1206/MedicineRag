# -*- coding: utf-8 -*-
"""
增量更新模块
功能: 向量库增量更新、定时任务、变更检测
"""

import os
import time
import json
import hashlib
import schedule
import threading
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("incremental_updater", LOGS_DIR / "incremental_update.log")


class ChangeDetector:
    """变更检测器"""
    
    def __init__(self, state_file: Path = None):
        """
        初始化变更检测器
        
        Args:
            state_file: 状态文件路径
        """
        self.state_file = state_file or (PROCESSED_DATA_DIR / "update_state.json")
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_update": None,
            "file_hashes": {},
            "processed_pmids": []
        }
    
    def _save_state(self):
        """保存状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def detect_changes(self, data_dir: Path = None) -> Dict:
        """
        检测数据变更
        
        Returns:
            变更信息
        """
        data_dir = data_dir or RAW_DATA_DIR
        
        changes = {
            "new_files": [],
            "modified_files": [],
            "deleted_files": [],
            "has_changes": False
        }
        
        # 扫描当前文件
        current_files = {}
        for file_path in data_dir.glob("*.json"):
            file_hash = self._compute_file_hash(file_path)
            current_files[str(file_path)] = file_hash
        
        # 检测新增和修改
        for file_path, file_hash in current_files.items():
            if file_path not in self.state["file_hashes"]:
                changes["new_files"].append(file_path)
            elif self.state["file_hashes"][file_path] != file_hash:
                changes["modified_files"].append(file_path)
        
        # 检测删除
        for file_path in self.state["file_hashes"]:
            if file_path not in current_files:
                changes["deleted_files"].append(file_path)
        
        changes["has_changes"] = bool(
            changes["new_files"] or 
            changes["modified_files"] or 
            changes["deleted_files"]
        )
        
        # 更新状态
        if changes["has_changes"]:
            self.state["file_hashes"] = current_files
            self.state["last_update"] = datetime.now().isoformat()
            self._save_state()
        
        return changes
    
    def get_new_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        获取新增文章（未处理过的）
        
        Args:
            articles: 文章列表
            
        Returns:
            新增文章列表
        """
        processed_pmids = set(self.state.get("processed_pmids", []))
        new_articles = [a for a in articles if a.get("pmid") not in processed_pmids]
        return new_articles
    
    def mark_processed(self, pmids: List[str]):
        """标记已处理的PMID"""
        self.state["processed_pmids"].extend(pmids)
        # 保持列表不要太大
        if len(self.state["processed_pmids"]) > 1000000:
            self.state["processed_pmids"] = self.state["processed_pmids"][-500000:]
        self._save_state()


class IncrementalUpdater:
    """增量更新器"""
    
    def __init__(self, embedder=None, milvus_manager=None):
        """
        初始化增量更新器
        
        Args:
            embedder: 向量化器
            milvus_manager: Milvus管理器
        """
        self.embedder = embedder
        self.milvus = milvus_manager
        self.change_detector = ChangeDetector()
        
        self.stats = {
            "total_updates": 0,
            "total_vectors_added": 0,
            "last_update_time": None,
            "errors": 0
        }
        
        self._running = False
        self._scheduler_thread = None
    
    def _init_components(self):
        """延迟初始化组件"""
        if self.embedder is None:
            from src.embedding.embedder import TextEmbedder
            self.embedder = TextEmbedder()
            logger.info("✅ 向量化器初始化完成")
        
        if self.milvus is None:
            from src.retrieval.milvus_manager import MilvusManager
            self.milvus = MilvusManager()
            self.milvus.load_collection()
            logger.info("✅ Milvus连接完成")
    
    def update_from_file(self, file_path: Path, batch_size: int = 128) -> Dict:
        """
        从文件增量更新
        
        Args:
            file_path: 数据文件路径
            batch_size: 批次大小
            
        Returns:
            更新统计
        """
        logger.info(f"📂 从文件增量更新: {file_path}")
        
        self._init_components()
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # 获取新增文章
        new_articles = self.change_detector.get_new_articles(articles)
        
        if not new_articles:
            logger.info("   无新增数据")
            return {"added": 0, "skipped": len(articles)}
        
        logger.info(f"   新增文章: {len(new_articles):,}")
        
        # 向量化并插入
        added = 0
        for i in range(0, len(new_articles), batch_size):
            batch = new_articles[i:i + batch_size]
            
            try:
                # 提取文本
                texts = [f"{a.get('title', '')} {a.get('abstract', '')}" for a in batch]
                
                # 向量化
                embeddings = self.embedder.encode_batch(texts)
                
                # 准备元数据
                metadata = [
                    {'pmid': str(a.get('pmid', '')), 'chunk_text': texts[j][:2000]}
                    for j, a in enumerate(batch)
                ]
                
                # 插入Milvus
                self.milvus.insert_vectors(embeddings, metadata)
                
                added += len(batch)
                
                # 标记已处理
                pmids = [a.get('pmid') for a in batch]
                self.change_detector.mark_processed(pmids)
                
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                self.stats["errors"] += 1
        
        self.stats["total_vectors_added"] += added
        self.stats["total_updates"] += 1
        self.stats["last_update_time"] = datetime.now().isoformat()
        
        logger.info(f"✅ 增量更新完成: 新增 {added:,} 条向量")
        
        return {"added": added, "skipped": len(articles) - len(new_articles)}
    
    def check_and_update(self) -> Dict:
        """
        检查变更并更新
        
        Returns:
            更新结果
        """
        logger.info("🔍 检查数据变更...")
        
        changes = self.change_detector.detect_changes()
        
        if not changes["has_changes"]:
            logger.info("   无变更")
            return {"status": "no_changes"}
        
        logger.info(f"   检测到变更:")
        logger.info(f"   - 新增文件: {len(changes['new_files'])}")
        logger.info(f"   - 修改文件: {len(changes['modified_files'])}")
        logger.info(f"   - 删除文件: {len(changes['deleted_files'])}")
        
        # 处理新增和修改的文件
        results = []
        for file_path in changes["new_files"] + changes["modified_files"]:
            result = self.update_from_file(Path(file_path))
            results.append(result)
        
        total_added = sum(r.get("added", 0) for r in results)
        
        return {
            "status": "updated",
            "changes": changes,
            "total_added": total_added
        }
    
    def start_scheduler(self, interval_minutes: int = 60):
        """
        启动定时更新任务
        
        Args:
            interval_minutes: 更新间隔（分钟）
        """
        logger.info(f"⏰ 启动定时更新任务 (间隔: {interval_minutes}分钟)")
        
        self._running = True
        
        # 设置定时任务
        schedule.every(interval_minutes).minutes.do(self._scheduled_update)
        
        # 启动调度线程
        def run_scheduler():
            while self._running:
                schedule.run_pending()
                time.sleep(10)
        
        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("✅ 定时任务已启动")
    
    def _scheduled_update(self):
        """定时更新任务"""
        logger.info(f"\n{'='*60}")
        logger.info(f"⏰ 执行定时更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        try:
            result = self.check_and_update()
            logger.info(f"更新结果: {result}")
        except Exception as e:
            logger.error(f"定时更新失败: {e}")
            self.stats["errors"] += 1
    
    def stop_scheduler(self):
        """停止定时任务"""
        self._running = False
        schedule.clear()
        logger.info("⏹️ 定时任务已停止")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "milvus_vectors": self.milvus.collection.num_entities if self.milvus else 0
        }


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("增量更新模块演示")
    logger.info("=" * 60)
    
    # 初始化
    updater = IncrementalUpdater()
    
    # 检查变更
    logger.info("\n1. 检查数据变更...")
    changes = updater.change_detector.detect_changes()
    logger.info(f"   变更检测结果: {changes}")
    
    # 手动更新
    data_file = RAW_DATA_DIR / "pubmed_articles_all.json"
    if data_file.exists():
        logger.info("\n2. 执行增量更新...")
        result = updater.update_from_file(data_file)
        logger.info(f"   更新结果: {result}")
    
    # 启动定时任务（演示）
    logger.info("\n3. 定时任务演示...")
    logger.info("   使用方法: updater.start_scheduler(interval_minutes=60)")
    logger.info("   停止方法: updater.stop_scheduler()")
    
    # 打印统计
    logger.info(f"\n📊 统计信息: {updater.get_stats()}")
    
    logger.info("\n✅ 演示完成!")


if __name__ == "__main__":
    main()
