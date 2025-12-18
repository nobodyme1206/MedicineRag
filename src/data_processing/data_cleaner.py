# -*- coding: utf-8 -*-
"""
数据降噪与异常检测模块
功能: 文本质量评估、去重、异常检测、数据清洗
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("data_cleaner", LOGS_DIR / "data_cleaning.log")


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        # 质量阈值
        self.min_abstract_length = 100  # 最小摘要长度
        self.max_abstract_length = 10000  # 最大摘要长度
        self.min_title_length = 10  # 最小标题长度
        self.max_duplicate_ratio = 0.3  # 最大重复内容比例
        
        # 垃圾词列表
        self.spam_patterns = [
            r'click here', r'buy now', r'free download',
            r'http[s]?://(?!www\.ncbi|pubmed|doi)',  # 非学术链接
            r'[A-Z]{10,}',  # 连续大写字母
            r'(.)\1{5,}',  # 重复字符
        ]
        
        # 必需字段
        self.required_fields = ['pmid', 'title', 'abstract']
    
    def check_article(self, article: Dict) -> Tuple[bool, List[str]]:
        """
        检查单篇文章质量
        
        Returns:
            (是否通过, 问题列表)
        """
        issues = []
        
        # 1. 检查必需字段
        for field in self.required_fields:
            if not article.get(field):
                issues.append(f"缺少必需字段: {field}")
        
        if issues:
            return False, issues
        
        title = article.get('title', '')
        abstract = article.get('abstract', '')
        
        # 2. 检查长度
        if len(abstract) < self.min_abstract_length:
            issues.append(f"摘要过短: {len(abstract)} < {self.min_abstract_length}")
        
        if len(abstract) > self.max_abstract_length:
            issues.append(f"摘要过长: {len(abstract)} > {self.max_abstract_length}")
        
        if len(title) < self.min_title_length:
            issues.append(f"标题过短: {len(title)} < {self.min_title_length}")
        
        # 3. 检查垃圾内容
        text = f"{title} {abstract}".lower()
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"检测到垃圾内容模式: {pattern}")
        
        # 4. 检查语言（简单检测是否为英文）
        english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
        if english_ratio < 0.5:
            issues.append(f"非英文内容比例过高: {1-english_ratio:.2%}")
        
        # 5. 检查重复内容
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < (1 - self.max_duplicate_ratio):
                issues.append(f"重复词比例过高: {1-unique_ratio:.2%}")
        
        return len(issues) == 0, issues
    
    def calculate_quality_score(self, article: Dict) -> float:
        """
        计算文章质量分数 (0-100)
        """
        score = 100.0
        
        title = article.get('title', '')
        abstract = article.get('abstract', '')
        
        # 长度评分
        if len(abstract) < 200:
            score -= 20
        elif len(abstract) < 500:
            score -= 10
        
        if len(title) < 20:
            score -= 10
        
        # 结构评分（有作者、日期、关键词）
        if not article.get('authors'):
            score -= 10
        if not article.get('pub_date'):
            score -= 5
        if not article.get('mesh_terms') and not article.get('keywords'):
            score -= 10
        
        # 内容质量评分
        text = f"{title} {abstract}"
        
        # 词汇丰富度
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score -= max(0, (0.5 - unique_ratio) * 30)
        
        # 句子完整性（以句号结尾的比例）
        sentences = re.split(r'[.!?]', abstract)
        if len(sentences) < 3:
            score -= 15
        
        return max(0, min(100, score))


class DataDeduplicator:
    """数据去重器"""
    
    def __init__(self):
        self.seen_pmids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
    
    def _compute_hash(self, text: str) -> str:
        """计算文本哈希"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, article: Dict) -> Tuple[bool, str]:
        """
        检查是否重复
        
        Returns:
            (是否重复, 重复类型)
        """
        pmid = article.get('pmid', '')
        
        # 1. PMID重复
        if pmid in self.seen_pmids:
            return True, "pmid_duplicate"
        
        # 2. 内容重复（基于摘要哈希）
        abstract = article.get('abstract', '')
        if abstract:
            content_hash = self._compute_hash(abstract)
            if content_hash in self.seen_hashes:
                return True, "content_duplicate"
            self.seen_hashes.add(content_hash)
        
        self.seen_pmids.add(pmid)
        return False, ""
    
    def deduplicate(self, articles: List[Dict]) -> Tuple[List[Dict], int]:
        """
        批量去重
        
        Returns:
            (去重后的文章列表, 移除数量)
        """
        unique_articles = []
        removed = 0
        
        for article in articles:
            is_dup, dup_type = self.is_duplicate(article)
            if not is_dup:
                unique_articles.append(article)
            else:
                removed += 1
        
        return unique_articles, removed


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self):
        self.stats = {
            'abstract_lengths': [],
            'title_lengths': [],
            'author_counts': []
        }
    
    def collect_stats(self, articles: List[Dict]):
        """收集统计信息"""
        for article in articles:
            self.stats['abstract_lengths'].append(len(article.get('abstract', '')))
            self.stats['title_lengths'].append(len(article.get('title', '')))
            self.stats['author_counts'].append(len(article.get('authors', [])))
    
    def _calculate_bounds(self, values: List[float], k: float = 2.5) -> Tuple[float, float]:
        """计算异常边界（基于IQR）"""
        if not values:
            return 0, float('inf')
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        
        return max(0, lower), upper
    
    def detect_anomalies(self, article: Dict) -> List[str]:
        """检测单篇文章的异常"""
        anomalies = []
        
        # 摘要长度异常
        abstract_len = len(article.get('abstract', ''))
        lower, upper = self._calculate_bounds(self.stats['abstract_lengths'])
        if abstract_len < lower or abstract_len > upper:
            anomalies.append(f"摘要长度异常: {abstract_len} (正常范围: {lower:.0f}-{upper:.0f})")
        
        # 标题长度异常
        title_len = len(article.get('title', ''))
        lower, upper = self._calculate_bounds(self.stats['title_lengths'])
        if title_len < lower or title_len > upper:
            anomalies.append(f"标题长度异常: {title_len} (正常范围: {lower:.0f}-{upper:.0f})")
        
        # 作者数量异常
        author_count = len(article.get('authors', []))
        lower, upper = self._calculate_bounds(self.stats['author_counts'])
        if author_count > upper:
            anomalies.append(f"作者数量异常: {author_count} (正常上限: {upper:.0f})")
        
        return anomalies


class DataCleaner:
    """数据清洗主类"""
    
    def __init__(self):
        self.quality_checker = DataQualityChecker()
        self.deduplicator = DataDeduplicator()
        self.anomaly_detector = AnomalyDetector()
        
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_quality': 0,
            'duplicates': 0,
            'anomalies': 0,
            'quality_scores': []
        }
    
    def clean_dataset(self, input_file: Path, output_file: Path = None,
                     remove_anomalies: bool = False) -> Dict:
        """
        清洗数据集
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）
            remove_anomalies: 是否移除异常数据
            
        Returns:
            清洗统计
        """
        logger.info("=" * 60)
        logger.info("🧹 开始数据清洗")
        logger.info(f"   输入文件: {input_file}")
        logger.info("=" * 60)
        
        # 加载数据
        logger.info("📂 加载数据...")
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        self.stats['total'] = len(articles)
        logger.info(f"   总文章数: {len(articles):,}")
        
        # 第一遍：收集统计信息
        logger.info("\n📊 收集统计信息...")
        self.anomaly_detector.collect_stats(articles)
        
        # 第二遍：清洗
        logger.info("\n🔍 开始清洗...")
        cleaned_articles = []
        quality_issues = Counter()
        
        for i, article in enumerate(articles):
            if (i + 1) % 50000 == 0:
                logger.info(f"   进度: {i+1:,}/{len(articles):,}")
            
            # 1. 质量检查
            passed, issues = self.quality_checker.check_article(article)
            if not passed:
                self.stats['failed_quality'] += 1
                for issue in issues:
                    quality_issues[issue.split(':')[0]] += 1
                continue
            
            # 2. 去重检查
            is_dup, _ = self.deduplicator.is_duplicate(article)
            if is_dup:
                self.stats['duplicates'] += 1
                continue
            
            # 3. 异常检测
            anomalies = self.anomaly_detector.detect_anomalies(article)
            if anomalies:
                self.stats['anomalies'] += 1
                article['_anomalies'] = anomalies
                if remove_anomalies:
                    continue
            
            # 4. 计算质量分数
            score = self.quality_checker.calculate_quality_score(article)
            article['_quality_score'] = score
            self.stats['quality_scores'].append(score)
            
            cleaned_articles.append(article)
            self.stats['passed'] += 1
        
        # 保存结果
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
            logger.info(f"\n💾 已保存清洗后数据: {output_file}")
        
        # 打印统计
        self._print_stats(quality_issues)
        
        return {
            'stats': self.stats,
            'quality_issues': dict(quality_issues),
            'cleaned_articles': cleaned_articles
        }
    
    def _print_stats(self, quality_issues: Counter):
        """打印统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 清洗统计")
        logger.info("=" * 60)
        logger.info(f"   总文章数: {self.stats['total']:,}")
        logger.info(f"   通过数量: {self.stats['passed']:,} ({self.stats['passed']/self.stats['total']*100:.1f}%)")
        logger.info(f"   质量不合格: {self.stats['failed_quality']:,}")
        logger.info(f"   重复移除: {self.stats['duplicates']:,}")
        logger.info(f"   异常标记: {self.stats['anomalies']:,}")
        
        if self.stats['quality_scores']:
            avg_score = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            logger.info(f"   平均质量分: {avg_score:.1f}/100")
        
        if quality_issues:
            logger.info("\n📋 质量问题分布:")
            for issue, count in quality_issues.most_common(10):
                logger.info(f"   - {issue}: {count:,}")


def main():
    """主函数"""
    input_file = RAW_DATA_DIR / "pubmed_articles_all.json"
    output_file = PROCESSED_DATA_DIR / "pubmed_cleaned.json"
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    cleaner = DataCleaner()
    result = cleaner.clean_dataset(input_file, output_file, remove_anomalies=False)
    
    logger.info("\n✅ 数据清洗完成!")


if __name__ == "__main__":
    main()
