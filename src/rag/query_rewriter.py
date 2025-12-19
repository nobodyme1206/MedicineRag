#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询改写模块 - 增强版
功能: 医学术语标准化、同义词扩展、查询改写
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Set

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL, LOGS_DIR
)
from src.utils.logger import setup_logger
from src.utils.exceptions import GenerationError, handle_errors

logger = setup_logger("query_rewriter", LOGS_DIR / "query_rewriter.log")


# 医学术语同义词库（中英文映射 + 同义词扩展）
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    # 疾病
    "diabetes": ["diabetes mellitus", "DM", "diabetic", "hyperglycemia", "blood sugar disorder"],
    "糖尿病": ["diabetes", "diabetes mellitus", "DM", "血糖异常", "高血糖"],
    "hypertension": ["high blood pressure", "HTN", "elevated blood pressure", "arterial hypertension"],
    "高血压": ["hypertension", "HTN", "血压升高", "动脉高压"],
    "cancer": ["carcinoma", "malignancy", "tumor", "neoplasm", "oncology"],
    "癌症": ["cancer", "carcinoma", "malignancy", "肿瘤", "恶性肿瘤"],
    "heart disease": ["cardiovascular disease", "CVD", "cardiac disease", "heart condition"],
    "心脏病": ["heart disease", "cardiovascular disease", "CVD", "心血管疾病"],
    "stroke": ["cerebrovascular accident", "CVA", "brain attack", "cerebral infarction"],
    "中风": ["stroke", "CVA", "脑卒中", "脑梗塞", "脑血管意外"],
    "alzheimer": ["alzheimer's disease", "AD", "dementia", "cognitive decline"],
    "阿尔茨海默": ["alzheimer", "AD", "老年痴呆", "认知障碍"],
    "asthma": ["bronchial asthma", "reactive airway disease", "wheezing disorder"],
    "哮喘": ["asthma", "bronchial asthma", "支气管哮喘"],
    "arthritis": ["joint inflammation", "rheumatoid arthritis", "osteoarthritis", "RA", "OA"],
    "关节炎": ["arthritis", "RA", "OA", "风湿性关节炎", "骨关节炎"],
    "pneumonia": ["lung infection", "pulmonary infection", "chest infection"],
    "肺炎": ["pneumonia", "lung infection", "肺部感染"],
    "obesity": ["overweight", "adiposity", "excess weight", "BMI elevation"],
    "肥胖": ["obesity", "overweight", "超重", "体重过重"],
    
    # 症状
    "fever": ["pyrexia", "elevated temperature", "febrile", "hyperthermia"],
    "发烧": ["fever", "pyrexia", "发热", "体温升高"],
    "pain": ["ache", "discomfort", "soreness", "tenderness", "algesia"],
    "疼痛": ["pain", "ache", "不适", "酸痛"],
    "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy", "asthenia"],
    "疲劳": ["fatigue", "tiredness", "乏力", "虚弱"],
    "cough": ["tussis", "coughing", "expectoration"],
    "咳嗽": ["cough", "tussis", "干咳", "咳痰"],
    "headache": ["cephalalgia", "head pain", "migraine"],
    "头痛": ["headache", "cephalalgia", "偏头痛"],
    
    # 治疗
    "treatment": ["therapy", "management", "intervention", "cure", "remedy"],
    "治疗": ["treatment", "therapy", "management", "疗法", "干预"],
    "surgery": ["operation", "surgical procedure", "surgical intervention"],
    "手术": ["surgery", "operation", "外科手术"],
    "medication": ["drug", "medicine", "pharmaceutical", "pharmacotherapy"],
    "药物": ["medication", "drug", "medicine", "用药"],
    "prevention": ["prophylaxis", "preventive measures", "risk reduction"],
    "预防": ["prevention", "prophylaxis", "预防措施"],
    
    # 检查
    "diagnosis": ["diagnostic", "detection", "identification", "assessment"],
    "诊断": ["diagnosis", "diagnostic", "检测", "评估"],
    "symptoms": ["signs", "manifestations", "clinical features", "presentations"],
    "症状": ["symptoms", "signs", "表现", "临床特征"],
}

# 医学缩写展开
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "DM": "diabetes mellitus",
    "HTN": "hypertension",
    "CVD": "cardiovascular disease",
    "CVA": "cerebrovascular accident",
    "MI": "myocardial infarction",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
    "RA": "rheumatoid arthritis",
    "OA": "osteoarthritis",
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "HbA1c": "glycated hemoglobin",
    "BMI": "body mass index",
    "BP": "blood pressure",
    "HR": "heart rate",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "PET": "positron emission tomography",
    "ICU": "intensive care unit",
    "ED": "emergency department",
    "GI": "gastrointestinal",
    "CNS": "central nervous system",
    "HIV": "human immunodeficiency virus",
    "AIDS": "acquired immunodeficiency syndrome",
    "COVID": "coronavirus disease",
    "SARS": "severe acute respiratory syndrome",
}


class MedicalTermNormalizer:
    """医学术语标准化器"""
    
    def __init__(self):
        self.synonyms = MEDICAL_SYNONYMS
        self.abbreviations = MEDICAL_ABBREVIATIONS
        # 构建反向索引
        self._build_reverse_index()
    
    def _build_reverse_index(self):
        """构建同义词反向索引"""
        self.term_to_canonical: Dict[str, str] = {}
        for canonical, synonyms in self.synonyms.items():
            self.term_to_canonical[canonical.lower()] = canonical
            for syn in synonyms:
                self.term_to_canonical[syn.lower()] = canonical
    
    def normalize(self, text: str) -> str:
        """
        标准化医学术语
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        # 1. 展开缩写
        words = text.split()
        expanded_words = []
        for word in words:
            clean_word = word.strip('.,?!;:')
            if clean_word.upper() in self.abbreviations:
                expanded = self.abbreviations[clean_word.upper()]
                expanded_words.append(expanded)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        获取术语的同义词
        
        Args:
            term: 医学术语
            
        Returns:
            同义词列表
        """
        term_lower = term.lower()
        
        # 直接匹配
        if term_lower in self.synonyms:
            return self.synonyms[term_lower]
        
        # 通过反向索引查找
        if term_lower in self.term_to_canonical:
            canonical = self.term_to_canonical[term_lower]
            return self.synonyms.get(canonical, [])
        
        return []
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        扩展查询（添加同义词）
        
        Args:
            query: 原始查询
            max_expansions: 最大扩展数
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = [query]
        query_lower = query.lower()
        
        # 查找可扩展的术语
        for term, synonyms in self.synonyms.items():
            if term.lower() in query_lower:
                for syn in synonyms[:max_expansions]:
                    if syn.lower() != term.lower():
                        new_query = re.sub(
                            re.escape(term), syn, query, flags=re.IGNORECASE
                        )
                        if new_query not in expanded_queries:
                            expanded_queries.append(new_query)
                break  # 只扩展第一个匹配的术语
        
        return expanded_queries[:max_expansions + 1]


class QueryRewriter:
    """查询改写器 - 增强版：医学术语标准化 + 同义词扩展 + LLM改写"""
    
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True) -> None:
        """
        初始化查询改写器
        
        Args:
            api_key: API密钥，默认使用配置文件
            use_llm: 是否使用LLM进行改写
        """
        self.normalizer = MedicalTermNormalizer()
        self.use_llm = use_llm
        self.client = None
        
        if use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key or SILICONFLOW_API_KEY,
                    base_url=SILICONFLOW_BASE_URL
                )
                self.model = SILICONFLOW_MODEL
            except Exception as e:
                logger.warning(f"LLM客户端初始化失败: {e}")
                self.use_llm = False
        
        logger.info(f"查询改写器初始化完成 (LLM: {self.use_llm})")
    
    def normalize_query(self, query: str) -> str:
        """
        标准化查询（展开缩写、统一术语）
        
        Args:
            query: 原始查询
            
        Returns:
            标准化后的查询
        """
        normalized = self.normalizer.normalize(query)
        if normalized != query:
            logger.debug(f"术语标准化: '{query}' -> '{normalized}'")
        return normalized
    
    def expand_with_synonyms(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        使用同义词扩展查询（不调用LLM，速度快）
        
        Args:
            query: 原始查询
            max_expansions: 最大扩展数
            
        Returns:
            扩展后的查询列表
        """
        return self.normalizer.expand_query(query, max_expansions)
    
    def get_enhanced_query(self, query: str) -> str:
        """
        获取增强查询（标准化 + 同义词补充）
        
        Args:
            query: 原始查询
            
        Returns:
            增强后的查询
        """
        # 1. 标准化
        normalized = self.normalize_query(query)
        
        # 2. 提取关键术语的同义词
        synonyms_to_add: Set[str] = set()
        query_lower = query.lower()
        
        for term in self.normalizer.synonyms.keys():
            if term.lower() in query_lower:
                syns = self.normalizer.get_synonyms(term)
                # 添加最相关的2个同义词
                for syn in syns[:2]:
                    if syn.lower() not in query_lower:
                        synonyms_to_add.add(syn)
        
        # 3. 构建增强查询
        if synonyms_to_add:
            enhanced = f"{normalized} ({' '.join(list(synonyms_to_add)[:3])})"
            logger.debug(f"查询增强: '{query}' -> '{enhanced}'")
            return enhanced
        
        return normalized
    
    @handle_errors(default_return=None, log_level="warning")
    def rewrite(self, query: str) -> Optional[str]:
        """
        改写查询，使其更适合检索
        
        Args:
            query: 原始查询
            
        Returns:
            改写后的查询
        """
        # 先进行本地标准化
        normalized = self.normalize_query(query)
        
        # 如果不使用LLM，返回标准化结果
        if not self.use_llm or not self.client:
            return self.get_enhanced_query(query)
        
        prompt = f"""Rewrite the following medical query to improve search results.
Make it more specific and include relevant medical terms.
Keep it concise (under 100 words).

Original query: {normalized}

Rewritten query:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        rewritten = response.choices[0].message.content.strip()
        logger.debug(f"查询改写: '{query}' -> '{rewritten}'")
        return rewritten
    
    @handle_errors(default_return=[], log_level="warning")
    def expand(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        扩展查询，生成多个相关查询
        
        Args:
            query: 原始查询
            num_expansions: 扩展数量
            
        Returns:
            扩展查询列表
        """
        # 先用本地同义词扩展
        local_expansions = self.expand_with_synonyms(query, num_expansions)
        
        # 如果不使用LLM，返回本地扩展
        if not self.use_llm or not self.client:
            return local_expansions
        
        prompt = f"""Generate {num_expansions} alternative search queries for the following medical question.
Each query should focus on different aspects or use different terminology.
Return only the queries, one per line.

Original: {query}

Alternative queries:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        expansions = [line.strip() for line in content.split('\n') if line.strip()]
        expansions = [e.lstrip('0123456789.-) ') for e in expansions]
        
        # 合并本地和LLM扩展，去重
        all_expansions = local_expansions + expansions
        seen = set()
        unique = []
        for exp in all_expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique.append(exp)
        
        logger.debug(f"查询扩展: '{query}' -> {len(unique)} 个变体")
        return unique[:num_expansions + len(local_expansions)]
    
    def expand_medical_query(self, query: str) -> str:
        """
        医学查询扩展（用于RAG系统）
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询字符串
        """
        enhanced = self.get_enhanced_query(query)
        return enhanced
    
    @handle_errors(default_return=[], log_level="warning")
    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键医学术语
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        keywords = []
        query_lower = query.lower()
        
        # 本地提取
        for term in self.normalizer.synonyms.keys():
            if term.lower() in query_lower:
                keywords.append(term)
        
        # 如果本地提取到了，直接返回
        if keywords:
            return keywords
        
        # 否则使用LLM
        if not self.use_llm or not self.client:
            # 简单分词
            return [w for w in query.split() if len(w) > 3]
        
        prompt = f"""Extract the key medical terms from this query.
Return only the terms, separated by commas.

Query: {query}

Key terms:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        content = response.choices[0].message.content.strip()
        keywords = [k.strip() for k in content.split(',') if k.strip()]
        return keywords


def main() -> None:
    """测试查询改写"""
    print("=" * 60)
    print("查询改写模块测试")
    print("=" * 60)
    
    rewriter = QueryRewriter(use_llm=False)  # 先测试本地功能
    
    test_queries = [
        "What causes diabetes?",
        "糖尿病的症状有哪些？",
        "DM treatment options",
        "How to prevent HTN?",
        "cancer symptoms and diagnosis",
    ]
    
    for query in test_queries:
        print(f"\n原始查询: {query}")
        print(f"标准化: {rewriter.normalize_query(query)}")
        print(f"增强查询: {rewriter.get_enhanced_query(query)}")
        print(f"同义词扩展: {rewriter.expand_with_synonyms(query)}")
        print(f"关键词: {rewriter.extract_keywords(query)}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")


if __name__ == "__main__":
    main()
