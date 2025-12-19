# -*- coding: utf-8 -*-
"""
Medical Adaptive RAG Agent
自适应RAG Agent：智能路由 + 查询分解 + 自我反思 + 检索增强

架构特点：
1. 智能路由 - 简单问题快速回答，复杂问题深度处理
2. 查询分解 - 复杂问题拆分为多个子查询
3. 自我反思 - 检查检索结果质量，不相关则重试
4. 检索增强 - 使用RRF融合、查询标准化等优化
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent.parent))

from openai import OpenAI

from config.config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL,
    LOGS_DIR, LLM_TEMPERATURE, LLM_MAX_TOKENS
)
from src.utils.logger import setup_logger

logger = setup_logger("adaptive_agent", LOGS_DIR / "agent.log")


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"      # 简单查询，直接检索
    MODERATE = "moderate"  # 中等复杂，需要查询增强
    COMPLEX = "complex"    # 复杂查询，需要分解


@dataclass
class RetrievalResult:
    """检索结果"""
    docs: List[Dict[str, Any]]
    query: str
    is_relevant: bool = True
    relevance_score: float = 0.0


@dataclass 
class AgentStep:
    """Agent执行步骤"""
    step_type: str  # route, retrieve, reflect, decompose, generate
    input_data: str
    output_data: str
    duration_ms: float


class AdaptiveRAGAgent:
    """
    自适应RAG Agent
    
    工作流程：
    1. 路由判断 - 分析问题复杂度
    2. 查询处理 - 简单直接检索，复杂则分解
    3. 检索执行 - 使用增强检索（RRF融合）
    4. 自我反思 - 验证检索质量
    5. 答案生成 - 基于文献生成答案
    """
    
    def __init__(self, verbose: bool = True, rag_system=None):
        """
        初始化Adaptive RAG Agent
        
        Args:
            verbose: 是否输出详细日志
            rag_system: 可选，复用已初始化的RAG系统（避免重复加载模型）
        """
        self.verbose = verbose
        self.steps: List[AgentStep] = []
        self._shared_rag = rag_system  # 保存传入的RAG系统
        
        # 初始化组件
        self._setup_llm()
        self._setup_retriever()
        self._setup_query_enhancer()
        
        logger.info("✅ Adaptive RAG Agent 初始化完成")
    
    def _setup_llm(self) -> None:
        """配置LLM"""
        self.llm = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
        self.model = SILICONFLOW_MODEL
        logger.info(f"LLM配置完成: {self.model}")
    
    def _setup_retriever(self) -> None:
        """配置检索器 - 优先复用传入的RAG系统"""
        if self._shared_rag is not None:
            # 复用已初始化的RAG系统
            self.rag = self._shared_rag
            logger.info("RAG检索器初始化成功（复用已有实例）")
            return
        
        # 否则创建新实例
        try:
            from src.rag.rag_system import RAGSystem
            self.rag = RAGSystem(use_hyde=False, use_cache=True)
            logger.info("RAG检索器初始化成功")
        except Exception as e:
            logger.error(f"RAG检索器初始化失败: {e}")
            self.rag = None
    
    def _setup_query_enhancer(self) -> None:
        """配置查询增强器 - 优先复用RAG系统的组件"""
        # 如果RAG系统有查询改写器，直接复用
        if self.rag and hasattr(self.rag, 'query_rewriter') and self.rag.query_rewriter:
            self.query_rewriter = self.rag.query_rewriter
            logger.info("查询增强器初始化成功（复用RAG组件）")
            return
        
        try:
            from src.rag.query_rewriter import QueryRewriter
            self.query_rewriter = QueryRewriter(use_llm=False)
            logger.info("查询增强器初始化成功")
        except Exception as e:
            logger.warning(f"查询增强器初始化失败: {e}")
            self.query_rewriter = None

    # ==================== 核心方法 ====================
    
    def _call_llm(self, prompt: str, system_prompt: str = None, temperature: float = None) -> str:
        """调用LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    
    def _add_step(self, step_type: str, input_data: str, output_data: str, duration_ms: float):
        """记录执行步骤"""
        step = AgentStep(
            step_type=step_type,
            input_data=input_data[:200],  # 截断
            output_data=output_data[:500],
            duration_ms=duration_ms
        )
        self.steps.append(step)
        if self.verbose:
            logger.info(f"[{step_type}] {duration_ms:.0f}ms")
    
    # ==================== 路由判断 ====================
    
    def _route_query(self, query: str) -> QueryComplexity:
        """
        分析查询复杂度，决定处理策略
        
        简单: 单一概念查询（什么是X、X的定义）
        中等: 需要综合信息（X的症状、X的治疗方法）
        复杂: 多概念关联、比较、推理（X和Y的区别、为什么X导致Y）
        """
        start = time.time()
        
        prompt = f"""分析以下医学问题的复杂度，只回答一个词：simple、moderate、complex

判断标准：
- simple: 单一概念定义或事实查询（如"什么是糖尿病"）
- moderate: 需要综合多个信息点（如"糖尿病的症状有哪些"）
- complex: 涉及多概念关联、比较或推理（如"1型和2型糖尿病的区别"、"为什么糖尿病会导致视网膜病变"）

问题: {query}

复杂度:"""
        
        result = self._call_llm(prompt, temperature=0.1)
        result_lower = result.lower().strip()
        
        if "complex" in result_lower:
            complexity = QueryComplexity.COMPLEX
        elif "moderate" in result_lower:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        duration = (time.time() - start) * 1000
        self._add_step("route", query, complexity.value, duration)
        
        logger.info(f"查询复杂度: {complexity.value}")
        return complexity
    
    # ==================== 查询分解 ====================
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        将复杂查询分解为多个子查询
        """
        start = time.time()
        
        prompt = f"""将以下复杂医学问题分解为2-3个简单的子问题，每行一个问题。

原问题: {query}

子问题（每行一个）:"""
        
        result = self._call_llm(prompt, temperature=0.3)
        
        # 解析子问题
        sub_queries = []
        for line in result.split('\n'):
            line = line.strip()
            # 去除序号
            line = line.lstrip('0123456789.-) ')
            if line and len(line) > 5:
                sub_queries.append(line)
        
        # 至少保留原问题
        if not sub_queries:
            sub_queries = [query]
        
        duration = (time.time() - start) * 1000
        self._add_step("decompose", query, str(sub_queries), duration)
        
        logger.info(f"查询分解: {len(sub_queries)} 个子问题")
        return sub_queries[:3]  # 最多3个
    
    # ==================== 检索执行 ====================
    
    def _retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        执行检索（使用增强检索）
        """
        start = time.time()
        
        if not self.rag:
            return RetrievalResult(docs=[], query=query, is_relevant=False)
        
        # 查询增强
        enhanced_query = query
        if self.query_rewriter:
            enhanced_query = self.query_rewriter.normalize_query(query)
        
        # 执行检索
        docs = self.rag.retrieve(enhanced_query, top_k=top_k)
        
        duration = (time.time() - start) * 1000
        self._add_step("retrieve", query, f"{len(docs)} docs", duration)
        
        return RetrievalResult(
            docs=docs,
            query=query,
            is_relevant=len(docs) > 0
        )
    
    def _retrieve_multiple(self, queries: List[str], top_k: int = 8) -> List[Dict]:
        """
        多查询检索并合并去重
        """
        all_docs = []
        seen_pmids = set()
        
        for query in queries:
            result = self._retrieve(query, top_k=top_k)
            for doc in result.docs:
                pmid = doc.get('pmid', '')
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    all_docs.append(doc)
        
        # 按分数排序
        all_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_docs[:top_k * 2]  # 返回更多结果

    # ==================== 自我反思 ====================
    
    def _reflect_on_retrieval(self, query: str, docs: List[Dict]) -> Tuple[bool, float]:
        """
        反思检索结果质量（多维度评估）
        
        评估维度：
        1. 文档数量充足性
        2. 分数分布（使用相对排名而非绝对阈值）
        3. 查询词覆盖率
        
        Returns:
            (is_relevant, relevance_score)
        """
        if not docs:
            return False, 0.0
        
        start = time.time()
        
        # 1. 文档数量评分 (0-3分)
        doc_count = len(docs)
        count_score = min(doc_count / 5, 1.0) * 3  # 5个文档得满分3分
        
        # 2. 分数质量评分 (0-4分)
        # 使用相对评估：top1 vs top5 的分数差距
        def get_doc_score(doc):
            if 'rerank_score' in doc:
                return doc['rerank_score']
            return doc.get('score', 0)
        
        top_scores = [get_doc_score(d) for d in docs[:5]]
        if top_scores:
            max_score = max(top_scores)
            avg_score = sum(top_scores) / len(top_scores)
            # 分数集中度：如果 top 文档分数接近，说明都相关
            score_concentration = avg_score / max_score if max_score > 0 else 0
            # 分数质量：基于最高分的相对值（避免绝对阈值）
            # 只要有分数且分布合理就给分
            quality_score = (2.0 if max_score > 0 else 0) + (score_concentration * 2)
        else:
            quality_score = 0
            max_score = 0
        
        # 3. 查询词覆盖评分 (0-3分)
        query_terms = set(query.lower().split())
        if query_terms and docs:
            covered = 0
            for term in query_terms:
                for doc in docs[:3]:
                    if term in doc.get('text', '').lower():
                        covered += 1
                        break
            coverage_score = (covered / len(query_terms)) * 3
        else:
            coverage_score = 0
        
        # 总分 (0-10)
        relevance_score = count_score + quality_score + coverage_score
        
        # 判断是否相关：总分 >= 5 或 文档数 >= 3 且有分数
        is_relevant = relevance_score >= 5.0 or (doc_count >= 3 and max_score > 0)
        
        duration = (time.time() - start) * 1000
        self._add_step("reflect", query, f"score={relevance_score:.2f}, relevant={is_relevant}", duration)
        
        logger.info(f"检索质量评估: 分数={relevance_score:.2f}/10 (数量={count_score:.1f}, 质量={quality_score:.1f}, 覆盖={coverage_score:.1f}), 相关={is_relevant}")
        return is_relevant, relevance_score
    
    # ==================== 答案生成 ====================
    
    def _generate_answer(self, query: str, docs: List[Dict]) -> str:
        """
        基于检索结果生成答案
        """
        start = time.time()
        
        if not docs:
            return "抱歉，没有找到相关的医学文献来回答您的问题。"
        
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(docs[:10]):
            pmid = doc.get('pmid', '')
            text = doc.get('text', '')
            context_parts.append(f"[文档{i+1}, PMID:{pmid}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """你是一个专业的医学知识问答助手。请基于提供的医学文献回答问题。

要求：
1. 答案必须基于提供的文献内容
2. 使用[文档N]格式引用来源
3. 如果文献信息不足，明确说明
4. 使用专业但易懂的语言
5. 结构化回答（如有必要使用分点）"""
        
        prompt = f"""## 参考文献
{context}

## 用户问题
{query}

## 请回答："""
        
        answer = self._call_llm(prompt, system_prompt=system_prompt)
        
        duration = (time.time() - start) * 1000
        self._add_step("generate", query, answer[:200], duration)
        
        return answer
    
    # ==================== 主入口 ====================
    
    def chat(self, query: str, max_retries: int = 1) -> Dict[str, Any]:
        """
        与Agent对话（主入口）
        
        Args:
            query: 用户问题
            max_retries: 最大重试次数
            
        Returns:
            包含答案和元信息的字典
        """
        logger.info(f"用户问题: {query}")
        self.steps = []  # 重置步骤
        total_start = time.time()
        
        try:
            # 1. 路由判断
            complexity = self._route_query(query)
            
            # 2. 根据复杂度处理
            if complexity == QueryComplexity.COMPLEX:
                # 复杂问题：分解后多次检索
                sub_queries = self._decompose_query(query)
                docs = self._retrieve_multiple(sub_queries)
            else:
                # 简单/中等问题：直接检索
                result = self._retrieve(query, top_k=15)
                docs = result.docs
            
            # 3. 自我反思（检查检索质量）
            is_relevant, score = self._reflect_on_retrieval(query, docs)
            
            # 4. 如果不相关且有重试机会，尝试查询增强后重新检索
            retry_count = 0
            while not is_relevant and retry_count < max_retries:
                logger.info(f"检索质量不佳，尝试重新检索 ({retry_count + 1}/{max_retries})")
                
                # 使用LLM优化查询
                optimized_query = self._optimize_query(query)
                result = self._retrieve(optimized_query, top_k=15)
                docs = result.docs
                
                is_relevant, score = self._reflect_on_retrieval(query, docs)
                retry_count += 1
            
            # 5. 生成答案
            answer = self._generate_answer(query, docs)
            
            total_time = time.time() - total_start
            
            return {
                "query": query,
                "answer": answer,
                "success": True,
                "complexity": complexity.value,
                "num_sources": len(docs),
                "sources": [{"pmid": d.get("pmid"), "score": d.get("score")} for d in docs[:5]],
                "relevance_score": score,
                "steps": [{"type": s.step_type, "duration_ms": s.duration_ms} for s in self.steps],
                "total_time_ms": total_time * 1000,
                "retries": retry_count if 'retry_count' in dir() else 0
            }
            
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _optimize_query(self, query: str) -> str:
        """使用LLM优化查询"""
        prompt = f"""请将以下医学问题改写为更适合文献检索的查询。
使用标准医学术语，保持简洁。

原问题: {query}

优化后的查询:"""
        
        return self._call_llm(prompt, temperature=0.3)
    
    def query(self, question: str) -> str:
        """简单查询接口"""
        result = self.chat(question)
        return result["answer"]
    
    def reset(self) -> None:
        """重置Agent"""
        self.steps = []
        logger.info("Agent已重置")


# 保持向后兼容
MedicalLlamaAgent = AdaptiveRAGAgent


def main() -> None:
    """测试Adaptive RAG Agent"""
    print("=" * 60)
    print("Adaptive RAG Agent 测试")
    print("=" * 60)
    
    agent = AdaptiveRAGAgent(verbose=True)
    
    test_queries = [
        "什么是糖尿病？",  # 简单
        "糖尿病有哪些常见症状？",  # 中等
        "1型糖尿病和2型糖尿病有什么区别？",  # 复杂
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"问题: {query}")
        print("-" * 60)
        
        result = agent.chat(query)
        
        print(f"复杂度: {result.get('complexity', 'N/A')}")
        print(f"来源数: {result.get('num_sources', 0)}")
        print(f"相关性: {result.get('relevance_score', 0):.1f}")
        print(f"总耗时: {result.get('total_time_ms', 0):.0f}ms")
        print(f"\n答案: {result['answer'][:500]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()
