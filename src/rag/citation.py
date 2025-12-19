#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案溯源模块
为生成的答案标注来源文档，增加可信度
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("citation", LOGS_DIR / "citation.log")


@dataclass
class Citation:
    """引用信息"""
    index: int
    pmid: str
    title: str
    snippet: str
    score: float
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "pmid": self.pmid,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "url": self.url or f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        }


@dataclass
class CitedAnswer:
    """带引用的答案"""
    answer: str
    citations: List[Citation]
    citation_map: Dict[int, Citation]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "num_citations": len(self.citations)
        }
    
    def get_formatted_answer(self) -> str:
        """获取格式化的答案（带引用链接）"""
        return self.answer
    
    def get_references_section(self) -> str:
        """获取参考文献部分"""
        if not self.citations:
            return ""
        
        lines = ["\n\n---\n**参考文献:**"]
        for c in self.citations:
            lines.append(f"[{c.index}] {c.title} (PMID: {c.pmid})")
        return "\n".join(lines)


class CitationManager:
    """引用管理器"""
    
    def __init__(self) -> None:
        self.citation_pattern = re.compile(r'\[(?:Document\s*)?(\d+)\]', re.IGNORECASE)
    
    def create_citations(self, contexts: List[Dict]) -> List[Citation]:
        """
        从检索上下文创建引用列表
        
        Args:
            contexts: 检索到的上下文
            
        Returns:
            引用列表
        """
        citations = []
        for i, ctx in enumerate(contexts):
            citation = Citation(
                index=i + 1,
                pmid=ctx.get("pmid", ""),
                title=ctx.get("title", f"Document {i+1}"),
                snippet=self._extract_snippet(ctx.get("text", "")),
                score=ctx.get("score", 0.0)
            )
            citations.append(citation)
        return citations
    
    def _extract_snippet(self, text: str, max_length: int = 200) -> str:
        """提取文本片段"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + "..."
    
    def add_citations_to_answer(
        self,
        answer: str,
        contexts: List[Dict]
    ) -> CitedAnswer:
        """
        为答案添加引用标注
        
        Args:
            answer: 原始答案
            contexts: 检索上下文
            
        Returns:
            带引用的答案
        """
        citations = self.create_citations(contexts)
        citation_map = {c.index: c for c in citations}
        
        # 检查答案中已有的引用
        found_indices = set()
        for match in self.citation_pattern.finditer(answer):
            idx = int(match.group(1))
            if idx in citation_map:
                found_indices.add(idx)
        
        # 过滤只保留被引用的文档
        used_citations = [c for c in citations if c.index in found_indices]
        
        # 如果没有引用，添加top-3作为参考
        if not used_citations and citations:
            used_citations = citations[:3]
            # 在答案末尾添加引用标记
            refs = ", ".join([f"[{c.index}]" for c in used_citations])
            answer = f"{answer}\n\n*参考来源: {refs}*"
        
        return CitedAnswer(
            answer=answer,
            citations=used_citations,
            citation_map={c.index: c for c in used_citations}
        )
    
    def format_contexts_for_prompt(
        self,
        contexts: List[Dict],
        max_contexts: int = 10
    ) -> str:
        """
        格式化上下文用于LLM提示词
        
        Args:
            contexts: 检索上下文
            max_contexts: 最大上下文数
            
        Returns:
            格式化的上下文文本
        """
        lines = []
        for i, ctx in enumerate(contexts[:max_contexts]):
            pmid = ctx.get("pmid", "")
            text = ctx.get("text", "")
            lines.append(f"[Document {i+1}] (PMID: {pmid})\n{text}")
        return "\n\n".join(lines)
    
    def extract_cited_indices(self, answer: str) -> List[int]:
        """
        从答案中提取引用的文档索引
        
        Args:
            answer: 答案文本
            
        Returns:
            引用的索引列表
        """
        indices = []
        for match in self.citation_pattern.finditer(answer):
            idx = int(match.group(1))
            if idx not in indices:
                indices.append(idx)
        return indices


class SourceTracker:
    """来源追踪器 - 追踪答案中每个陈述的来源"""
    
    def __init__(self) -> None:
        self.citation_manager = CitationManager()
    
    def analyze_answer(
        self,
        answer: str,
        contexts: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析答案的来源覆盖情况
        
        Args:
            answer: 答案文本
            contexts: 检索上下文
            
        Returns:
            分析结果
        """
        citations = self.citation_manager.create_citations(contexts)
        cited_indices = self.citation_manager.extract_cited_indices(answer)
        
        # 计算覆盖率
        coverage = len(cited_indices) / len(contexts) if contexts else 0
        
        # 检查未引用的高分文档
        uncited_high_score = [
            c for c in citations 
            if c.index not in cited_indices and c.score > 0.7
        ]
        
        return {
            "total_contexts": len(contexts),
            "cited_count": len(cited_indices),
            "cited_indices": cited_indices,
            "coverage_ratio": coverage,
            "uncited_high_score": [c.to_dict() for c in uncited_high_score],
            "has_citations": len(cited_indices) > 0
        }
    
    def generate_source_report(
        self,
        answer: str,
        contexts: List[Dict]
    ) -> str:
        """
        生成来源报告
        
        Args:
            answer: 答案文本
            contexts: 检索上下文
            
        Returns:
            来源报告文本
        """
        analysis = self.analyze_answer(answer, contexts)
        
        lines = [
            "## 来源分析报告",
            f"- 检索文档数: {analysis['total_contexts']}",
            f"- 引用文档数: {analysis['cited_count']}",
            f"- 覆盖率: {analysis['coverage_ratio']:.1%}",
        ]
        
        if analysis['uncited_high_score']:
            lines.append("\n### 未引用的高相关文档:")
            for doc in analysis['uncited_high_score'][:3]:
                lines.append(f"- [{doc['index']}] {doc['title']} (score: {doc['score']:.2f})")
        
        return "\n".join(lines)


def main() -> None:
    """测试答案溯源"""
    print("=" * 50)
    print("答案溯源测试")
    print("=" * 50)
    
    manager = CitationManager()
    
    # 模拟上下文
    contexts = [
        {"pmid": "123", "title": "Diabetes Overview", "text": "Diabetes is...", "score": 0.9},
        {"pmid": "456", "title": "Treatment Options", "text": "Treatment includes...", "score": 0.8},
        {"pmid": "789", "title": "Prevention", "text": "Prevention methods...", "score": 0.7}
    ]
    
    # 模拟答案
    answer = "Diabetes is a chronic condition [Document 1]. Treatment options include medication [Document 2]."
    
    cited_answer = manager.add_citations_to_answer(answer, contexts)
    
    print(f"\n原始答案:\n{answer}")
    print(f"\n引用数量: {len(cited_answer.citations)}")
    print(f"\n参考文献:{cited_answer.get_references_section()}")
    
    # 来源分析
    tracker = SourceTracker()
    report = tracker.generate_source_report(answer, contexts)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
