#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式输出模块
支持LLM答案的流式生成
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, List, Dict, Optional, Any, Callable
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P, LOGS_DIR
)
from src.utils.logger import setup_logger
from src.utils.exceptions import GenerationError, handle_errors

logger = setup_logger("streaming", LOGS_DIR / "streaming.log")


class StreamingGenerator:
    """流式答案生成器"""
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        初始化流式生成器
        
        Args:
            api_key: API密钥
        """
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=api_key or SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
        self.model = SILICONFLOW_MODEL
        logger.info("流式生成器初始化完成")
    
    def generate_stream(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        """
        流式生成答案
        
        Args:
            query: 用户问题
            contexts: 检索到的上下文
            system_prompt: 系统提示词
            on_token: 每个token的回调函数
            
        Yields:
            生成的token
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        user_prompt = self._build_user_prompt(query, contexts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if on_token:
                        on_token(token)
                    yield token
                    
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"生成出错: {str(e)}"
    
    def generate_with_sources(
        self,
        query: str,
        contexts: List[Dict],
        on_token: Optional[Callable[[str], None]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成答案并附带来源信息
        
        Args:
            query: 用户问题
            contexts: 检索到的上下文
            on_token: token回调
            
        Yields:
            包含token和元信息的字典
        """
        start_time = time.time()
        full_answer = []
        token_count = 0
        
        # 先yield来源信息
        yield {
            "type": "sources",
            "sources": self._format_sources(contexts)
        }
        
        # 流式生成答案
        for token in self.generate_stream(query, contexts, on_token=on_token):
            full_answer.append(token)
            token_count += 1
            yield {
                "type": "token",
                "token": token,
                "token_count": token_count
            }
        
        # 最后yield完成信息
        yield {
            "type": "done",
            "full_answer": "".join(full_answer),
            "token_count": token_count,
            "generation_time": time.time() - start_time
        }
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """You are a medical knowledge assistant. Answer questions based on the provided context.

Rules:
1. Only use information from the provided documents
2. Cite sources using [Document N] format
3. If information is insufficient, say so
4. Be accurate and professional
5. Respond in the same language as the question"""
    
    def _build_user_prompt(self, query: str, contexts: List[Dict]) -> str:
        """构建用户提示词"""
        context_text = "\n\n".join([
            f"[Document {i+1}] {ctx.get('text', '')}"
            for i, ctx in enumerate(contexts[:10])
        ])
        
        return f"""Reference Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the documents above:"""
    
    def _format_sources(self, contexts: List[Dict]) -> List[Dict[str, Any]]:
        """格式化来源信息"""
        sources = []
        for i, ctx in enumerate(contexts[:5]):
            sources.append({
                "index": i + 1,
                "pmid": ctx.get("pmid", ""),
                "title": ctx.get("title", ""),
                "score": ctx.get("score", 0),
                "snippet": ctx.get("text", "")[:200] + "..."
            })
        return sources


class StreamingRAG:
    """支持流式输出的RAG系统"""
    
    def __init__(self, rag_system: Any) -> None:
        """
        初始化流式RAG
        
        Args:
            rag_system: RAG系统实例
        """
        self.rag = rag_system
        self.generator = StreamingGenerator()
        logger.info("流式RAG初始化完成")
    
    def answer_stream(
        self,
        query: str,
        top_k: int = 10,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式问答
        
        Args:
            query: 用户问题
            top_k: 检索数量
            on_token: token回调
            
        Yields:
            流式结果
        """
        # 检索阶段
        retrieval_start = time.time()
        contexts = self.rag.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        yield {
            "type": "retrieval_done",
            "num_contexts": len(contexts),
            "retrieval_time": retrieval_time
        }
        
        # 生成阶段（流式）
        for item in self.generator.generate_with_sources(query, contexts, on_token):
            yield item
    
    def answer_stream_simple(
        self,
        query: str,
        top_k: int = 10
    ) -> Generator[str, None, None]:
        """
        简单流式问答（只返回token）
        
        Args:
            query: 用户问题
            top_k: 检索数量
            
        Yields:
            生成的token
        """
        contexts = self.rag.retrieve(query, top_k=top_k)
        yield from self.generator.generate_stream(query, contexts)


def main() -> None:
    """测试流式输出"""
    print("=" * 50)
    print("流式输出测试")
    print("=" * 50)
    
    generator = StreamingGenerator()
    
    # 模拟上下文
    mock_contexts = [
        {"text": "Diabetes is a chronic disease...", "pmid": "123", "score": 0.9},
        {"text": "Type 2 diabetes symptoms include...", "pmid": "456", "score": 0.8}
    ]
    
    print("\n流式生成:")
    for token in generator.generate_stream("What is diabetes?", mock_contexts):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
