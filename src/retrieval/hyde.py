#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE (Hypothetical Document Embeddings) 模块
通过LLM生成假设性答案文档，然后用该文档进行检索
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from config.config import *
from src.utils.logger import setup_logger

logger = setup_logger("hyde", LOGS_DIR / "hyde.log")


class HyDE:
    """HyDE假设文档嵌入：通过生成假设答案来增强检索"""
    
    def __init__(self):
        """初始化HyDE模块"""
        logger.info("初始化HyDE模块...")
        self.llm_client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
        logger.info("✅ HyDE模块初始化完成")
    
    def generate_hypothetical_document(self, query: str, num_docs: int = 1) -> list:
        """
        为查询生成假设性答案文档
        
        Args:
            query: 用户查询
            num_docs: 生成的假设文档数量
            
        Returns:
            假设文档列表
        """
        system_prompt = """You are a medical knowledge base. Generate a hypothetical document that would perfectly answer the given medical question. 

Write as if you are a medical research paper or textbook passage. Be factual, comprehensive, and use medical terminology appropriately.

Requirements:
1. Write 150-250 words
2. Include relevant medical terms and concepts
3. Be informative and authoritative
4. Structure the response as a cohesive passage
5. Focus on the specific medical topic asked

Do NOT include disclaimers or caveats. Write directly as an authoritative medical source."""

        hypothetical_docs = []
        
        for i in range(num_docs):
            user_prompt = f"""Medical Question: {query}

Generate a hypothetical medical document that would perfectly answer this question:"""
            
            try:
                response = self.llm_client.chat.completions.create(
                    model=SILICONFLOW_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7 if num_docs > 1 else 0.5,  # 多文档时增加多样性
                    max_tokens=400
                )
                
                hypo_doc = response.choices[0].message.content.strip()
                hypothetical_docs.append(hypo_doc)
                logger.info(f"HyDE生成假设文档 {i+1}/{num_docs}: {len(hypo_doc)} 字符")
                
            except Exception as e:
                logger.error(f"HyDE生成失败: {e}")
                # 失败时返回原查询
                hypothetical_docs.append(query)
        
        return hypothetical_docs
    
    def get_hyde_query(self, query: str) -> str:
        """
        获取HyDE增强的查询（单文档版本）
        
        Args:
            query: 原始查询
            
        Returns:
            假设文档（用于嵌入检索）
        """
        docs = self.generate_hypothetical_document(query, num_docs=1)
        return docs[0] if docs else query
    
    def get_multiple_hyde_queries(self, query: str, num_docs: int = 3) -> list:
        """
        获取多个HyDE假设文档（用于集成检索）
        
        Args:
            query: 原始查询
            num_docs: 假设文档数量
            
        Returns:
            假设文档列表
        """
        return self.generate_hypothetical_document(query, num_docs=num_docs)


if __name__ == "__main__":
    # 测试HyDE模块
    print("=" * 70)
    print("🔮 HyDE模块测试")
    print("=" * 70)
    
    hyde = HyDE()
    
    test_query = "What are the symptoms of type 2 diabetes?"
    print(f"\n📝 测试查询: {test_query}")
    
    hypo_doc = hyde.get_hyde_query(test_query)
    print(f"\n📄 生成的假设文档:")
    print("-" * 50)
    print(hypo_doc[:500] + "..." if len(hypo_doc) > 500 else hypo_doc)
    print("-" * 50)
    
    print("\n✅ HyDE模块测试完成!")
