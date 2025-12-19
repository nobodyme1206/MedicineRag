#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话模块
支持对话历史上下文管理
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("conversation", LOGS_DIR / "conversation.log")


@dataclass
class Message:
    """对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def to_llm_format(self) -> Dict[str, str]:
        """转换为LLM API格式"""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """对话会话"""
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> Message:
        """添加消息"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg
    
    def add_user_message(self, content: str, **kwargs) -> Message:
        """添加用户消息"""
        return self.add_message("user", content, kwargs)
    
    def add_assistant_message(
        self, 
        content: str, 
        contexts: Optional[List[Dict]] = None,
        **kwargs
    ) -> Message:
        """添加助手消息"""
        metadata = kwargs
        if contexts:
            metadata["contexts"] = contexts
        return self.add_message("assistant", content, metadata)
    
    def get_history(self, max_turns: int = 5) -> List[Message]:
        """获取最近的对话历史"""
        return self.messages[-max_turns * 2:] if self.messages else []
    
    def get_llm_messages(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """获取LLM格式的消息历史"""
        history = self.get_history(max_turns)
        return [msg.to_llm_format() for msg in history]
    
    def get_context_summary(self) -> str:
        """生成对话上下文摘要"""
        if not self.messages:
            return ""
        
        summary_parts = []
        for msg in self.messages[-4:]:  # 最近4条
            role = "用户" if msg.role == "user" else "助手"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def clear(self) -> None:
        """清空对话历史"""
        self.messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "metadata": self.metadata
        }


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_conversations: int = 100) -> None:
        """
        初始化对话管理器
        
        Args:
            max_conversations: 最大保存的对话数
        """
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self._counter = 0
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """创建新对话"""
        if conversation_id is None:
            self._counter += 1
            conversation_id = f"conv_{self._counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        conv = Conversation(id=conversation_id)
        self.conversations[conversation_id] = conv
        
        # 清理旧对话
        if len(self.conversations) > self.max_conversations:
            oldest_id = next(iter(self.conversations))
            del self.conversations[oldest_id]
        
        logger.info(f"创建对话: {conversation_id}")
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self.conversations.get(conversation_id)
    
    def get_or_create(self, conversation_id: str) -> Conversation:
        """获取或创建对话"""
        if conversation_id not in self.conversations:
            return self.create_conversation(conversation_id)
        return self.conversations[conversation_id]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """删除对话"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"删除对话: {conversation_id}")
            return True
        return False
    
    def list_conversations(self) -> List[str]:
        """列出所有对话ID"""
        return list(self.conversations.keys())
    
    def export_conversation(self, conversation_id: str) -> Optional[str]:
        """导出对话为JSON"""
        conv = self.get_conversation(conversation_id)
        if conv:
            return json.dumps(conv.to_dict(), ensure_ascii=False, indent=2)
        return None
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息到默认对话"""
        conv = self.get_or_create("default")
        conv.add_message(role, content)
    
    def get_context_for_query(self, query: str) -> str:
        """
        获取带上下文的查询
        
        Args:
            query: 当前用户问题
            
        Returns:
            带历史上下文的增强查询
        """
        conv = self.get_or_create("default")
        
        if len(conv.messages) == 0:
            return query
        
        # 获取最近的对话历史
        recent_messages = conv.messages[-4:]  # 最近4条消息
        
        if not recent_messages:
            return query
        
        # 构建上下文
        context_parts = []
        for msg in recent_messages:
            role = "用户" if msg.role == "user" else "助手"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
        
        context = "\n".join(context_parts)
        
        # 返回增强查询
        return f"对话历史:\n{context}\n\n当前问题: {query}"


class ConversationalRAG:
    """支持多轮对话的RAG系统"""
    
    def __init__(self, rag_system: Any) -> None:
        """
        初始化对话式RAG
        
        Args:
            rag_system: RAG系统实例
        """
        self.rag = rag_system
        self.manager = ConversationManager()
        logger.info("对话式RAG初始化完成")
    
    def chat(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        use_history: bool = True,
        max_history_turns: int = 3
    ) -> Dict[str, Any]:
        """
        对话式问答
        
        Args:
            query: 用户问题
            conversation_id: 对话ID
            use_history: 是否使用历史上下文
            max_history_turns: 最大历史轮数
            
        Returns:
            回答结果
        """
        # 获取或创建对话
        conv = self.manager.get_or_create(conversation_id or "default")
        
        # 添加用户消息
        conv.add_user_message(query)
        
        # 构建带历史的查询
        if use_history and len(conv.messages) > 1:
            context_summary = conv.get_context_summary()
            enhanced_query = f"对话上下文:\n{context_summary}\n\n当前问题: {query}"
        else:
            enhanced_query = query
        
        # 调用RAG系统
        result = self.rag.answer(enhanced_query, return_contexts=True)
        
        # 添加助手消息
        conv.add_assistant_message(
            result["answer"],
            contexts=result.get("contexts", []),
            retrieval_time=result.get("retrieval_time"),
            generation_time=result.get("generation_time")
        )
        
        # 返回结果
        return {
            "conversation_id": conv.id,
            "query": query,
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "history_length": len(conv.messages),
            "retrieval_time": result.get("retrieval_time", 0),
            "generation_time": result.get("generation_time", 0),
            "total_time": result.get("total_time", 0)
        }
    
    def get_history(self, conversation_id: str) -> List[Dict]:
        """获取对话历史"""
        conv = self.manager.get_conversation(conversation_id)
        if conv:
            return [m.to_dict() for m in conv.messages]
        return []
    
    def clear_history(self, conversation_id: str) -> bool:
        """清空对话历史"""
        conv = self.manager.get_conversation(conversation_id)
        if conv:
            conv.clear()
            return True
        return False


def main() -> None:
    """测试多轮对话"""
    # 模拟RAG系统
    class MockRAG:
        def answer(self, query: str, return_contexts: bool = True) -> Dict:
            return {
                "answer": f"这是对'{query}'的回答",
                "contexts": [{"text": "示例上下文", "score": 0.9}],
                "retrieval_time": 0.1,
                "generation_time": 0.5,
                "total_time": 0.6
            }
    
    conv_rag = ConversationalRAG(MockRAG())
    
    # 多轮对话测试
    print("=" * 50)
    print("多轮对话测试")
    print("=" * 50)
    
    result1 = conv_rag.chat("什么是糖尿病?", conversation_id="test")
    print(f"Q1: 什么是糖尿病?")
    print(f"A1: {result1['answer']}")
    
    result2 = conv_rag.chat("它有哪些症状?", conversation_id="test")
    print(f"\nQ2: 它有哪些症状?")
    print(f"A2: {result2['answer']}")
    
    print(f"\n对话历史长度: {result2['history_length']}")


if __name__ == "__main__":
    main()
