# -*- coding: utf-8 -*-
"""
FastAPI应用 - 医学知识问答API
替代Gradio，提供RESTful API接口
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import LOGS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("api", LOGS_DIR / "api.log")

# 全局RAG系统实例
rag_system = None
medical_agent = None
conversation_managers: Dict[str, Any] = {}


# ==================== Pydantic模型 ====================

class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=10, ge=1, le=50, description="检索文档数量")
    use_rewrite: bool = Field(default=False, description="是否使用查询改写")
    session_id: Optional[str] = Field(default=None, description="会话ID（多轮对话）")


class AnswerResponse(BaseModel):
    """答案响应"""
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, float]
    session_id: Optional[str] = None


class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="hybrid", pattern="^(bm25|vector|hybrid)$")


class RetrievalResponse(BaseModel):
    """检索响应"""
    results: List[Dict[str, Any]]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    services: Dict[str, str]
    version: str = "1.0.0"


class EvaluationRequest(BaseModel):
    """评估请求"""
    mode: str = Field(default="rag", pattern="^(rag|distributed|full)$")
    samples: int = Field(default=100, ge=10, le=1000)


class BatchQuestionRequest(BaseModel):
    """批量问题请求"""
    questions: List[str] = Field(..., min_items=1, max_items=20, description="问题列表（最多20个）")
    top_k: int = Field(default=10, ge=1, le=50, description="每个问题检索文档数量")
    use_semantic_cache: bool = Field(default=True, description="是否使用语义缓存")


class BatchAnswerResponse(BaseModel):
    """批量答案响应"""
    results: List[Dict[str, Any]]
    total: int
    success_count: int
    failed_count: int
    total_time: float
    cache_hits: int


class AgentRequest(BaseModel):
    """Agent请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    max_steps: int = Field(default=5, ge=1, le=10, description="最大执行步数")
    verbose: bool = Field(default=False, description="是否返回详细步骤")


class AgentResponse(BaseModel):
    """Agent响应"""
    query: str
    answer: str
    steps: Optional[List[Dict[str, Any]]] = None
    num_steps: int
    success: bool


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_system, medical_agent
    
    logger.info("🚀 启动FastAPI应用...")
    
    # 初始化RAG系统
    try:
        from src.rag.rag_system import RAGSystem
        rag_system = RAGSystem()
        logger.info("✅ RAG系统初始化成功")
    except Exception as e:
        logger.error(f"❌ RAG系统初始化失败: {e}")
        rag_system = None
    
    # 初始化Agent - 复用RAG系统组件，避免重复初始化
    try:
        from src.agent.llama_agent import MedicalLlamaAgent
        # 传入已初始化的RAG系统，避免重复加载模型
        medical_agent = MedicalLlamaAgent(verbose=False, rag_system=rag_system)
        logger.info("✅ Medical Agent初始化成功（复用RAG系统）")
    except Exception as e:
        logger.error(f"❌ Medical Agent初始化失败: {e}")
        import traceback
        traceback.print_exc()
        medical_agent = None
    
    yield
    
    # 清理资源
    logger.info("🛑 关闭FastAPI应用...")
    conversation_managers.clear()


# ==================== 创建应用 ====================

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="医学知识问答API",
        description="基于RAG的智能医学文献检索与问答系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ==================== API端点 ====================

@app.get("/", tags=["Root"])
async def root():
    """根路径 - API信息"""
    return {
        "message": "医学知识问答API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """健康检查"""
    from src.monitoring.metrics import health_checker
    
    services = {}
    
    # 检查各服务状态
    services["rag"] = "healthy" if rag_system else "unavailable"
    services["redis"] = "healthy" if health_checker.check_redis() else "unavailable"
    services["milvus"] = "healthy" if health_checker.check_milvus() else "unavailable"
    
    overall = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(status=overall, services=services)


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """获取Prometheus指标"""
    from src.monitoring.metrics import metrics_collector, CONTENT_TYPE_LATEST
    
    return StreamingResponse(
        iter([metrics_collector.get_metrics()]),
        media_type=CONTENT_TYPE_LATEST if 'CONTENT_TYPE_LATEST' in dir() else "text/plain"
    )


@app.post("/api/v1/ask", response_model=AnswerResponse, tags=["QA"])
async def ask_question(request: QuestionRequest):
    """
    问答接口
    
    - 支持单轮/多轮对话
    - 支持查询改写
    - 返回答案和来源
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    start_time = time.time()
    
    try:
        # 处理多轮对话
        query = request.question
        if request.session_id:
            from src.rag.conversation import ConversationManager
            
            if request.session_id not in conversation_managers:
                conversation_managers[request.session_id] = ConversationManager()
            
            manager = conversation_managers[request.session_id]
            query = manager.get_context_for_query(query)
        
        # 查询改写
        if request.use_rewrite:
            try:
                from src.rag.query_rewriter import QueryRewriter
                rewriter = QueryRewriter()
                query = rewriter.rewrite(query)
            except Exception as e:
                logger.warning(f"查询改写失败: {e}")
        
        # 执行RAG
        result = rag_system.answer(query, return_contexts=True)
        
        # 添加引用
        from src.rag.citation import CitationManager
        citation_mgr = CitationManager()
        cited = citation_mgr.add_citations_to_answer(
            result["answer"], 
            result.get("contexts", [])
        )
        
        # 更新对话历史
        if request.session_id and request.session_id in conversation_managers:
            manager = conversation_managers[request.session_id]
            manager.add_message("user", request.question)
            manager.add_message("assistant", cited.answer)
        
        total_time = time.time() - start_time
        
        return AnswerResponse(
            answer=cited.answer + cited.get_references_section(),
            sources=[c.to_dict() for c in cited.citations],
            metrics={
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "total_time": total_time
            },
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail="问答服务暂时不可用，请稍后重试")


@app.post("/api/v1/ask/batch", response_model=BatchAnswerResponse, tags=["QA"])
async def ask_batch_questions(request: BatchQuestionRequest):
    """
    批量问答接口
    
    - 支持最多20个问题并发处理
    - 支持语义缓存加速
    - 返回所有答案和统计信息
    """
    import asyncio
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    start_time = time.time()
    results = []
    cache_hits = 0
    success_count = 0
    failed_count = 0
    
    # 复用RAG系统的语义缓存，避免重复初始化
    semantic_cache = None
    if request.use_semantic_cache and rag_system.semantic_cache:
        semantic_cache = rag_system.semantic_cache
    
    async def process_question(question: str, index: int) -> Dict:
        """处理单个问题"""
        nonlocal cache_hits, success_count, failed_count
        
        try:
            q_start = time.time()
            
            # 检查语义缓存
            if semantic_cache:
                cached = semantic_cache.get(question)
                if cached:
                    cache_hits += 1
                    success_count += 1
                    return {
                        "index": index,
                        "question": question,
                        "answer": cached["answer"],
                        "sources": cached.get("contexts", [])[:3],
                        "cache_hit": True,
                        "similarity": cached.get("similarity", 1.0),
                        "time": time.time() - q_start
                    }
            
            # 执行RAG
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: rag_system.answer(question, return_contexts=True)
            )
            
            # 存入语义缓存
            if semantic_cache:
                semantic_cache.set(
                    question,
                    result["answer"],
                    result.get("contexts", []),
                    {"retrieval_time": result.get("retrieval_time", 0)}
                )
            
            success_count += 1
            return {
                "index": index,
                "question": question,
                "answer": result["answer"],
                "sources": [{"pmid": c["pmid"], "score": c["score"]} for c in result.get("contexts", [])[:3]],
                "cache_hit": False,
                "time": result.get("total_time", time.time() - q_start)
            }
            
        except Exception as e:
            failed_count += 1
            logger.error(f"批量问答失败 [{index}]: {e}")
            return {
                "index": index,
                "question": question,
                "answer": f"处理失败: {str(e)}",
                "sources": [],
                "cache_hit": False,
                "error": True,
                "time": 0
            }
    
    # 并发处理所有问题
    tasks = [process_question(q, i) for i, q in enumerate(request.questions)]
    results = await asyncio.gather(*tasks)
    
    # 按原始顺序排序
    results.sort(key=lambda x: x["index"])
    
    total_time = time.time() - start_time
    
    return BatchAnswerResponse(
        results=results,
        total=len(request.questions),
        success_count=success_count,
        failed_count=failed_count,
        total_time=total_time,
        cache_hits=cache_hits
    )


@app.post("/api/v1/ask/stream", tags=["QA"])
async def ask_question_stream(request: QuestionRequest):
    """
    流式问答接口
    
    返回Server-Sent Events流
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            from src.rag.streaming import StreamingRAG
            import json
            
            streaming_rag = StreamingRAG(rag_system)
            
            for item in streaming_rag.answer_stream(request.question, top_k=request.top_k):
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Retrieval"])
async def retrieve_documents(request: RetrievalRequest):
    """
    文档检索接口
    
    支持BM25、向量、混合检索
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    start_time = time.time()
    
    try:
        results = rag_system.retrieve(
            request.query, 
            top_k=request.top_k
        )
        
        latency = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            results=results,
            total=len(results),
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail="检索服务暂时不可用，请稍后重试")


@app.post("/api/v1/rewrite", tags=["Utils"])
async def rewrite_query(query: str = Query(..., min_length=1)):
    """查询改写接口"""
    try:
        # 复用RAG系统的查询改写器
        if rag_system and rag_system.query_rewriter:
            rewriter = rag_system.query_rewriter
        else:
            from src.rag.query_rewriter import QueryRewriter
            rewriter = QueryRewriter()
        
        rewritten = rewriter.rewrite(query) if hasattr(rewriter, 'rewrite') else query
        expanded = rewriter.expand_query(query) if hasattr(rewriter, 'expand_query') else []
        
        return {
            "original": query,
            "rewritten": rewritten,
            "expanded_terms": expanded
        }
        
    except Exception as e:
        logger.error(f"查询改写失败: {e}")
        raise HTTPException(status_code=500, detail="查询改写服务暂时不可用")


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """清除会话历史"""
    if session_id in conversation_managers:
        conversation_managers[session_id].clear()
        del conversation_managers[session_id]
        return {"message": f"会话 {session_id} 已清除"}
    
    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/api/v1/session/{session_id}/history", tags=["Session"])
async def get_session_history(session_id: str):
    """获取会话历史"""
    if session_id not in conversation_managers:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    manager = conversation_managers[session_id]
    return {
        "session_id": session_id,
        "history": [msg.to_dict() for msg in manager.history]
    }


@app.post("/api/v1/evaluate", tags=["Evaluation"])
async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    运行系统评估（后台任务）
    """
    def evaluate_task(mode: str, samples: int):
        try:
            if mode == "rag":
                from src.evaluation.rag_evaluator import RAGEvaluator
                evaluator = RAGEvaluator()
                evaluator.load_pubmedqa(max_samples=samples)
                return evaluator.run_evaluation()
            elif mode == "distributed":
                from src.evaluation.distributed_evaluator import DistributedEvaluator
                evaluator = DistributedEvaluator()
                return evaluator.run_evaluation()
        except Exception as e:
            logger.error(f"评估失败: {e}")
    
    background_tasks.add_task(evaluate_task, request.mode, request.samples)
    
    return {"message": f"评估任务已启动 (mode={request.mode}, samples={request.samples})"}


@app.get("/api/v1/stats", tags=["System"])
async def get_system_stats():
    """获取系统统计信息"""
    stats = {
        "active_sessions": len(conversation_managers),
        "rag_status": "ready" if rag_system else "not_ready",
        "agent_status": "ready" if medical_agent else "not_ready"
    }
    
    # 尝试获取Milvus统计
    try:
        from pymilvus import connections, Collection
        from config.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
        
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        stats["documents_indexed"] = collection.num_entities
        connections.disconnect("default")
    except:
        stats["documents_indexed"] = "unknown"
    
    return stats


@app.get("/api/v1/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """获取缓存统计信息"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    try:
        return rag_system.get_cache_stats()
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取缓存统计失败")


@app.post("/api/v1/cache/prewarm", tags=["Cache"])
async def prewarm_cache(queries: List[str] = None, background_tasks: BackgroundTasks = None):
    """
    预热缓存（热门查询）
    
    - 如果不提供queries，使用默认的医学热门查询
    - 后台异步执行
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    # 默认热门医学查询
    default_queries = [
        "What are the symptoms of diabetes?",
        "How to prevent cardiovascular disease?",
        "What are the common treatments for cancer?",
        "What causes hypertension?",
        "How is COVID-19 transmitted?",
        "What are the risk factors for stroke?",
        "How to manage obesity?",
        "What are the symptoms of Alzheimer's disease?",
        "How to treat asthma?",
        "What causes arthritis?"
    ]
    
    queries_to_prewarm = queries or default_queries
    
    def prewarm_task():
        try:
            return rag_system.prewarm_hot_queries(queries_to_prewarm)
        except Exception as e:
            logger.error(f"预热失败: {e}")
    
    background_tasks.add_task(prewarm_task)
    
    return {
        "message": f"预热任务已启动",
        "queries_count": len(queries_to_prewarm)
    }


@app.delete("/api/v1/cache/clear", tags=["Cache"])
async def clear_cache(cache_type: str = Query(default="all", pattern="^(all|semantic|vector|query)$")):
    """
    清空缓存
    
    - all: 清空所有缓存
    - semantic: 仅清空语义缓存
    - vector: 仅清空向量缓存
    - query: 仅清空查询缓存
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG系统未就绪")
    
    try:
        cleared = {}
        
        if cache_type in ["all", "semantic"] and rag_system.semantic_cache:
            cleared["semantic"] = rag_system.semantic_cache.clear()
        
        if cache_type in ["all", "query"] and rag_system.redis_cache:
            cleared["query"] = rag_system.redis_cache.clear_cache("query:*")
        
        if cache_type in ["all", "vector"] and rag_system.redis_cache:
            cleared["vector"] = rag_system.redis_cache.clear_cache("vector:*")
        
        return {"message": "缓存已清空", "cleared": cleared}
        
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail="清空缓存失败")


# ==================== Agent API ====================

@app.post("/api/v1/agent", response_model=AgentResponse, tags=["Agent"])
async def agent_query(request: AgentRequest):
    """
    Agent智能问答接口
    
    使用Adaptive RAG Agent，支持智能路由、查询分解、自我反思
    """
    if not medical_agent:
        raise HTTPException(status_code=503, detail="Agent未就绪")
    
    try:
        result = medical_agent.chat(request.query)
        
        response = AgentResponse(
            query=result["query"],
            answer=result["answer"],
            steps=result.get("steps"),
            num_steps=result.get("num_sources", 0),
            success=result.get("success", True)
        )
        return response
        
    except Exception as e:
        logger.error(f"Agent执行失败: {e}")
        raise HTTPException(status_code=500, detail="智能问答服务暂时不可用，请稍后重试")


@app.get("/api/v1/agent/tools", tags=["Agent"])
async def list_agent_tools():
    """列出Agent可用的工具"""
    if not medical_agent:
        raise HTTPException(status_code=503, detail="Agent未就绪")
    
    tools = [
        {"name": "search_medical_literature", "description": "搜索PubMed医学文献数据库"},
        {"name": "optimize_query", "description": "优化医学查询"},
        {"name": "explain_medical_term", "description": "解释医学术语"}
    ]
    
    return {"tools": tools, "total": len(tools)}


# ==================== 启动入口 ====================

def main():
    """启动FastAPI服务"""
    from config.config import GRADIO_SERVER_NAME, GRADIO_PORT
    
    logger.info("="*50)
    logger.info("启动医学知识问答API服务")
    logger.info("="*50)
    
    uvicorn.run(
        "src.api.main:app",
        host=GRADIO_SERVER_NAME,
        port=GRADIO_PORT,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
