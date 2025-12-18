# -*- coding: utf-8 -*-
"""
Gradio Web界面
"""

import gradio as gr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.rag.rag_system import RAGSystem
from src.utils.logger import setup_logger

logger = setup_logger("web_interface", LOGS_DIR / "web.log")

# 全局RAG系统实例
rag_system = None


def initialize_rag():
    """初始化RAG系统"""
    global rag_system
    if rag_system is None:
        logger.info("初始化RAG系统...")
        try:
            rag_system = RAGSystem()
            logger.info("✅ RAG系统初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ RAG系统初始化失败: {e}")
            return False
    return True


def answer_question(query: str, show_contexts: bool = True):
    """
    回答问题
    
    Args:
        query: 用户问题
        show_contexts: 是否显示检索的上下文
        
    Returns:
        答案文本, 上下文文本, 性能指标
    """
    if not query.strip():
        return "请输入问题", "", ""
    
    if not initialize_rag():
        return "系统未就绪，请检查服务状态", "", ""
    
    try:
        # 调用RAG系统
        result = rag_system.answer(query, return_contexts=True)
        
        # 答案
        answer = result["answer"]
        
        # 上下文
        contexts_text = ""
        if show_contexts and "contexts" in result:
            contexts_text = "\n\n".join([
                f"**[文档 {i+1}]** (相似度: {ctx['score']:.3f})\n"
                f"PMID: {ctx['pmid']}\n"
                f"{ctx['text'][:500]}..."
                for i, ctx in enumerate(result["contexts"])
            ])
        
        # 性能指标
        metrics = f"""**性能指标**
- 检索时间: {result['retrieval_time']:.3f} 秒
- 生成时间: {result['generation_time']:.3f} 秒
- 总耗时: {result['total_time']:.3f} 秒
- 参考文档数: {result['num_contexts']}
"""
        
        return answer, contexts_text, metrics
        
    except Exception as e:
        logger.error(f"问答失败: {e}")
        return f"出错了: {str(e)}", "", ""


def create_interface():
    """创建Gradio界面"""
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', Arial, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(title="医学知识问答系统") as demo:
        
        # 标题
        gr.HTML("""
        <div class="header">
            <h1>🏥 医学知识问答系统</h1>
            <p>基于RAG的智能医学文献检索与问答</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # 输入区域
                query_input = gr.Textbox(
                    label="请输入您的医学问题",
                    placeholder="例如：什么是糖尿病？如何预防心血管疾病？",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🔍 提问", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ 清空", scale=1)
                
                show_contexts_checkbox = gr.Checkbox(
                    label="显示参考文献",
                    value=True
                )
                
                # 示例问题
                gr.Examples(
                    examples=[
                        "什么是糖尿病？有哪些类型？",
                        "如何预防心血管疾病？",
                        "癌症的常见治疗方法有哪些？",
                        "新冠病毒的传播途径是什么？",
                        "阿尔茨海默症的早期症状有哪些？",
                    ],
                    inputs=query_input
                )
            
            with gr.Column(scale=3):
                # 输出区域
                answer_output = gr.Textbox(
                    label="📝 答案",
                    lines=10,
                    interactive=False
                )
                
                metrics_output = gr.Markdown(
                    label="⚡ 性能指标"
                )
        
        # 参考文献（可折叠）
        with gr.Accordion("📚 参考文献", open=False):
            contexts_output = gr.Markdown()
        
        # 系统信息
        with gr.Accordion("ℹ️ 系统信息", open=False):
            gr.Markdown(f"""
            **配置信息**
            - LLM模型: {SILICONFLOW_MODEL}
            - Embedding模型: {EMBEDDING_MODEL_NAME}
            - 向量维度: {EMBEDDING_DIMENSION}
            - 检索Top-K: {RETRIEVAL_TOP_K}
            - 重排序Top-K: {RERANK_TOP_K}
            
            **数据来源**
            - PubMed医学文献数据库
            - 多个医学主题领域
            """)
        
        # 事件绑定
        submit_btn.click(
            fn=answer_question,
            inputs=[query_input, show_contexts_checkbox],
            outputs=[answer_output, contexts_output, metrics_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            inputs=[],
            outputs=[query_input, answer_output, contexts_output, metrics_output]
        )
    
    return demo


def main():
    """启动Web服务"""
    logger.info("="*50)
    logger.info("启动医学知识问答Web服务")
    logger.info("="*50)
    
    # 预初始化RAG系统
    logger.info("预加载RAG系统...")
    if not initialize_rag():
        logger.error("RAG系统初始化失败，请检查:")
        logger.error("1. Milvus是否已启动")
        logger.error("2. 向量数据是否已导入")
        logger.error("3. API Key是否正确")
        return
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    logger.info(f"启动Gradio服务: {GRADIO_SERVER_NAME}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE
    )


if __name__ == "__main__":
    main()
