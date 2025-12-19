# 医学RAG系统 - 架构文档

## 架构概览

```
┌─────────────────┐     ┌─────────────────────┐
│   Vue 3 前端    │────▶│  Python FastAPI     │
│   (Vite)        │     │  (RAG + Agent)      │
│   Port: 5173    │     │  Port: 7861         │
└─────────────────┘     └─────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Milvus      Redis      MongoDB
                 :19530      :6379      :27017
```

## 技术栈

### 前端 (frontend/)
- Vue 3 + Composition API
- Vite 构建工具
- Element Plus UI组件库
- Vue Router 路由
- Axios HTTP客户端

### Python后端 (src/)
- FastAPI - 高性能异步Web框架
- Milvus - 向量数据库
- Sentence Transformers - Embedding模型
- BGE Reranker - 重排序模型
- OpenAI API - LLM调用

## 目录结构

```
project/
├── frontend/                 # Vue 3 前端
│   ├── src/
│   │   ├── api/             # API调用
│   │   ├── views/           # 页面组件
│   │   ├── router/          # 路由配置
│   │   └── styles/          # 样式
│   ├── package.json
│   └── vite.config.js
│
├── src/                      # Python后端
│   ├── api/                 # FastAPI接口
│   ├── rag/                 # RAG核心模块
│   ├── agent/               # Adaptive RAG Agent
│   ├── retrieval/           # 检索模块
│   ├── embedding/           # 向量化模块
│   ├── caching/             # Redis缓存
│   └── storage/             # 存储模块
│
├── config/                   # 配置文件
│   ├── docker/              # Docker配置
│   └── config.py            # 系统配置
│
└── tests/                    # 测试文件
```

## 启动方式

1. **启动基础服务**
```bash
docker compose -f config/docker/docker-compose.yml up -d
```

2. **启动Python后端**
```bash
python -m src.api.main
```

3. **启动Vue前端**
```bash
cd frontend
npm install
npm run dev
```

## API端点

### FastAPI后端 (http://localhost:7861)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /api/v1/ask | RAG问答 |
| POST | /api/v1/ask/batch | 批量问答 |
| POST | /api/v1/agent | Agent智能问答 |
| POST | /api/v1/retrieve | 文献检索 |
| GET | /api/v1/stats | 系统统计 |
| GET | /api/v1/cache/stats | 缓存统计 |

### API文档
http://localhost:7861/docs

## 数据流

1. 用户在Vue前端输入问题
2. 前端通过Axios发送请求到FastAPI后端
3. FastAPI执行RAG/Agent逻辑
4. 调用Milvus进行向量检索
5. 调用LLM生成答案
6. 结果返回到前端展示

## 扩展建议

### 后端可扩展功能
- 用户认证 (FastAPI + JWT)
- 请求限流 (slowapi)
- 分布式部署 (Gunicorn + Nginx)
- 日志审计
- Prometheus监控

### 前端可扩展功能
- 用户登录/注册
- 历史记录管理
- 收藏夹功能
- 深色模式
- 国际化
