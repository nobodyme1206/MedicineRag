# 医学RAG系统 - 混合架构

## 架构概览

```
┌─────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│   Vue 3 前端    │────▶│  Java Spring Boot  │────▶│  Python ML 微服务   │
│   (Vite)        │     │  (API Gateway)     │     │  (FastAPI)          │
│   Port: 5173    │     │  Port: 8080        │     │  Port: 7861         │
└─────────────────┘     └────────────────────┘     └─────────────────────┘
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
- Pinia 状态管理
- Vue Router 路由
- Axios HTTP客户端

### Java后端 (backend/)
- Spring Boot 3.2
- Spring WebFlux (WebClient)
- Spring Data Redis
- Spring Data MongoDB
- SpringDoc OpenAPI (Swagger)
- Lombok

### Python ML微服务 (src/)
- FastAPI
- Milvus 向量数据库
- Sentence Transformers (Embedding)
- BGE Reranker
- OpenAI API (LLM)

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
├── backend/                  # Java Spring Boot
│   ├── src/main/java/com/medical/rag/
│   │   ├── controller/      # REST控制器
│   │   ├── service/         # 业务服务
│   │   ├── dto/             # 数据传输对象
│   │   └── config/          # 配置类
│   ├── src/main/resources/
│   │   └── application.yml
│   └── pom.xml
│
├── src/                      # Python ML微服务
│   ├── api/                 # FastAPI接口
│   ├── rag/                 # RAG核心
│   ├── agent/               # Agent实现
│   └── retrieval/           # 检索模块
│
└── docker/                   # Docker配置
```

## 启动方式

### 方式一：使用启动脚本
```bash
# Windows
start-services.bat
```

### 方式二：手动启动

1. **启动基础服务**
```bash
cd docker
docker-compose up -d
```

2. **启动Python ML微服务**
```bash
set NO_PROXY=localhost,127.0.0.1
python main.py --api
```

3. **启动Java后端**
```bash
cd backend
mvn spring-boot:run
```

4. **启动Vue前端**
```bash
cd frontend
npm install
npm run dev
```

## API端点

### Java后端 (http://localhost:8080)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /api/v1/ask | RAG问答 |
| POST | /api/v1/agent | Agent问答 |
| POST | /api/v1/retrieve | 文献检索 |
| GET | /api/v1/stats | 系统统计 |

### Swagger文档
http://localhost:8080/swagger-ui.html

## 数据流

1. 用户在Vue前端输入问题
2. 前端通过Axios发送请求到Java后端
3. Java后端进行参数校验、日志记录
4. Java后端通过WebClient调用Python ML微服务
5. Python执行RAG/Agent逻辑，调用Milvus检索
6. 结果逐层返回到前端展示

## 扩展建议

### Java后端可扩展功能
- 用户认证 (Spring Security + JWT)
- 请求限流 (Bucket4j)
- 缓存优化 (Spring Cache + Redis)
- 日志审计
- 多租户支持

### 前端可扩展功能
- 用户登录/注册
- 历史记录管理
- 收藏夹功能
- 深色模式
- 国际化
