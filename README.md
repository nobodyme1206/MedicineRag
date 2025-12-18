# 医学知识问答RAG系统

基于检索增强生成(RAG)的医学知识问答系统，支持PubMed文献检索、向量化存储、混合检索和智能问答。

## 🚀 功能特性

- **数据采集**: PubMed医学文献爬虫，支持30个医学主题，断点续传
- **数据处理**: PySpark分布式处理，Parquet列式存储
- **向量化**: BGE中文优化模型，GPU加速
- **向量检索**: Milvus向量数据库，支持百万级向量
- **混合检索**: BM25关键词 + 向量语义融合
- **重排序**: BGE-Reranker精排
- **HyDE**: 假设文档嵌入增强检索

- **缓存**: Redis查询缓存和向量缓存
- **存储**: MongoDB日志存储，MinIO对象存储
- **Web界面**: Gradio交互式问答

## 📁 项目结构

```
medical-rag/
├── config/             # 配置文件
├── data/               # 数据目录
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后数据
│   └── embeddings/    # 向量数据
├── docker/            # Docker配置
├── logs/              # 日志文件
├── models/            # 模型文件
├── src/               # 源代码
│   ├── caching/       # Redis缓存
│   ├── data_processing/  # 数据处理
│   ├── embedding/     # 向量化
│   ├── evaluation/    # 评估模块
│   ├── rag/          # RAG核心
│   ├── retrieval/    # 检索模块
│   ├── storage/      # 存储模块
│   └── utils/        # 工具函数
├── tests/            # 单元测试
├── web/              # Web界面
├── main.py           # 主入口
└── requirements.txt  # 依赖
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo_url>
cd medical-rag

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑.env文件，填入API Key等配置
```

### 3. 启动服务

```bash
# 启动所有Docker服务（Milvus, Redis, MongoDB, Spark等）
docker compose -f docker/docker-compose.yml up -d
```

### 4. 运行Pipeline

```bash
# 完整Pipeline（采集→处理→向量化→入库→评估）
python main.py --full

# 或分步执行
python main.py --collect              # 数据采集
python main.py --process              # 数据处理
python main.py --embed                # 向量化
python main.py --setup-db             # 构建向量库
python main.py --eval                 # 系统评估
python main.py --web                  # 启动Web界面
```

## 📖 命令参考

### 数据采集
```bash
python main.py --collect                    # 采集数据（支持断点续传）
python main.py --collect --clear-checkpoint # 清除进度重新开始
python main.py --collect --workers 5        # 5线程并行
python main.py --collect --max-per-topic 10000  # 每主题最多1万篇
```

### 数据库操作
```bash
python main.py --rebuild              # 重建向量数据库
python main.py --rebuild --resume     # 断点续传重建
```

### 评估
```bash
python main.py --eval                 # 完整评估
python main.py --eval-rag             # 仅RAG检索评估
python main.py --eval-pyspark --scale 10  # PySpark大数据评估(10x数据)
```

### Spark增强
```bash
python main.py --spark-cluster        # 启动Spark集群
python main.py --spark-embed          # Spark分布式向量化
python main.py --incremental          # 启动增量索引
python main.py --cache-prewarm        # 预热Redis缓存
```



## 🔧 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 向量数据库 | Milvus | 向量存储和检索 |
| 消息队列 | Apache Kafka | 数据流解耦、异步处理 |
| 任务调度 | Apache Airflow | Pipeline编排、定时调度 |
| 缓存 | Redis | 查询缓存、向量缓存 |
| 文档存储 | MongoDB | 日志、评估结果 |
| 对象存储 | MinIO | 模型备份、数据备份 |
| 大数据处理 | PySpark | 分布式数据处理 |
| 向量化 | BGE-small-zh | 中文文本向量化 |
| 重排序 | BGE-Reranker | 检索结果精排 |
| LLM | Qwen2.5-7B | 答案生成 |
| Web框架 | Gradio | 交互界面 |

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| Precision@5 | 0.667 |
| MRR | 0.725 |
| Hit Rate | 0.861 |
| 综合评分 | 80.7/100 |
| 平均延迟 | <500ms |
| 向量数量 | 363,464 |

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 跳过慢速测试
pytest tests/ -v -m "not slow"

# 运行特定测试
pytest tests/test_core.py::TestTextEmbedder -v
```

## 📝 配置说明

主要配置项在 `.env` 文件中：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| SILICONFLOW_API_KEY | LLM API密钥 | - |
| EMBEDDING_DEVICE | 向量化设备 | cuda |
| MILVUS_HOST | Milvus地址 | localhost |
| REDIS_HOST | Redis地址 | localhost |

## 🐳 Docker部署

```bash
# 一键启动所有服务
docker compose -f docker/docker-compose.yml up -d

# 查看服务状态
docker compose -f docker/docker-compose.yml ps

# 停止服务
docker compose -f docker/docker-compose.yml down
```

服务端口：
- Milvus: 19530
- Redis: 6379
- MongoDB: 27017
- MinIO: 9000 (API), 9001 (Console)
- Spark Master: 8080
- Kafka: 9092
- Kafka UI: 8082
- Airflow: 8081
- Gradio: 7860

## 🔄 Kafka + Airflow 集成

### 启动Kafka和Airflow服务

```bash
# 启动Kafka + Airflow
python main.py --kafka-start

# 创建Kafka Topics
python main.py --kafka-topics

# 查看Kafka统计
python main.py --kafka-stats
```

### 使用Kafka集成的Pipeline

```bash
# Kafka集成爬虫（爬取数据发送到Kafka）
python main.py --kafka-crawl

# 启动数据处理消费者
python main.py --kafka-consumer processor

# 启动向量化消费者
python main.py --kafka-consumer embedder

# 运行完整Kafka Pipeline
python main.py --kafka-pipeline
```

### Airflow DAG

访问 http://localhost:8081 (admin/admin) 查看和管理DAG：

- `medical_rag_daily`: 每日增量更新（凌晨2点）
- `medical_rag_weekly_eval`: 每周评估（周日6点）
- `medical_rag_full_pipeline`: 手动触发完整Pipeline

### 架构对比

**引入前（串行）:**
```
爬虫 → 等待 → 处理 → 等待 → 向量化 → 等待 → 入库
```

**引入后（并行）:**
```
爬虫 ──→ Kafka ──→ 处理消费者×N ──→ Kafka ──→ 向量化消费者×N ──→ Milvus
         ↑                                    ↑
      消息持久化                           多实例并行
```

**效果提升:**
- 吞吐量: 3-5倍提升
- 容错性: 支持断点续传
- 可扩展: 增加消费者即可水平扩展
- 可观测: Airflow UI + Kafka UI

## 📄 License

MIT License
