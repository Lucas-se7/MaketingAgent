# Marketing Agent

营销内容自动生成 Agent - MVP

## 技术栈

- **工作流**: LangGraph
- **LLM**: OpenAI SDK (兼容 DeepSeek 等)
- **数据库**: SQLite (MVP) / PostgreSQL (生产)
- **前端**: Streamlit + FastAPI
- **包管理**: uv

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 启动 API 服务
uv run python main.py

# 启动 Streamlit 前端
uv run streamlit run ui/streamlit_app.py
```

## 项目结构

```
marketing_agent/
├── core/
│   ├── agent/          # Agent 核心
│   ├── llm/            # LLM 调用
│   └── tools/          # 工具
├── storage/            # 数据库
├── api/                # API 路由
├── ui/                 # 前端
├── config/             # 配置
└── observability/      # 日志
```
