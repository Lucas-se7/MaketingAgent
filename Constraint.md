# 实现约束清单

> 本文件是 ARCHITECTURE.md 的实现指南，确保代码实现过程中遵循架构设计，不偏离 MVP 目标。

---

## 核心理念

**这是一个 MVP**，结合软件工程思想 + Agent 开发：
- 软件工程：可维护性、可测试性、错误处理、边界保护
- Agent 开发：LLM 调用、状态管理、Tool Calling、反思闭环

**不追求的功能**：多租户、复杂权限管理、微服务架构、Kubernetes 部署

---

## 状态管理

- 使用 `pydantic.BaseModel` 管理所有状态，不用 `TypedDict` 或 `dataclass`
- 状态流转：`planning` → `generating` → `reviewing` → `reflection` → 循环或结束
- `status`: `running` → `waiting_confirm` → `completed` / `failed`

---

## 迭代控制

- `max_iterations = 3`（可配置），超过上限直接结束，标记 `status=failed`
- 收敛检测：连续两次反思结果相似度 ≥ 0.9 时提前终止，标记 `converged_failed`

---

## 关键安全约束

| 约束 | 措施 |
|------|------|
| 提示词注入 | `sanitize_input()` 过滤外部输入 |
| 敏感内容 | Generator + Reviewer 双重拦截 |
| 死循环 | `max_iterations` 熔断 + 收敛检测 |
| 用户确认超时 | 30 分钟超时，释放 SSE 和数据库连接 |
| API 限流 | Rate Limiting 装饰器 |
| 工具调用异常 | `call_tool_safely()` 统一捕获 |
| 配置校验 | Pydantic `field_validator` 启动时校验环境变量 |

---

## 前端交互

- 使用 SSE（Server-Sent Events），禁止轮询

---

## 数据库

- SQLite 用于 MVP，写入使用事务
- 切换 PostgreSQL 只需改 `DATABASE_URL`

---

## 测试策略

| 层级 | 重点 | 工具 | 运行频率 |
|------|------|------|---------|
| 单元测试 | State 序列化、熔断逻辑、sanitize_input、收敛判据 | `pytest` | 每次 commit |
| 节点测试 | 单节点 I/O（mock LLM）、工具调用错误隔离 | `pytest` + mock | 每次 commit |
| 集成测试 | 完整工作流 1 次迭代（真实 LLM） | `pytest` + 真实 API Key | 每次 PR |
| E2E | UI 交互、SSE 事件接收 | `Playwright` | 发版前 |

---

## 警告清单

| 场景 | 风险 | 防御措施 |
|------|------|----------|
| 用户输入超长 | 上下文溢出 | `sanitize_input()` 截断 |
| 提示词注入 | Prompt 被劫持 | 输入过滤 |
| LLM 幻觉 | 生成违规内容 | 双重敏感词拦截 |
| 死循环 | 无限迭代烧钱 | `max_iterations` 熔断 |
| 用户确认超时 | 连接不释放 | 30 分钟超时 |
| API 被刷 | API Key 被封 | Rate Limiting |
| 工具调用失败 | Agent 崩溃 | `call_tool_safely()` |
| 启动时无校验 | 运行中才发现配置错误 | `field_validator` |
| 缺少测试覆盖 | 改动引入回归 | 四层测试策略 |

---

> 最后提醒：**MVP 的核心是跑通工作流，不要过度设计**。
> 遇到复杂需求，先问自己："这是 MVP 必须的吗？"