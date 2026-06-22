"""
Streamlit 前端 - SSE 实时交互
"""
import json
import logging
import requests
import sseclient  # pip install sseclient-py

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="营销内容生成 Agent",
    page_icon="📝",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"


def wait_for_workflow_settle(topic_id: int, max_wait: float = 30.0) -> None:
    """
    等待后台工作流到达稳定状态（waiting_confirm / completed / failed）

    解决竞态条件：用户确认后 st.rerun() 重新打开 SSE 连接时，
    如果后台 LLM 调用尚未完成，SSE 端点从 DB 读取的状态可能是过时的。
    通过在 rerun 前轮询 API，确保 DB 已反映最新状态。

    Args:
        topic_id: 选题 ID
        max_wait: 最大等待时间（秒）
    """
    import time as _time
    _start = _time.time()
    while _time.time() - _start < max_wait:
        try:
            r = requests.get(f"{API_BASE_URL}/api/topics/{topic_id}", timeout=3)
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status != "running":  # waiting_confirm / completed / failed
                    logger.info(f"工作流已稳定: status={status}")
                    return
        except Exception:
            pass
        _time.sleep(0.5)
    logger.warning(f"等待工作流稳定超时 ({max_wait}s)，继续执行")


def listen_sse(topic_id: int):
    """
    监听 SSE 事件，实时更新 UI

    Args:
        topic_id: 选题 ID
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/topics/{topic_id}/events",
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)

        for event in client.events():
            data = json.loads(event.data)
            logger.info(f"SSE event: {data}")

            # 运行中 — 显示进度，用 rerun 保持实时刷新
            if data["status"] == "running":
                st.info(f"⏳ {data['step'].upper()} 处理中...")
                # 不要在此处 rerun —— 让事件循环继续，避免反复断连重建
                # 仅记录当前进度，等待下一个事件
                continue

            # 等待确认
            elif data["status"] == "waiting_confirm":
                st.subheader("📋 请确认当前结果")

                result = data.get("result", "")
                step = data.get("step", "unknown")

                # 用 session_state 计数器保证每次确认 UI 的 key 都不同，
                # 防止 Streamlit 缓存旧值导致驳回后内容不刷新
                round_num = st.session_state.get("confirm_round", 0)

                st.text_area(
                    "结果",
                    value=result,
                    height=200,
                    disabled=True,
                    key=f"waiting_result_{step}_{round_num}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ 通过", type="primary", key=f"btn_approve_{step}_{round_num}"):
                        resp = requests.post(
                            f"{API_BASE_URL}/api/topics/{topic_id}/confirm",
                            json={"approved": True}
                        )
                        if resp.status_code == 200:
                            st.session_state["confirm_round"] = round_num + 1
                            st.session_state["confirmed"] = True
                            wait_for_workflow_settle(topic_id)
                            st.rerun()
                        else:
                            st.error(f"确认失败: {resp.text}")

                with col2:
                    feedback = st.text_input(
                        "驳回理由（请说明问题）",
                        key=f"feedback_input_{step}_{round_num}"
                    )
                    if st.button("❌ 驳回", key=f"btn_reject_{step}_{round_num}", disabled=not feedback):
                        resp = requests.post(
                            f"{API_BASE_URL}/api/topics/{topic_id}/confirm",
                            json={"approved": False, "feedback": feedback}
                        )
                        if resp.status_code == 200:
                            st.session_state["confirm_round"] = round_num + 1
                            st.session_state["confirmed"] = True
                            wait_for_workflow_settle(topic_id)
                            st.rerun()
                        else:
                            st.error(f"确认失败: {resp.text}")

                break

            # 完成
            elif data["status"] == "completed":
                st.success("🎉 内容生成完成！")
                st.text_area(
                    "最终内容",
                    value=data.get("content", ""),
                    height=300,
                    disabled=True,
                    key=f"final_content_{st.session_state.get('confirm_round', 0)}"
                )
                if st.button("🔄 重新开始", key="btn_restart"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                break

            # 失败
            elif data["status"] == "failed":
                st.error(f"❌ 生成失败: {data.get('fail_reason') or '未知错误'}")
                if st.button("🔄 重试", key="btn_retry"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                break

            # 错误
            elif "error" in data:
                st.error(f"错误: {data['error']}")
                break

    except requests.exceptions.ConnectionError:
        st.error(f"无法连接到 API 服务，请确保后端已启动 ({API_BASE_URL})")
    except Exception as e:
        logger.error(f"SSE error: {e}")
        st.error(f"连接错误: {e}")


def main():
    st.title("📝 营销内容生成 Agent")

    st.markdown("""
    ### 工作流程
    1. **策划** - AI 分析选题，制定营销方案
    2. **生成** - 根据方案生成营销文案
    3. **审核** - AI 审核内容质量

    每一步都需要您确认后才会继续。
    """)

    st.divider()

    # 选题输入
    user_input = st.text_input(
        "请输入选题",
        placeholder="例如：推广我们的新产品——智能手表X，目标受众是年轻人...",
        help="描述您想要推广的产品或主题"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        start_button = st.button("🚀 开始生成", type="primary", key="btn_start")

    # 开始生成
    if start_button and user_input:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/topics",
                json={"user_input": user_input},
                timeout=10
            )

            if resp.status_code == 200:
                topic_id = resp.json()["topic_id"]
                st.session_state["topic_id"] = topic_id
                st.session_state["user_input"] = user_input
                st.session_state["started"] = True
                st.session_state["confirm_round"] = 0  # 初始化确认轮次
                st.rerun()
            elif resp.status_code == 429:
                st.warning("⚠️ 请求过于频繁，请稍后再试")
            else:
                st.error(f"创建选题失败: {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"无法连接到 API 服务，请确保后端已启动 ({API_BASE_URL})")
        except Exception as e:
            st.error(f"错误: {e}")

    # 已有进行中的选题
    if st.session_state.get("started") and "topic_id" in st.session_state:
        st.divider()
        st.subheader(f"📌 选题: {st.session_state.get('user_input', '')[:50]}...")

        topic_id = st.session_state["topic_id"]

        # 获取当前状态
        try:
            resp = requests.get(f"{API_BASE_URL}/api/topics/{topic_id}", timeout=5)
            if resp.status_code == 200:
                topic = resp.json()
                st.info(f"当前状态: **{topic['status'].upper()}** | 步骤: **{topic['current_step'].upper()}**")
        except:
            pass

        # 监听 SSE
        listen_sse(topic_id)


if __name__ == "__main__":
    main()