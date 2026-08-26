"""AI 对话流式调试 CLI。

用法：
    python scripts/chat.py --model deepseek-v4-flash --content "用一句话介绍你自己"
    python scripts/chat.py --backend java --model gpt-4o --content "回复ok"
    python scripts/chat.py --conversation-id 123 --content "继续"   # 复用已有会话

输出：会话 ID、事件序列（去 ping）、完整回复文本、该会话计费记录。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import api, sse


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 对话流式调试")
    parser.add_argument("--backend", "-b", default="python", choices=["java", "go", "python"])
    parser.add_argument("--model", "-m", default=None, help="模型标识（默认系统默认模型）")
    parser.add_argument("--content", "-c", default="用一句话介绍你自己", help="消息内容")
    parser.add_argument("--conversation-id", "-id", default=None, type=int, help="复用已有会话（缺省新建）")
    args = parser.parse_args()

    # 1. 确定会话：复用已有或新建（标题 "chat调试"）
    if args.conversation_id:
        conversation_id = args.conversation_id
    else:
        created = api.post(
            "/api/v1/ai/conversations",
            backend=args.backend,
            json={"title": "chat调试", "model": args.model},
        )
        conversation_id = created["data"]["id"]
        print(f"created conversation: {conversation_id}")

    # 2. SSE 流式发送（Idempotency-Key / 会话注入由 utils/sse.py 处理）
    body = {"content": args.content}
    if args.model:
        body["model"] = args.model
    result = sse.stream_request(
        "POST",
        f"/api/v1/ai/conversations/{conversation_id}/messages",
        backend=args.backend,
        json_body=body,
    )

    # 3. 输出事件序列（已去 ping）与完整回复
    print(f"\nconversationId: {conversation_id}")
    print("events:")
    for name, payload in result.events:
        print(f"  {name:<22} {json.dumps(payload, ensure_ascii=False)}")
    if result.thought:
        print(f"\nthought:\n{result.thought}")
    print(f"\ntext:\n{result.text}")

    # 4. 该会话计费记录
    records = api.get(
        "/api/v1/ai-billing/records",
        backend=args.backend,
        params={"conversationId": conversation_id, "pageSize": 5},
    )
    print("\nbilling records:")
    for rec in records["data"]["list"]:
        # 计费记录走 OrmResult 序列化别名，字段为驼峰（inputTokens/outputTokens）
        print(
            f"  model={rec.get('model')} "
            f"inputTokens={rec.get('inputTokens')} "
            f"outputTokens={rec.get('outputTokens')} "
            f"credits={rec.get('credits')}"
        )


if __name__ == "__main__":
    main()
