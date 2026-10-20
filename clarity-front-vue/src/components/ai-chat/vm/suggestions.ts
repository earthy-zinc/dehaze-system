// 推荐追问展示位置判定：纯函数、无副作用，零外部依赖。
import type { ChatMessageVM } from "../types";

/** 推荐追问仅挂在最后一条 assistant 消息之后（无 assistant 时为 false） */
export function shouldShowSuggestions(
  messages: ChatMessageVM[],
  messageId: number
): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role === "assistant") return message.id === messageId;
  }
  return false;
}
