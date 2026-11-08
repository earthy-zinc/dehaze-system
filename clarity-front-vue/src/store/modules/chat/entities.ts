// 消息实体表操作：Record<id, Message> 的增删改、顺序维护与本地占位构造。
//
// 一条消息一个实体，blocks 是唯一内容源；本文件只做"写入实体"这一件事，
// 不做任何展示派生（展示派生见 components/ai-chat/model 与组件 computed）。
import type { Block, Message, Role } from "clarity-sdk-js";
import { EMPTY_USAGE, nextLocalMessageId, type ChatCtx } from "./shared";

/** 按 id 查找消息实体 */
export function findMessage(
  ctx: ChatCtx,
  messageId: number
): Message | undefined {
  return ctx.messages.value[messageId];
}

/** 写入实体（已存在则原位替换；不存在时按 at 插入顺序，缺省追加到末尾） */
export function putMessage(ctx: ChatCtx, message: Message, at?: number): void {
  ctx.messages.value[message.id] = message;
  if (ctx.messageOrder.value.includes(message.id)) return;
  const order = ctx.messageOrder.value.slice();
  const index =
    at === undefined ? order.length : Math.max(0, Math.min(at, order.length));
  order.splice(index, 0, message.id);
  ctx.messageOrder.value = order;
}

/** 删除实体（含顺序） */
export function removeMessage(ctx: ChatCtx, messageId: number): void {
  if (!(messageId in ctx.messages.value)) return;
  delete ctx.messages.value[messageId];
  ctx.messageOrder.value = ctx.messageOrder.value.filter(
    (id) => id !== messageId
  );
}

/** 构造本地占位消息（负数 id，status=running；服务端实体到达后整体替换） */
export function createLocalMessage(params: {
  conversationId: number;
  role: Role;
  status: Message["status"];
  parentMessageId?: number | null;
  /** 用户消息正文（落成单个 text 块）；助手占位留空 */
  text?: string;
}): Message {
  const id = nextLocalMessageId();
  const blocks: Block[] = params.text
    ? [
        {
          id: `local-${id}-text-1`,
          kind: "text",
          seq: 1,
          text: params.text,
          durationMs: null,
          tool: null,
        },
      ]
    : [];
  return {
    id,
    conversationId: params.conversationId,
    parentMessageId: params.parentMessageId ?? null,
    role: params.role,
    status: params.status,
    model: null,
    error: null,
    blocks,
    plan: null,
    pendingInterrupt: null,
    usage: { ...EMPTY_USAGE },
    artifactIds: [],
    memoryIds: [],
    suggestions: [],
    edited: false,
    createTime: new Date().toISOString(),
    updateTime: null,
  };
}
