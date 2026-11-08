// 流式副作用层：归约产出的 effect 解释执行、失败收口、自动朗读与 async 轮询。
//
// 只通过注入的钩子写全局流状态/释放会话；消息实体读写直连 ctx（与归约器同一张表）。
import { AiConversationAPI, type Message } from "clarity-sdk-js";
import { textOfBlocks } from "@/components/ai-chat/model/blocks";
import { useVoiceStore } from "@/store/modules/voice";
import { findMessage, putMessage } from "./entities";
import {
  ASYNC_POLL_INTERVAL_MS,
  ASYNC_POLL_MAX_TIMES,
  sleep,
  toSpeechText,
  type ChatCtx,
  type ChatScope,
  type StreamSession,
  type StreamStatus,
} from "./shared";
import type { StreamEffect } from "./stream-reducer";

/** 副作用层向 slice 回调的状态钩子 */
export interface StreamEffectHooks {
  /** 写全局流式态（status / messageId / sessionId） */
  setStream(status: StreamStatus, messageId: number | null): void;
  /** 流终态收尾：清定时器、释放会话、补已读 */
  finishSession(session: StreamSession): void;
}

export function createStreamEffects(ctx: ChatCtx, hooks: StreamEffectHooks) {
  let asyncPollToken = 0;

  /** 使在途的 async 轮询失效（停止/离开会话时调用） */
  function cancelAsyncPoll() {
    asyncPollToken++;
  }

  /** 失败收口：消息置 failed 并写可读原因（仅"本轮未结束"的消息，已中断/终态不覆盖） */
  function failMessage(session: StreamSession, reason: string) {
    const message = findMessage(ctx, session.messageId);
    if (message && message.status === "running") {
      message.status = "failed";
      message.error = reason;
    }
    hooks.setStream("idle", null);
    hooks.finishSession(session);
  }

  /** 语音回复开关开启时自动朗读助手回复（仅用户端；合成失败静默降级） */
  function autoSpeak(message: Message, originScope: ChatScope) {
    if (originScope !== "self") return;
    const voiceStore = useVoiceStore();
    if (!voiceStore.ttsPreference.enabled) return;
    const text = toSpeechText(textOfBlocks(message.blocks));
    if (!text) return;
    void voiceStore.playSpeech(text).catch(() => {
      // 降级为纯文本回复
    });
  }

  /** async_wait 中断：message.end 仅表示通道关闭，轮询到消息终态后整体替换实体 */
  async function pollAsyncMessage(conversationId: number, messageId: number) {
    const token = ++asyncPollToken;
    for (let i = 0; i < ASYNC_POLL_MAX_TIMES; i++) {
      await sleep(ASYNC_POLL_INTERVAL_MS);
      if (
        token !== asyncPollToken ||
        ctx.currentConversationId.value !== conversationId
      ) {
        return;
      }
      try {
        const detail = await AiConversationAPI.getMessageDetail(messageId);
        if (detail.status !== "running") {
          putMessage(ctx, detail);
          if (
            ctx.streamStatus.value === "interrupted" &&
            ctx.streamMessageId.value === messageId
          ) {
            hooks.setStream("idle", null);
          }
          return;
        }
      } catch {
        // 单次轮询失败忽略，下一轮重试
      }
    }
  }

  /** 把归约产出的副作用解释执行为全局态切换 / 收尾 / 轮询 */
  function applyEffect(session: StreamSession, effect: StreamEffect) {
    switch (effect.type) {
      case "status":
        hooks.setStream(effect.status, session.messageId);
        return;
      case "end": {
        const message = findMessage(ctx, session.messageId);
        if (effect.status === "awaiting_input") {
          hooks.setStream("interrupted", session.messageId);
        } else {
          hooks.setStream("idle", null);
        }
        if (effect.status === "completed" && message)
          autoSpeak(message, session.originScope);
        if (
          effect.status === "awaiting_input" &&
          message?.pendingInterrupt?.type === "async_wait"
        ) {
          void pollAsyncMessage(session.conversationId, session.messageId);
        }
        hooks.finishSession(session);
        return;
      }
      case "failed":
        failMessage(session, effect.reason);
        return;
    }
  }

  /** 中断后连接关闭：本轮通道已关，消息进入等待确认态（契约内正常收束） */
  function settleInterruptedClose(session: StreamSession) {
    const message = findMessage(ctx, session.messageId);
    if (message?.pendingInterrupt && message.status === "running") {
      message.status = "awaiting_input";
    }
    hooks.setStream("interrupted", session.messageId);
    hooks.finishSession(session);
  }

  return {
    applyEffect,
    failMessage,
    settleInterruptedClose,
    cancelAsyncPoll,
  };
}
