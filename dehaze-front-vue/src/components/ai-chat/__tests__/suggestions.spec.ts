// 推荐追问展示位置单测：仅最后一条 assistant + 边界（空列表、无 assistant、user 命中）。
import { describe, expect, it } from "vitest";
import type {
  ChatAssistantMessageVM,
  ChatMessageVM,
  ChatUserMessageVM,
} from "../types";
import { shouldShowSuggestions } from "../vm/suggestions";

const user = (id: number): ChatUserMessageVM => ({
  role: "user",
  id,
  content: "",
});

const assistant = (id: number): ChatAssistantMessageVM => ({
  role: "assistant",
  id,
  status: "completed",
  text: "",
  thinking: null,
  steps: [],
  toolCalls: [],
  artifacts: [],
  memories: [],
  feedback: null,
  suggestions: [],
});

describe("shouldShowSuggestions", () => {
  it("最后一条 assistant 命中", () => {
    const messages: ChatMessageVM[] = [user(1), assistant(2)];
    expect(shouldShowSuggestions(messages, 2)).toBe(true);
  });

  it("非最后一条 assistant 不命中", () => {
    const messages: ChatMessageVM[] = [user(1), assistant(2), assistant(4)];
    expect(shouldShowSuggestions(messages, 2)).toBe(false);
    expect(shouldShowSuggestions(messages, 4)).toBe(true);
  });

  it("尾部 user 消息不影响（仍取最后一条 assistant）", () => {
    const messages: ChatMessageVM[] = [user(1), assistant(2), user(3)];
    expect(shouldShowSuggestions(messages, 2)).toBe(true);
    expect(shouldShowSuggestions(messages, 3)).toBe(false);
  });

  it("空列表返回 false", () => {
    expect(shouldShowSuggestions([], 1)).toBe(false);
  });

  it("无 assistant 消息返回 false", () => {
    expect(shouldShowSuggestions([user(1), user(2)], 1)).toBe(false);
  });

  it("messageId 不存在返回 false", () => {
    expect(shouldShowSuggestions([user(1), assistant(2)], 99)).toBe(false);
  });
});
