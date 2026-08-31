// 用户端对话页 Store：可用模型缓存、配额信息、按会话维度的输入草稿（localStorage 持久化）
// 会话/消息/流式等共享状态在公共 chatStore 中，本 Store 仅承载页面级数据
import {
  AiBillingAPI,
  AiModelAPI,
  type AiModelVO,
  type BalanceVO,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { ref } from "vue";

const DRAFT_PREFIX = "chat:draft:";

export const useChatUserStore = defineStore("chatUser", () => {
  // ===== 可用模型（应用级缓存，登录会话内仅拉取一次） =====
  const availableModels = ref<AiModelVO[]>([]);
  const modelsLoaded = ref(false);

  async function ensureModels() {
    if (modelsLoaded.value) return;
    availableModels.value = await AiModelAPI.listEnabledModels("chat");
    modelsLoaded.value = true;
  }

  // ===== 配额（发送消息前后刷新） =====
  const quota = ref<BalanceVO | null>(null);

  async function fetchQuota() {
    quota.value = await AiBillingAPI.getBalance();
  }

  // ===== 输入草稿（localStorage 按会话持久化，刷新/误关闭不丢失） =====
  function getDraft(conversationId: number | null) {
    if (!conversationId) return "";
    return localStorage.getItem(DRAFT_PREFIX + conversationId) ?? "";
  }

  function saveDraft(conversationId: number | null, content: string) {
    const key = DRAFT_PREFIX + conversationId;
    if (!conversationId || !content) {
      localStorage.removeItem(key);
      return;
    }
    localStorage.setItem(key, content);
  }

  return {
    availableModels,
    modelsLoaded,
    quota,
    ensureModels,
    fetchQuota,
    getDraft,
    saveDraft,
  };
});
