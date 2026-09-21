// 反馈 slice：消息点赞/点踩反馈的提交与查询
import { AiConversationAPI, type FeedbackForm } from "dehaze-sdk-js";
import type { ChatCtx } from "./shared";

export function createFeedbackSlice(ctx: ChatCtx) {
  const { feedbackByMessage } = ctx;

  async function submitFeedback(messageId: number, data: FeedbackForm | null) {
    if (data) {
      const feedback = await AiConversationAPI.submitFeedback(messageId, data);
      feedbackByMessage.value[messageId] = feedback;
    } else {
      await AiConversationAPI.deleteFeedback(messageId);
      feedbackByMessage.value[messageId] = null;
    }
  }

  async function fetchFeedback(messageId: number) {
    if (messageId <= 0) return;
    try {
      feedbackByMessage.value[messageId] =
        (await AiConversationAPI.getFeedback(messageId)) ?? null;
    } catch {
      // 反馈查询失败按无反馈展示
    }
  }

  return {
    submitFeedback,
    fetchFeedback,
  };
}
