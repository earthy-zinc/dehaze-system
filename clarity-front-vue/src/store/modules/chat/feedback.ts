// 反馈 slice：消息点赞/点踩反馈的提交与会话级批量回填（反馈是独立资源，由宿主按消息 id 注入组件）
import {
  AiConversationAPI,
  type Feedback,
  type FeedbackForm,
} from "clarity-sdk-js";
import type { ChatCtx } from "./shared";

export function createFeedbackSlice(ctx: ChatCtx) {
  const { feedbackByMessage } = ctx;

  function writeFeedback(messageId: number, feedback: Feedback | null) {
    feedbackByMessage.value[messageId] = feedback;
  }

  async function submitFeedback(messageId: number, data: FeedbackForm | null) {
    if (data) {
      writeFeedback(
        messageId,
        await AiConversationAPI.submitFeedback(messageId, data)
      );
    } else {
      await AiConversationAPI.deleteFeedback(messageId);
      writeFeedback(messageId, null);
    }
  }

  /**
   * 会话级批量回填：批量结果是本会话反馈的权威快照。
   *
   * messageIds 为本会话的消息范围：范围内的条目以本批结果为准，批量未返回说明已撤销，
   * 删除既有条目还原状态；范围外的条目（其它会话/其它消息）原样保留。
   */
  function applyConversationFeedback(
    list: readonly Feedback[],
    messageIds: readonly number[]
  ) {
    const next = { ...feedbackByMessage.value };
    for (const messageId of messageIds) delete next[messageId];
    for (const item of list) {
      next[item.messageId] = item;
    }
    feedbackByMessage.value = next;
  }

  return {
    submitFeedback,
    applyConversationFeedback,
  };
}
