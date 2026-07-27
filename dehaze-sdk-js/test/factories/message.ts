import { AnnouncementForm, MessageSendRequest } from "@/api/message/model";
import { uniqueName } from "./common";

export function createAnnouncementForm(overrides: Partial<AnnouncementForm> = {}): AnnouncementForm {
  return {
    title: uniqueName("test_公告"),
    content: "测试公告内容",
    type: "operation",
    importance: 1,
    targetScope: "all",
    ...overrides,
  };
}

export function createMessageSendRequest(overrides: Partial<MessageSendRequest> = {}): MessageSendRequest {
  return {
    type: "business",
    title: uniqueName("test_消息"),
    content: "测试消息正文",
    recipientIds: [1],
    bizModule: "test",
    bizId: uniqueName("biz"),
    priority: 2,
    ...overrides,
  };
}
