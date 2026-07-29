import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  AnnouncementForm,
  AnnouncementQuery,
  AnnouncementSendResult,
  AnnouncementVO,
  MessageQuery,
  MessageSearchQuery,
  MessageSendRequest,
  MessageSendResult,
  MessageTemplateForm,
  MessageTemplateQuery,
  MessageTemplateVO,
  MessageVO,
  NotificationSettings,
  NotificationSettingsForm,
  ReadAllResult,
  UnreadCountVO,
} from "./model";

class MessageAPI {
  static getPage(queryParams?: MessageQuery) {
    return request<PageResult<MessageVO[]>>({
      url: "/api/v1/messages",
      method: "get",
      params: queryParams,
    });
  }

  static getUnreadCount() {
    return request<UnreadCountVO>({
      url: "/api/v1/messages/unread-count",
      method: "get",
    });
  }

  static getDetail(id: number) {
    return request<MessageVO>({
      url: "/api/v1/messages/" + id,
      method: "get",
    });
  }

  static markRead(id: number) {
    return request({
      url: `/api/v1/messages/${id}/_read`,
      method: "patch",
    });
  }

  static markAllRead(type?: string) {
    return request<ReadAllResult>({
      url: "/api/v1/messages/_read-all",
      method: "patch",
      params: type ? { type } : undefined,
    });
  }

  static deleteByIds(ids: string) {
    return request({
      url: "/api/v1/messages/" + ids,
      method: "delete",
    });
  }

  static search(queryParams: MessageSearchQuery) {
    return request<PageResult<MessageVO[]>>({
      url: "/api/v1/messages/search",
      method: "get",
      params: queryParams,
    });
  }

  static send(data: MessageSendRequest) {
    return request<MessageSendResult>({
      url: "/api/v1/messages/send",
      method: "post",
      data,
    });
  }
}

class AnnouncementAPI {
  static getPage(queryParams?: AnnouncementQuery) {
    return request<PageResult<AnnouncementVO[]>>({
      url: "/api/v1/announcements/page",
      method: "get",
      params: queryParams,
    });
  }

  static create(data: AnnouncementForm) {
    return request<{ id: number }>({
      url: "/api/v1/announcements",
      method: "post",
      data,
    });
  }

  static getDetail(id: number) {
    return request<AnnouncementVO>({
      url: "/api/v1/announcements/" + id,
      method: "get",
    });
  }

  static update(id: number, data: Partial<AnnouncementForm>) {
    return request({
      url: "/api/v1/announcements/" + id,
      method: "put",
      data,
    });
  }

  static deleteById(id: number) {
    return request({
      url: "/api/v1/announcements/" + id,
      method: "delete",
    });
  }

  static send(id: number) {
    return request<AnnouncementSendResult>({
      url: `/api/v1/announcements/${id}/_send`,
      method: "post",
    });
  }

  static cancel(id: number) {
    return request({
      url: `/api/v1/announcements/${id}/_cancel`,
      method: "patch",
    });
  }
}

class MessageTemplateAPI {
  static getPage(queryParams?: MessageTemplateQuery) {
    return request<PageResult<MessageTemplateVO[]>>({
      url: "/api/v1/message-templates/page",
      method: "get",
      params: queryParams,
    });
  }

  static getDetail(id: number) {
    return request<MessageTemplateVO>({
      url: "/api/v1/message-templates/" + id,
      method: "get",
    });
  }

  static update(id: number, data: MessageTemplateForm) {
    return request({
      url: "/api/v1/message-templates/" + id,
      method: "put",
      data,
    });
  }
}

class NotificationSettingAPI {
  static get() {
    return request<NotificationSettings>({
      url: "/api/v1/notification-settings",
      method: "get",
    });
  }

  static update(data: NotificationSettingsForm) {
    return request({
      url: "/api/v1/notification-settings",
      method: "patch",
      data,
    });
  }
}

export default MessageAPI;
export { AnnouncementAPI, MessageTemplateAPI, NotificationSettingAPI };
