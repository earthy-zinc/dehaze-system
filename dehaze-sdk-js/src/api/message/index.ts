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
    return request<any, PageResult<MessageVO[]>>({
      url: "/api/v1/messages",
      method: "get",
      params: queryParams,
    });
  }

  static getUnreadCount() {
    return request<any, UnreadCountVO>({
      url: "/api/v1/messages/unread-count",
      method: "get",
    });
  }

  static getDetail(id: number) {
    return request<any, MessageVO>({
      url: "/api/v1/messages/" + id,
      method: "get",
    });
  }

  static markRead(id: number) {
    return request({
      url: `/api/v1/messages/${id}/read`,
      method: "put",
    });
  }

  static markAllRead(type?: string) {
    return request<any, ReadAllResult>({
      url: "/api/v1/messages/read-all",
      method: "put",
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
    return request<any, PageResult<MessageVO[]>>({
      url: "/api/v1/messages/search",
      method: "get",
      params: queryParams,
    });
  }

  static send(data: MessageSendRequest) {
    return request<any, MessageSendResult>({
      url: "/api/v1/messages/send",
      method: "post",
      data,
    });
  }
}

class AnnouncementAPI {
  static getPage(queryParams?: AnnouncementQuery) {
    return request<any, PageResult<AnnouncementVO[]>>({
      url: "/api/v1/announcements/page",
      method: "get",
      params: queryParams,
    });
  }

  static create(data: AnnouncementForm) {
    return request<any, { id: number }>({
      url: "/api/v1/announcements",
      method: "post",
      data,
    });
  }

  static getDetail(id: number) {
    return request<any, AnnouncementVO>({
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
    return request<any, AnnouncementSendResult>({
      url: `/api/v1/announcements/${id}/send`,
      method: "post",
    });
  }

  static cancel(id: number) {
    return request({
      url: `/api/v1/announcements/${id}/cancel`,
      method: "put",
    });
  }
}

class MessageTemplateAPI {
  static getPage(queryParams?: MessageTemplateQuery) {
    return request<any, PageResult<MessageTemplateVO[]>>({
      url: "/api/v1/message-templates/page",
      method: "get",
      params: queryParams,
    });
  }

  static getDetail(id: number) {
    return request<any, MessageTemplateVO>({
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
    return request<any, NotificationSettings>({
      url: "/api/v1/notification-settings",
      method: "get",
    });
  }

  static update(data: NotificationSettingsForm) {
    return request({
      url: "/api/v1/notification-settings",
      method: "put",
      data,
    });
  }
}

export default MessageAPI;
export { AnnouncementAPI, MessageTemplateAPI, NotificationSettingAPI };
