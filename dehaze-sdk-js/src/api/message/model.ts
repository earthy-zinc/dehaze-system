import { PageQuery } from "@/types";

export interface MessageQuery extends PageQuery {
  type?: string;
  readStatus?: number;
}

export interface MessageSearchQuery extends PageQuery {
  keyword: string;
}

export interface MessageVO {
  id: number;
  type: string;
  typeLabel: string;
  title: string;
  summary?: string;
  content?: string;
  priority: number;
  readStatus: number;
  senderType: number;
  senderTypeLabel?: string;
  readTime?: string;
  jumpUrl?: string;
  extra?: Record<string, any>;
  createTime: string;
}

export interface UnreadCountVO {
  count: number;
}

export interface ReadAllResult {
  affectedCount: number;
}

export interface NotificationSettings {
  pushEnabled: boolean;
  dndEnabled: boolean;
  dndStart: string;
  dndEnd: string;
  preferences: {
    typeChannels: Record<string, { push: boolean }>;
    moduleSwitches: Record<string, boolean>;
  };
}

export interface NotificationSettingsForm {
  pushEnabled?: boolean;
  dndEnabled?: boolean;
  dndStart?: string;
  dndEnd?: string;
  preferences?: {
    typeChannels?: Record<string, { push: boolean }>;
    moduleSwitches?: Record<string, boolean>;
  };
}

export interface AnnouncementQuery extends PageQuery {
  title?: string;
  type?: string;
  status?: number;
}

export interface AnnouncementVO {
  id: number;
  title: string;
  content?: string;
  type: string;
  typeLabel?: string;
  importance: number;
  importanceLabel?: string;
  targetScope: string;
  targetScopeLabel?: string;
  targetParams?: Record<string, any>;
  status: number;
  statusLabel?: string;
  sendTime?: string;
  expireTime?: string;
  sentCount?: number;
  createTime: string;
  updateTime?: string;
  createBy?: number;
}

export interface AnnouncementForm {
  title: string;
  content: string;
  type: string;
  importance: number;
  targetScope: string;
  targetParams?: Record<string, any>;
  sendTime?: string;
  expireTime?: string;
}

export interface AnnouncementSendResult {
  sentCount: number;
}

export interface MessageTemplateQuery extends PageQuery {
  name?: string;
  type?: string;
  status?: number;
}

export interface MessageTemplateVO {
  id: number;
  code: string;
  name: string;
  type: string;
  titleTemplate: string;
  contentTemplate?: string;
  priority: number;
  channels?: Record<string, boolean>;
  variables?: { name: string; desc: string }[];
  status: number;
  createTime: string;
  updateTime?: string;
}

export interface MessageTemplateForm {
  name?: string;
  titleTemplate?: string;
  contentTemplate?: string;
  priority?: number;
  channels?: Record<string, boolean>;
  status?: number;
}

export interface MessageSendRequest {
  templateCode?: string;
  type: string;
  title?: string;
  content?: string;
  recipientIds: number[];
  bizModule?: string;
  bizId?: string;
  priority?: number;
  jumpUrl?: string;
  variables?: Record<string, string>;
  extra?: Record<string, any>;
}

export interface MessageSendResult {
  messageIds: number[];
}
