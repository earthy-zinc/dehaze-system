import { pageQuery } from "./common";
import { uniqueName } from "./common";
import type {
  ConversationCreateForm,
  ConversationQuery,
  ConversationUpdateForm,
  EditMessageForm,
  FeedbackForm,
  MemoryCreateForm,
  MemoryQuery,
} from "../../src/api/ai-conversation/model";

/** 会话创建表单工厂 */
export const createConversationForm = (
  overrides?: Partial<ConversationCreateForm>
): ConversationCreateForm => ({
  ...overrides,
});

/** 会话列表查询参数工厂 */
export const createConversationQuery = (overrides?: Partial<ConversationQuery>) =>
  pageQuery<ConversationQuery>({ ...overrides });

/** 会话更新表单工厂 */
export const createConversationUpdateForm = (
  overrides?: Partial<ConversationUpdateForm>
): ConversationUpdateForm => ({
  ...overrides,
});

/** 编辑消息表单工厂 */
export const createEditMessageForm = (overrides?: Partial<EditMessageForm>): EditMessageForm => ({
  content: "编辑后的测试消息",
  ...overrides,
});

/** 反馈表单工厂 */
export const createFeedbackForm = (overrides?: Partial<FeedbackForm>): FeedbackForm => ({
  rating: 1,
  tags: ["accurate"],
  ...overrides,
});

/** 记忆创建表单工厂（source=manual 便于按来源筛选，前缀 test_ 便于清理） */
export const createMemoryForm = (overrides?: Partial<MemoryCreateForm>): MemoryCreateForm => ({
  memoryType: "semantic",
  content: uniqueName("test_memory"),
  importance: 50,
  source: "manual",
  ...overrides,
});

/** 记忆分页/筛选查询参数工厂 */
export const createMemoryQuery = (overrides?: Partial<MemoryQuery>) =>
  pageQuery<MemoryQuery>({ ...overrides });
