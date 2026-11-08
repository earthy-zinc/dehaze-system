// AI 对话无状态组件层入口：导出纯 UI 派生类型 / 块派生函数 / 全部无状态组件。
// 组件直接消费契约 Message（import type 自 clarity-sdk-js），不再有适配层或 VM 镜像层。
export * from "./types";
export * from "./model/blocks";

// 消息数据视图层（全部无状态：props 进、事件出）
export { default as MessageList } from "./components/MessageList.vue";
export { default as UserMessage } from "./components/UserMessage.vue";
export { default as AssistantMessage } from "./components/AssistantMessage.vue";
export { default as MessageActionBar } from "./components/MessageActionBar.vue";
export { default as ThinkingPanel } from "./components/ThinkingPanel.vue";
export { default as ThoughtChain } from "./components/ThoughtChain.vue";
export { default as ThoughtStep } from "./components/ThoughtStep.vue";
export { default as InterruptCard } from "./components/InterruptCard.vue";
export { default as ArtifactCard } from "./components/ArtifactCard.vue";
export { default as FeedbackBar } from "./components/FeedbackBar.vue";
export { default as SuggestionList } from "./components/SuggestionList.vue";
export { default as SubAgentPanel } from "./components/SubAgentPanel.vue";

// 计划/分支组件
export { default as BranchSwitcher } from "./components/BranchSwitcher.vue";
export { default as PlanPanel } from "./components/PlanPanel.vue";
