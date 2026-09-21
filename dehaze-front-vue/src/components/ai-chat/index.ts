// AI 对话无状态组件层入口：导出 VM 类型 / 归约函数 / wire 适配 / 全部无状态组件。
export * from "./types";
export * from "./vm";
export * from "./adapters/fromSdk";

// 消息数据视图层与过程透明组件族（全部无状态：props 进、事件出）
export { default as MessageViews } from "./components/MessageViews.vue";
export { default as MessageList } from "./components/MessageList.vue";
export { default as UserMessage } from "./components/UserMessage.vue";
export { default as AssistantMessage } from "./components/AssistantMessage.vue";
export { default as ToolMessage } from "./components/ToolMessage.vue";
export { default as MessageActionBar } from "./components/MessageActionBar.vue";
export { default as ThinkingPanel } from "./components/ThinkingPanel.vue";
export { default as ThoughtChain } from "./components/ThoughtChain.vue";
export { default as ThoughtStep } from "./components/ThoughtStep.vue";
export { default as InterruptCard } from "./components/InterruptCard.vue";
export { default as ArtifactCard } from "./components/ArtifactCard.vue";
export { default as FeedbackBar } from "./components/FeedbackBar.vue";
export { default as SuggestionList } from "./components/SuggestionList.vue";
export { default as MemoryReferenceList } from "./components/MemoryReferenceList.vue";
export { default as SubAgentPanel } from "./components/SubAgentPanel.vue";

// 过程透明 + 计划/分支组件
export { default as ProcessPanel } from "./components/ProcessPanel.vue";
export { default as ContextChip } from "./components/ContextChip.vue";
export { default as StepTimeline } from "./components/StepTimeline.vue";
export { default as TraceSummary } from "./components/TraceSummary.vue";
export { default as BranchSwitcher } from "./components/BranchSwitcher.vue";
export { default as PlanPanel } from "./components/PlanPanel.vue";
