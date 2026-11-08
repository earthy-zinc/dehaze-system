<!-- 消息操作栏（悬停出现）：默认只露「复制」，其余操作收进「···」菜单
     （7 个按钮平铺会让操作栏比正文更抢眼）；管理端审计不提供内容操作（纯展示）。
     内容动作统一上抛 message-action（判别联合）；「反馈」仅打开编辑面板，属组件内交互。 -->
<script lang="ts" setup>
import { computed } from "vue";
import type { Message, MessageStatus } from "clarity-sdk-js";
import type { ChatMessageActionVM, ChatScopeVM } from "../types";

defineOptions({ name: "MessageActionBar" });

const props = withDefaults(
  defineProps<{
    message: Message;
    scope: ChatScopeVM;
    /** 是否为链上最后一条助手回复：中间回复有后续对话，后端拒绝删除 */
    canDelete?: boolean;
  }>(),
  { canDelete: true }
);

const emit = defineEmits<{
  action: [action: ChatMessageActionVM];
  /** 打开反馈编辑面板（提交/撤销仍经 message-action 上抛） */
  feedback: [];
}>();

/**
 * 本栏可直接上抛的动作（命令字符串的唯一来源）。
 * 以 ChatMessageActionVM["type"] 约束子集：动作重命名/删除时这里同步报错，不留裸字符串。
 */
const BAR_ACTIONS = [
  "copy",
  "quote",
  "regenerate",
  "speak",
  "delete",
] as const satisfies readonly ChatMessageActionVM["type"][];

type BarAction = (typeof BAR_ACTIONS)[number];

/** 「···」菜单项：feedback 只打开面板，其余转换为消息动作上抛 */
const MENU_ITEMS = [
  { command: "quote", label: "引用", divided: false },
  { command: "regenerate", label: "重新生成", divided: false },
  { command: "speak", label: "朗读", divided: false },
  { command: "feedback", label: "反馈", divided: false },
  { command: "delete", label: "删除", divided: true },
] as const satisfies readonly {
  command: BarAction | "feedback";
  label: string;
  divided: boolean;
}[];

/** 删除只在叶子回复上提供；中间回复的后续对话不可删，入口不展示 */
const menuItems = computed(() =>
  MENU_ITEMS.filter(
    (item) => item.command !== "delete" || props.canDelete !== false
  )
);

function emitAction(type: BarAction) {
  emit("action", { type, messageId: props.message.id });
}

const isAssistant = computed(() => props.message.role === "assistant");

const assistantStatus = computed<MessageStatus | undefined>(() =>
  props.message.role === "assistant" ? props.message.status : undefined
);

// 仅用户端且助手消息已到终态才提供内容操作入口。
// awaiting_input 也排除：中断挂起时后端仍持有该轮流锁，重新生成/删除会被拒绝（400）。
const interactive = computed(
  () =>
    props.scope === "self" &&
    isAssistant.value &&
    assistantStatus.value !== "running" &&
    assistantStatus.value !== "awaiting_input"
);

/** el-dropdown 的 command 是运行时字符串：先收窄到动作子集，再上抛 */
function isBarAction(command: string): command is BarAction {
  return BAR_ACTIONS.some((action) => action === command);
}

/** 「···」菜单项 → 动作类型 */
function handleCommand(command: string): void {
  if (command === "feedback") {
    emit("feedback");
    return;
  }
  if (isBarAction(command)) emitAction(command);
}
</script>

<template>
  <div v-if="scope === 'self'" class="message-action-bar">
    <el-button link size="small" @click="emitAction('copy')">复制</el-button>
    <el-dropdown v-if="interactive" trigger="click" @command="handleCommand">
      <el-button link size="small" class="more-actions">···</el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item
            v-for="item in menuItems"
            :key="item.command"
            :command="item.command"
            :divided="item.divided"
          >
            {{ item.label }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<style scoped lang="scss">
.message-action-bar {
  display: flex;
  gap: 4px;
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;

  .message-block:hover & {
    opacity: 1;
  }

  .more-actions {
    letter-spacing: 1px;
  }
}
</style>
