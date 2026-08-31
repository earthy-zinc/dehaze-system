<!-- 消息列表：按时间正序渲染，role 分流组件，status 驱动流式/失败/取消态，自动滚动跟随 -->
<script lang="ts" setup>
import type { AiMessageVO } from "dehaze-sdk-js";
import { nextTick, onMounted, watch } from "vue";
import { storeToRefs } from "pinia";
import { useChatStore } from "@/store/modules/chat";
import AssistantMessage from "./AssistantMessage.vue";
import InterruptCard from "./InterruptCard.vue";
import ToolMessage from "./ToolMessage.vue";
import UserMessage from "./UserMessage.vue";

defineOptions({ name: "MessageList" });

const props = defineProps<{
  messages: AiMessageVO[];
  streamingMessageId: number | null;
  scrollFollowEnabled: boolean;
}>();

const emit = defineEmits<{
  "scroll-follow-toggle": [enabled: boolean];
  "reach-bottom": [];
  edit: [message: AiMessageVO];
  copy: [];
  quote: [message: AiMessageVO];
  regenerate: [message: AiMessageVO];
  feedback: [
    message: AiMessageVO,
    data: import("dehaze-sdk-js").FeedbackForm | null,
  ];
  delete: [message: AiMessageVO];
  speak: [message: AiMessageVO];
  trace: [message: AiMessageVO];
}>();

const chatStore = useChatStore();
const { interruptedMessageId } = storeToRefs(chatStore);

const BOTTOM_THRESHOLD_PX = 40;
const bodyRef = ref<HTMLElement>();

function scrollToBottom() {
  nextTick(() => {
    if (bodyRef.value) bodyRef.value.scrollTop = bodyRef.value.scrollHeight;
  });
}

function handleScroll() {
  const el = bodyRef.value;
  if (!el) return;
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
  const atBottom = distance < BOTTOM_THRESHOLD_PX;
  if (atBottom) {
    emit("reach-bottom");
    if (!props.scrollFollowEnabled) emit("scroll-follow-toggle", true);
  } else if (props.scrollFollowEnabled) {
    emit("scroll-follow-toggle", false);
  }
}

function backToBottom() {
  scrollToBottom();
  emit("scroll-follow-toggle", true);
}

// 流式增量经 rAF 批量写入，监听最后一条消息内容长度即可低成本跟随
watch(
  () => {
    const last = props.messages[props.messages.length - 1];
    return `${props.messages.length}:${last?.content?.length ?? 0}`;
  },
  () => {
    if (props.scrollFollowEnabled) scrollToBottom();
  }
);

watch(
  () => props.scrollFollowEnabled,
  (enabled) => {
    if (enabled) scrollToBottom();
  }
);

onMounted(() => {
  scrollToBottom();
});
</script>

<template>
  <div class="chat-message-list">
    <div ref="bodyRef" class="chat-message-list__body" @scroll="handleScroll">
      <template v-for="message in messages" :key="message.id">
        <UserMessage
          v-if="message.role === 'user'"
          :message="message"
          @edit="(m) => emit('edit', m)"
          @copy="emit('copy')"
          @quote="(m) => emit('quote', m)"
        />
        <AssistantMessage
          v-else-if="message.role === 'assistant'"
          :message="message"
          :scope="chatStore.scope"
          @regenerate="(m) => emit('regenerate', m)"
          @quote="(m) => emit('quote', m)"
          @feedback="(m, data) => emit('feedback', m, data)"
          @delete="(m) => emit('delete', m)"
          @speak="(m) => emit('speak', m)"
          @copy="emit('copy')"
          @trace="(m) => emit('trace', m)"
        />
        <ToolMessage
          v-else
          :message="message"
          :tool-calls="
            chatStore.toolCallsByMessage[message.id] ?? message.toolCalls
          "
        />
      </template>

      <InterruptCard
        v-for="(interrupt, index) in chatStore.interrupts"
        :key="index"
        :interrupt="interrupt"
        @confirm="
          (data) => chatStore.resumeInterrupt(interruptedMessageId!, data)
        "
        @reject="
          chatStore.resumeInterrupt(interruptedMessageId!, { confirm: false })
        "
        @resume="
          (data) => chatStore.resumeInterrupt(interruptedMessageId!, data)
        "
      />

      <el-empty
        v-if="messages.length === 0"
        description="开始你的第一轮对话吧"
      />
    </div>

    <Transition name="fade">
      <el-button
        v-if="!scrollFollowEnabled"
        class="chat-message-list__back-bottom"
        size="small"
        circle
        @click="backToBottom"
      >
        <el-icon><Bottom /></el-icon>
      </el-button>
    </Transition>
  </div>
</template>

<style scoped lang="scss">
.chat-message-list {
  position: relative;
  height: 100%;

  &__body {
    height: 100%;
    padding: 16px;
    overflow-y: auto;
  }

  &__back-bottom {
    position: absolute;
    right: 24px;
    bottom: 24px;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
