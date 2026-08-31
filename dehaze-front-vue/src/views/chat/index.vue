<!-- 用户端对话页：左会话侧栏 + 右对话区（头部/配额/消息流/输入区），消息与流式状态全部经公共 chatStore -->
<script lang="ts" setup>
import { ElMessage, ElMessageBox } from "element-plus";
import type { AiMessageVO, FeedbackForm } from "dehaze-sdk-js";
import { onMounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "ChatPage" });

const route = useRoute();
const router = useRouter();
const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

const settingsVisible = ref(false);
const memoryVisible = ref(false);
const taskDialogVisible = ref(false);

const LAST_CONVERSATION_KEY = "dehaze.chat.lastConversationId";

/** 读取本会话维度（self/admin）上次访问的会话 ID（localStorage 记忆） */
function readLastConversationId(): number {
  const raw = localStorage.getItem(`${LAST_CONVERSATION_KEY}.${chatStore.scope}`);
  const id = Number(raw);
  return Number.isInteger(id) && id > 0 ? id : 0;
}

/** 记录本会话维度最近访问的会话 ID（退出后再次进入恢复到离开位置） */
function rememberConversation(id: number) {
  if (!id) return;
  localStorage.setItem(`${LAST_CONVERSATION_KEY}.${chatStore.scope}`, String(id));
}

/** 从会话列表中取"最近活跃"会话（有最后发言的最新的；置顶的旧会话不作为默认入口） */
function latestActiveConversation() {
  const withMessage = chatStore.conversations
    .filter((c) => c.lastMessageAt)
    .sort((a, b) => Date.parse(b.lastMessageAt ?? "") - Date.parse(a.lastMessageAt ?? ""));
  return withMessage[0] ?? chatStore.conversations[0];
}

// 路由 → 会话：进入 /chat/:conversationId 时加载对应消息并记录上次访问
watch(
  () => route.params.conversationId,
  (value) => {
    const id = Number(value);
    if (!id || id === chatStore.currentConversationId) return;
    chatStore
      .fetchMessages(id)
      .then(() => rememberConversation(id))
      .catch(() => {});
  }
);

onMounted(async () => {
  chatStore.initScope("self");
  chatUserStore.ensureModels().catch(() => {});
  chatUserStore.fetchQuota().catch(() => {});
  await chatStore.fetchConversations();
  // 等待加载期间用户已发起对话（流式进行中）：跳过默认跳转/消息加载，
  // 避免 fetchMessages 覆盖占位消息或路由切换打断当前流（否则界面永久"正在思考"）
  if (chatStore.isStreaming) return;
  const routeId = Number(route.params.conversationId);
  if (routeId) {
    if (routeId !== chatStore.currentConversationId) {
      await chatStore.fetchMessages(routeId);
    }
    rememberConversation(routeId);
    return;
  }
  // 优先恢复到上次离开的会话（仍存在于列表则直达）
  const savedId = readLastConversationId();
  if (savedId && chatStore.conversations.some((c) => c.id === savedId)) {
    router.replace(`/chat/${savedId}`);
    return;
  }
  // 无记忆/已失效：默认进入最近活跃会话（有消息的最新发言者）
  const latest = latestActiveConversation();
  if (latest) router.replace(`/chat/${latest.id}`);
});

function handleSelect(conversationId: number) {
  router.push(`/chat/${conversationId}`);
}

async function handleSend(text: string) {
  if (!chatStore.currentConversationId) {
    // 空态直接发送：先建会话，加载完成后路由与消息状态就绪
    const conversation = await chatStore.createConversation();
    await chatStore.fetchMessages(conversation.id);
    rememberConversation(conversation.id);
    router.replace(`/chat/${conversation.id}`);
  }
  chatStore.sendMessage(text);
  chatUserStore.saveDraft(chatStore.currentConversationId, "");
  chatUserStore.fetchQuota().catch(() => {});
}

function handleStop() {
  chatStore.stopStreaming();
}

function handleEditMessage(message: AiMessageVO) {
  ElMessageBox.prompt("编辑后将以新分支重新触发回复", "编辑消息", {
    inputValue: message.content ?? "",
    inputType: "textarea",
  })
    .then(({ value }) => {
      if (value?.trim()) chatStore.editMessage(message.id, value);
    })
    .catch(() => {});
}

function handleDeleteMessage(message: AiMessageVO) {
  ElMessageBox.confirm("确认删除该条消息？", "删除确认", { type: "warning" })
    .then(() => chatStore.deleteMessage(message.id))
    .catch(() => {});
}
</script>

<template>
  <div class="chat-page">
    <aside class="chat-page__sidebar">
      <ChatSidebar
        @select="handleSelect"
        @open-settings="settingsVisible = true"
        @open-memory="memoryVisible = true"
      />
    </aside>

    <main class="chat-page__main">
      <ChatHeader
        @open-settings="settingsVisible = true"
        @open-memory="memoryVisible = true"
        @open-task="taskDialogVisible = true"
      />

      <QuotaIndicator />

      <div class="chat-page__body">
        <MessageList
          v-if="chatStore.messages.length > 0 || chatStore.messagesLoading"
          :messages="chatStore.messages"
          :streaming-message-id="chatStore.streamingMessageId"
          :scroll-follow-enabled="chatStore.scrollFollowEnabled"
          @scroll-follow-toggle="
            (enabled: boolean) => (chatStore.scrollFollowEnabled = enabled)
          "
          @edit="handleEditMessage"
          @quote="(message: AiMessageVO) => chatStore.quoteMessage(message)"
          @regenerate="
            (message: AiMessageVO) => chatStore.regenerate(message.id)
          "
          @feedback="
            (message: AiMessageVO, data: FeedbackForm | null) =>
              chatStore.submitFeedback(message.id, data)
          "
          @delete="handleDeleteMessage"
          @speak="(message: AiMessageVO) => chatStore.speakMessage(message)"
        />
        <EmptyState v-else @select="handleSend" />

        <MessageInput @send="handleSend" @stop="handleStop" />
      </div>
    </main>

    <ConversationSettings v-model="settingsVisible" />
    <MemoryPanel v-model="memoryVisible" />
    <SaveAsTaskDialog v-model="taskDialogVisible" />
  </div>
</template>

<style scoped lang="scss">
.chat-page {
  display: flex;
  height: calc(100vh - var(--navbar-height, 50px));
  overflow: hidden;

  &__sidebar {
    flex-shrink: 0;
    width: 280px;
    background-color: var(--el-bg-color);
    border-right: 1px solid var(--el-border-color-light);
  }

  &__main {
    display: flex;
    flex: 1;
    flex-direction: column;
    min-width: 0;
  }

  &__body {
    display: flex;
    flex: 1;
    flex-direction: column;
    min-height: 0;
    background-color: var(--el-bg-color-page);
  }
}
</style>
