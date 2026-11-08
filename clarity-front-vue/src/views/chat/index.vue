<!-- 用户端对话页：左会话侧栏 + 右对话区（头部/配额/消息流/输入区），消息与流式状态全部经公共 chatStore -->
<script lang="ts" setup>
import { ElMessage, ElMessageBox } from "element-plus";
import { onMounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import type { Artifact, Message } from "clarity-sdk-js";
import { MessageList } from "@/components/ai-chat";
import { textOfBlocks } from "@/components/ai-chat/model/blocks";
import type { ChatMessageActionVM } from "@/components/ai-chat/types";
import { useChatSessionView } from "@/composables/useChatSessionView";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";
import ArtifactDetailDialog from "@/components/ArtifactDetailDialog.vue";
import { copyText } from "@/utils/clipboard";

defineOptions({ name: "ChatPage" });

const route = useRoute();
const router = useRouter();
const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

const settingsVisible = ref(false);
const memoryVisible = ref(false);
const taskDialogVisible = ref(false);

// 会话视图：store 消息实体原样交给无状态组件；组件事件经此回接 store
const {
  messages: chatMessages,
  branches: chatBranches,
  artifacts: chatArtifacts,
  feedback: chatFeedback,
  scrollFollowEnabled: vmScrollFollowEnabled,
  hasMoreHistory: vmHasMoreHistory,
  loadingMore: vmLoadingMore,
  messagesError: vmMessagesError,
  branchDisabled: vmBranchDisabled,
  isStreaming,
  interruptedMessageId,
  onScrollFollowToggle,
  onReachTop,
  onRetryMessages,
  onMessageAction,
  onLoadArtifacts,
  onSwitchBranch,
} = useChatSessionView();

// 产物详情弹窗：由组件内移到页面层（页面/视图层拉取 getArtifactDetail 展示）
const artifactDialogRef = ref<InstanceType<typeof ArtifactDetailDialog>>();

function handleOpenArtifact(artifact: Artifact) {
  void artifactDialogRef.value?.open(artifact);
}

/** 复制：剪贴板写入与提示由页面层处理（不落 store） */
async function handleCopy(messageId: number) {
  const message = chatMessages.value.find((item) => item.id === messageId);
  const content = message ? textOfBlocks(message.blocks) : "";
  if (await copyText(content)) {
    ElMessage.success("已复制");
  } else {
    ElMessage.error("复制失败，请手动选择复制");
  }
}

/** 编辑用户消息：页面层弹窗，确认后走 store 编辑重发 */
function handleEdit(messageId: number) {
  const message = chatMessages.value.find(
    (item): item is Message => item.id === messageId && item.role === "user"
  );
  if (!message) return;
  ElMessageBox.prompt("编辑后将以新分支重新触发回复", "编辑消息", {
    inputValue: textOfBlocks(message.blocks),
    inputType: "textarea",
  })
    .then(({ value }) => {
      if (value?.trim()) chatStore.editMessage(message.id, value);
    })
    .catch(() => {});
}

/**
 * 动作处理表：键全集取自 ChatMessageActionVM["type"]，每个处理器只接自己的载荷变体。
 * 新增动作而这里没接处理时缺键，编译失败（store 侧另有穷尽检查，两处同时报错）。
 */
type MessageActionHandlers = {
  [T in ChatMessageActionVM["type"]]: (
    action: Extract<ChatMessageActionVM, { type: T }>
  ) => void;
};

/** copy/edit 页面就地处理（剪贴板、编辑弹窗），其余原样转交绑定层进 store */
const messageActionHandlers: MessageActionHandlers = {
  copy: (action) => void handleCopy(action.messageId),
  edit: (action) => handleEdit(action.messageId),
  regenerate: onMessageAction,
  retry: onMessageAction,
  quote: onMessageAction,
  speak: onMessageAction,
  delete: onMessageAction,
  feedback: onMessageAction,
  "resume-interrupt": onMessageAction,
  "apply-suggestion": onMessageAction,
};

/**
 * 单一消息动作入口：按 type 查表分发。
 * 分发处唯一的类型断言只用于恢复「动作类型 ↔ 载荷」的相关性——键与载荷同源于
 * ChatMessageActionVM，处理器与动作不会错配。
 */
function handleMessageAction(action: ChatMessageActionVM) {
  const handler = messageActionHandlers[action.type] as (
    action: ChatMessageActionVM
  ) => void;
  handler(action);
}

const LAST_CONVERSATION_KEY = "dehaze.chat.lastConversationId";

/** 读取本会话维度（self/admin）上次访问的会话 ID（localStorage 记忆） */
function readLastConversationId(): number {
  const raw = localStorage.getItem(
    `${LAST_CONVERSATION_KEY}.${chatStore.scope}`
  );
  const id = Number(raw);
  return Number.isInteger(id) && id > 0 ? id : 0;
}

/** 记录本会话维度最近访问的会话 ID（退出后再次进入恢复到离开位置） */
function rememberConversation(id: number) {
  if (!id) return;
  localStorage.setItem(
    `${LAST_CONVERSATION_KEY}.${chatStore.scope}`,
    String(id)
  );
}

/** 从会话列表中取"最近活跃"会话（有最后发言的最新的；置顶的旧会话不作为默认入口） */
function latestActiveConversation() {
  const withMessage = chatStore.conversations
    .filter((c) => c.lastMessageAt)
    .sort(
      (a, b) =>
        Date.parse(b.lastMessageAt ?? "") - Date.parse(a.lastMessageAt ?? "")
    );
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
  // joinInflight：ChatSidebar 挂载（子组件先于本页 onMounted）已发起同一请求时复用之，避免首屏重复请求
  await chatStore.fetchConversations(true);
  // 等待加载期间用户已发起对话（流式进行中）：跳过默认跳转/消息加载，
  // 避免 fetchMessages 覆盖占位消息或路由切换打断当前流（否则界面永久"正在思考"）
  if (isStreaming.value) return;
  const routeId = Number(route.params.conversationId);
  if (routeId) {
    if (routeId !== chatStore.currentConversationId) {
      try {
        await chatStore.fetchMessages(routeId);
      } catch {
        // 失败已进入 messagesError，由列表行内错误条提供重试
      }
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
    try {
      await chatStore.fetchMessages(conversation.id);
    } catch {
      // 历史加载失败已进入 messagesError（列表展示重试入口），本次不发送
      return;
    }
    rememberConversation(conversation.id);
    router.replace(`/chat/${conversation.id}`);
  }
  chatStore.sendMessage(text);
  chatUserStore.saveDraft(chatStore.currentConversationId, "");
  chatUserStore.fetchQuota().catch(() => {});
}

function handleStop() {
  chatStore.stopStreaming(interruptedMessageId.value ?? undefined);
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
          v-if="
            chatStore.messageList.length > 0 ||
            chatStore.messagesLoading ||
            vmMessagesError
          "
          :messages="chatMessages"
          :artifacts="chatArtifacts"
          :feedback="chatFeedback"
          :branches="chatBranches"
          scope="self"
          :scroll-follow-enabled="vmScrollFollowEnabled"
          :has-more-history="vmHasMoreHistory"
          :loading-more="vmLoadingMore"
          :error="vmMessagesError"
          :retrying="
            vmMessagesError?.kind === 'more'
              ? vmLoadingMore
              : chatStore.messagesLoading
          "
          :branch-disabled="vmBranchDisabled"
          @scroll-follow-toggle="onScrollFollowToggle"
          @reach-top="onReachTop"
          @retry="onRetryMessages"
          @message-action="handleMessageAction"
          @open-artifact="handleOpenArtifact"
          @load-artifacts="onLoadArtifacts"
          @switch-branch="onSwitchBranch"
        />
        <EmptyState v-else @select="handleSend" />

        <MessageInput
          :is-streaming="isStreaming"
          @send="handleSend"
          @stop="handleStop"
        />
      </div>
    </main>

    <ConversationSettings v-model="settingsVisible" />
    <MemoryPanel v-model="memoryVisible" />
    <SaveAsTaskDialog v-model="taskDialogVisible" />

    <ArtifactDetailDialog ref="artifactDialogRef" />
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
