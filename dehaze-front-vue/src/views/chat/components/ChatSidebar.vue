<!-- 会话侧栏：新建会话 + 公共会话列表（搜索/筛选/批量/重命名/删除），会话导航由本组件直接路由 -->
<!-- 命名说明：ChatSidebar 避免与布局侧栏 src/layout/components/Sidebar 自动导入组件名冲突 -->
<script lang="ts" setup>
import { Operation } from "@element-plus/icons-vue";
import type { ConversationVO } from "dehaze-sdk-js";
import { onMounted } from "vue";
import { useRouter } from "vue-router";
import { useChatStore } from "@/store/modules/chat";
import type { ConversationFilterStatus } from "@/store/modules/chat";

defineOptions({ name: "ChatSidebar" });

const emit = defineEmits<{
  select: [conversationId: number];
}>();

const router = useRouter();
const chatStore = useChatStore();

onMounted(() => {
  // 路由直入 /chat/:id 时页面仅拉取消息，会话列表在此兜底加载
  if (chatStore.conversations.length === 0 && !chatStore.conversationsLoading) {
    chatStore.fetchConversations();
  }
});

async function handleCreate() {
  const conversation = await chatStore.createConversation();
  chatStore.selectionMode = false;
  emit("select", conversation.id);
}

function handleSelect(conversation: ConversationVO) {
  emit("select", conversation.id);
}

function handleEdit(conversation: ConversationVO) {
  ElMessageBox.prompt("修改会话标题", "重命名", {
    inputValue: conversation.title,
  })
    .then(({ value }) => {
      if (!value?.trim()) return;
      chatStore.updateConversation(conversation.id, { title: value.trim() });
    })
    .catch(() => {});
}

function handleDelete(conversation: ConversationVO) {
  ElMessageBox.confirm(
    `确认删除会话 "${conversation.title}" ？删除后进入回收站`,
    "删除确认",
    { type: "warning" }
  )
    .then(async () => {
      const isCurrent = conversation.id === chatStore.currentConversationId;
      await chatStore.deleteConversation(conversation.id);
      if (isCurrent) router.replace("/chat");
    })
    .catch(() => {});
}

function handleBatch(ids: number[], action: "archive" | "restore" | "delete") {
  const submit = () => {
    chatStore
      .batchOperateConversations(action, ids)
      .then(() => {
        const currentId = chatStore.currentConversationId;
        if (action === "delete" && currentId && ids.includes(currentId)) {
          router.replace("/chat");
        }
      })
      .catch(() => {});
  };
  if (action === "delete") {
    ElMessageBox.confirm(
      `确认删除选中的 ${ids.length} 个会话？删除后进入回收站`,
      "批量删除",
      { type: "warning" }
    )
      .then(submit)
      .catch(() => {});
    return;
  }
  submit();
}

function handleSearch(keyword: string) {
  chatStore.conversationQuery.keyword = keyword;
  chatStore.fetchConversations();
}

function handleFilterChange(status: ConversationFilterStatus) {
  if (status === 0) {
    delete chatStore.conversationQuery.status;
  } else {
    chatStore.conversationQuery.status = status;
  }
  chatStore.fetchConversations();
}
</script>

<template>
  <div class="chat-sidebar">
    <div class="chat-sidebar__header">
      <el-button type="primary" class="chat-sidebar__new" @click="handleCreate">
        新建会话
      </el-button>
      <el-tooltip
        :content="chatStore.selectionMode ? '退出批量操作' : '批量操作'"
        placement="top"
      >
        <el-button
          :type="chatStore.selectionMode ? 'warning' : 'default'"
          :icon="Operation"
          @click="chatStore.selectionMode = !chatStore.selectionMode"
        />
      </el-tooltip>
    </div>

    <ConversationList
      scope="self"
      :conversations="chatStore.conversations"
      :loading="chatStore.conversationsLoading"
      :selection-mode="chatStore.selectionMode"
      @select="handleSelect"
      @edit="handleEdit"
      @delete="handleDelete"
      @batch="handleBatch"
      @search="handleSearch"
      @filter-change="handleFilterChange"
    />
  </div>
</template>

<style scoped lang="scss">
.chat-sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;

  &__header {
    display: flex;
    gap: 8px;
    padding: 12px;
  }

  &__new {
    flex: 1;
  }
}
</style>
