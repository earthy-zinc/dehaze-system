<!-- 会话列表：scope=self 本人会话（搜索/置顶/归档筛选/未读数/批量操作），scope=admin 全量审计只读 -->
<script lang="ts" setup>
import { Delete, Edit, Search, Top } from "@element-plus/icons-vue";
import type { CheckboxValueType } from "element-plus";
import type { ConversationVO } from "dehaze-sdk-js";
import { computed, ref } from "vue";
import type {
  BatchAction,
  ChatScope,
  ConversationFilterStatus,
} from "@/store/modules/chat";

defineOptions({ name: "ConversationList" });

const props = defineProps<{
  scope: ChatScope;
  conversations: ConversationVO[];
  loading?: boolean;
  selectionMode?: boolean;
}>();

const emit = defineEmits<{
  select: [conversation: ConversationVO];
  create: [];
  edit: [conversation: ConversationVO];
  delete: [conversation: ConversationVO];
  batch: [ids: number[], action: BatchAction];
  search: [keyword: string];
  "filter-change": [status: ConversationFilterStatus];
}>();

const keyword = ref("");
const filterStatus = ref<ConversationFilterStatus>(0);
const checkedIds = ref<number[]>([]);

const isSelf = computed(() => props.scope === "self");

const filterOptions: Array<{ label: string; value: ConversationFilterStatus }> =
  [
    { label: "全部", value: 0 },
    { label: "活跃", value: 1 },
    { label: "已归档", value: 2 },
  ];

function handleSearch() {
  emit("search", keyword.value.trim());
}

function handleFilterChange(value: ConversationFilterStatus) {
  filterStatus.value = value;
  emit("filter-change", value);
}

function toggleChecked(id: number, checked: boolean) {
  checkedIds.value = checked
    ? [...checkedIds.value, id]
    : checkedIds.value.filter((item) => item !== id);
}

function batch(action: BatchAction) {
  emit("batch", [...checkedIds.value], action);
  checkedIds.value = [];
}

function formatTime(time?: string) {
  if (!time) return "";
  return time.slice(5, 16).replace("T", " ");
}
</script>

<template>
  <div class="conversation-list" v-loading="loading">
    <div class="conversation-list__toolbar">
      <el-input
        v-model="keyword"
        :placeholder="isSelf ? '搜索会话标题/消息内容' : '搜索用户/会话'"
        clearable
        :prefix-icon="Search"
        @keyup.enter="handleSearch"
        @clear="handleSearch"
      />
      <el-select
        :model-value="filterStatus"
        style="width: 96px"
        @change="handleFilterChange"
      >
        <el-option
          v-for="option in filterOptions"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
      <el-button
        v-if="isSelf && !selectionMode"
        type="primary"
        @click="emit('create')"
      >
        新会话
      </el-button>
    </div>

    <div v-if="isSelf && selectionMode" class="conversation-list__batch">
      <span>已选 {{ checkedIds.length }} 项</span>
      <el-button
        size="small"
        :disabled="checkedIds.length === 0"
        @click="batch('archive')"
      >
        归档
      </el-button>
      <el-button
        size="small"
        type="danger"
        :disabled="checkedIds.length === 0"
        @click="batch('delete')"
      >
        删除
      </el-button>
    </div>

    <div class="conversation-list__items">
      <div
        v-for="conversation in conversations"
        :key="conversation.id"
        class="conversation-item"
        @click="emit('select', conversation)"
      >
        <el-checkbox
          v-if="isSelf && selectionMode"
          :model-value="checkedIds.includes(conversation.id)"
          @click.stop
          @change="
            (checked: CheckboxValueType) =>
              toggleChecked(conversation.id, checked === true)
          "
        />
        <div class="conversation-item__main">
          <div class="conversation-item__title-row">
            <el-icon
              v-if="conversation.pinned === 1"
              class="conversation-item__pin"
            >
              <Top />
            </el-icon>
            <span class="conversation-item__title">{{
              conversation.title
            }}</span>
            <el-badge
              v-if="isSelf && (conversation.unreadCount ?? 0) > 0"
              :value="conversation.unreadCount"
              :max="99"
              class="conversation-item__unread"
            />
          </div>
          <div class="conversation-item__meta">
            <template v-if="isSelf">
              <span>{{ conversation.messageCount }} 条</span>
              <span>{{ formatTime(conversation.lastMessageAt) }}</span>
            </template>
            <template v-else>
              <span>{{
                conversation.userName ?? `用户 ${conversation.userId ?? "-"}`
              }}</span>
              <span>{{ conversation.model }}</span>
              <span>{{ conversation.messageCount }} 条</span>
              <span v-if="conversation.tokenConsumed != null">
                Token {{ conversation.tokenConsumed }}
              </span>
              <span v-if="conversation.creditsConsumed != null">
                积分 {{ conversation.creditsConsumed }}
              </span>
            </template>
          </div>
          <div
            v-if="conversation.anomalyLabel"
            class="conversation-item__anomaly"
          >
            <el-tag type="danger" size="small">{{
              conversation.anomalyLabel
            }}</el-tag>
          </div>
        </div>
        <div
          v-if="isSelf && !selectionMode"
          class="conversation-item__actions"
          @click.stop
        >
          <el-tooltip content="重命名" placement="top">
            <el-button link size="small" @click="emit('edit', conversation)">
              <el-icon><Edit /></el-icon>
            </el-button>
          </el-tooltip>
          <el-tooltip content="删除" placement="top">
            <el-button
              link
              size="small"
              type="danger"
              @click="emit('delete', conversation)"
            >
              <el-icon><Delete /></el-icon>
            </el-button>
          </el-tooltip>
        </div>
      </div>
      <el-empty
        v-if="conversations.length === 0 && !loading"
        description="暂无会话"
      />
    </div>
  </div>
</template>

<style scoped lang="scss">
.conversation-list {
  display: flex;
  flex-direction: column;
  height: 100%;

  &__toolbar {
    display: flex;
    gap: 8px;
    padding: 8px;
  }

  &__batch {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__items {
    flex: 1;
    overflow-y: auto;
  }
}

.conversation-item {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
  cursor: pointer;

  &:hover {
    background-color: var(--el-fill-color-light);

    .conversation-item__actions {
      visibility: visible;
    }
  }

  &__main {
    flex: 1;
    min-width: 0;
  }

  &__title-row {
    display: flex;
    gap: 4px;
    align-items: center;
  }

  &__title {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 14px;
    white-space: nowrap;
  }

  &__pin {
    color: var(--el-color-warning);
  }

  &__meta {
    display: flex;
    gap: 8px;
    margin-top: 2px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__actions {
    visibility: hidden;
  }
}
</style>
