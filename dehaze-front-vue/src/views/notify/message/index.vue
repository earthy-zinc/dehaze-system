<template>
  <div class="app-container message-center">
    <!-- 顶部标题区 -->
    <div class="page-header">
      <div class="header-title">
        <span class="title-text">消息中心</span>
        <span v-if="notificationStore.unreadCount > 0" class="unread-pill">
          {{ notificationStore.unreadCount }} 条未读
        </span>
      </div>
      <div class="header-actions">
        <el-button
          :disabled="notificationStore.unreadCount === 0"
          type="primary"
          plain
          @click="handleMarkAllRead"
        >
          <el-icon><Select /></el-icon>
          <span>全部已读</span>
        </el-button>
        <el-button link @click="router.push('/notify/settings')">
          <el-icon><Setting /></el-icon>
          <span>通知设置</span>
        </el-button>
      </div>
    </div>

    <!-- 类型筛选 + 搜索 -->
    <div class="filter-bar">
      <div class="type-tabs">
        <button
          v-for="tab in typeTabs"
          :key="tab.value ?? 'all'"
          :class="['type-tab', { active: activeTab === tab.value }]"
          @click="handleTabChange(tab.value)"
        >
          <span class="tab-label">{{ tab.label }}</span>
          <span
            v-if="tab.value === 'unread' && notificationStore.unreadCount > 0"
            class="tab-count"
          >
            {{ notificationStore.unreadCount }}
          </span>
        </button>
      </div>
      <el-input
        v-model="searchKeyword"
        class="search-input"
        clearable
        placeholder="搜索消息标题或正文"
        @keyup.enter="handleSearch"
        @clear="handleSearch"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
    </div>

    <!-- 消息列表 -->
    <div v-loading="loading" class="message-list">
      <template v-if="messageList.length > 0">
        <div
          v-for="message in messageList"
          :key="message.id"
          :class="['message-card', { unread: message.readStatus === 0 }]"
          @click="goDetail(message)"
        >
          <div :class="['type-stripe', `type-${message.type}`]"></div>
          <div class="card-body">
            <div class="card-meta">
              <span :class="['type-tag', `tag-${message.type}`]">
                <el-icon v-if="typeIconMap[message.type]">
                  <component :is="typeIconMap[message.type]" />
                </el-icon>
                {{ message.typeLabel }}
              </span>
              <span v-if="message.priority >= 3" class="priority-flag">
                <el-icon><WarnTriangleFilled /></el-icon>
                {{ message.priority === 4 ? "紧急" : "高优" }}
              </span>
              <span class="time-text">{{
                formatTime(message.createTime)
              }}</span>
            </div>
            <div class="card-title">
              <span v-if="message.readStatus === 0" class="unread-dot"></span>
              <span class="title-text">{{ message.title }}</span>
            </div>
            <div class="card-summary">{{ message.summary }}</div>
            <div class="card-footer">
              <span v-if="message.jumpUrl" class="jump-link">
                点击查看详情
                <el-icon><ArrowRight /></el-icon>
              </span>
              <el-button
                class="delete-btn"
                link
                type="danger"
                @click.stop="handleDelete(message)"
              >
                <el-icon><Delete /></el-icon>
                删除
              </el-button>
            </div>
          </div>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="暂无消息" :image-size="120">
        <template #description>
          <p class="empty-text">
            {{ searchKeyword ? "没有找到匹配的消息" : "所有消息都已处理完毕" }}
          </p>
        </template>
      </el-empty>
    </div>

    <pagination
      v-if="total > 0"
      v-model:limit="queryParams.pageSize"
      v-model:page="queryParams.pageNum"
      v-model:total="total"
      @pagination="handleQuery"
    />
  </div>
</template>

<script lang="ts" setup>
import { useNotificationStoreHook } from "@/store/modules/notification";
import { MessageAPI, MessageVO, MessageQuery } from "dehaze-sdk-js";
import {
  ArrowRight,
  Bell,
  Delete,
  Promotion,
  Search,
  Select,
  Setting,
  Star,
  WarnTriangleFilled,
} from "@element-plus/icons-vue";

defineOptions({ name: "NotifyMessage" });

const router = useRouter();
const notificationStore = useNotificationStoreHook();

const loading = ref(false);
const total = ref(0);
const messageList = ref<MessageVO[]>([]);
const activeTab = ref<string | null>(null);
const searchKeyword = ref("");
const queryParams = reactive<MessageQuery>({
  pageNum: 1,
  pageSize: 20,
});

const typeTabs = [
  { label: "全部", value: null },
  { label: "系统公告", value: "announcement" },
  { label: "业务通知", value: "business" },
  { label: "会员通知", value: "member" },
  { label: "未读", value: "unread" },
];

const typeIconMap: Record<string, any> = {
  announcement: Bell,
  business: Promotion,
  member: Star,
  alert: WarnTriangleFilled,
  critical_alert: WarnTriangleFilled,
  inbox: Promotion,
};

function formatTime(time: string) {
  if (!time) return "";
  const date = new Date(time.replace(/-/g, "/"));
  const now = new Date();
  const isSameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  if (isSameDay) return `${hh}:${mm}`;
  const M = String(date.getMonth() + 1).padStart(2, "0");
  const D = String(date.getDate()).padStart(2, "0");
  return `${M}-${D}`;
}

function handleTabChange(value: string | null) {
  activeTab.value = value;
  searchKeyword.value = "";
  queryParams.pageNum = 1;
  if (value === "unread") {
    queryParams.type = undefined;
    queryParams.readStatus = 0;
  } else {
    queryParams.type = value ?? undefined;
    queryParams.readStatus = undefined;
  }
  handleQuery();
}

function handleSearch() {
  queryParams.pageNum = 1;
  if (searchKeyword.value.trim()) {
    loading.value = true;
    MessageAPI.search({
      keyword: searchKeyword.value.trim(),
      pageNum: queryParams.pageNum,
      pageSize: queryParams.pageSize,
    })
      .then((data) => {
        messageList.value = data.list;
        total.value = data.total;
      })
      .finally(() => {
        loading.value = false;
      });
    return;
  }
  handleQuery();
}

function handleQuery() {
  loading.value = true;
  MessageAPI.getPage(queryParams)
    .then((data) => {
      messageList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

const debouncedSearch = useDebounceFn(handleSearch, 300);
watch(searchKeyword, () => {
  if (!searchKeyword.value) return;
  debouncedSearch();
});

function goDetail(message: MessageVO) {
  router.push(`/notify/message/detail?id=${message.id}`);
}

function handleDelete(message: MessageVO) {
  ElMessageBox.confirm(`确定删除消息「${message.title}」吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  })
    .then(() => {
      return MessageAPI.deleteByIds(String(message.id));
    })
    .then(() => {
      ElMessage.success("删除成功");
      messageList.value = messageList.value.filter((m) => m.id !== message.id);
      total.value -= 1;
    })
    .catch(() => {});
}

function handleMarkAllRead() {
  ElMessageBox.confirm("确定将所有未读消息标记为已读吗？", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "info",
    lockScroll: false,
  })
    .then(() => {
      return MessageAPI.markAllRead();
    })
    .then((res) => {
      ElMessage.success(`已标记 ${res.affectedCount} 条消息为已读`);
      messageList.value.forEach((m) => {
        m.readStatus = 1;
      });
      notificationStore.fetchUnreadCount();
    })
    .catch(() => {});
}

onMounted(() => {
  handleQuery();
  notificationStore.fetchUnreadCount();
});

onActivated(() => {
  notificationStore.fetchUnreadCount();
});
</script>

<style lang="scss" scoped>
.message-center {
  max-width: 960px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;

  .header-title {
    display: flex;
    gap: 12px;
    align-items: baseline;

    .title-text {
      font-size: 22px;
      font-weight: 600;
      color: var(--el-text-color-primary);
      letter-spacing: 0.5px;
    }

    .unread-pill {
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 500;
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
      border-radius: 10px;
    }
  }

  .header-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }
}

.filter-bar {
  display: flex;
  gap: 16px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;

  .type-tabs {
    display: flex;
    gap: 4px;
    align-items: center;
    overflow-x: auto;
    scrollbar-width: none;

    &::-webkit-scrollbar {
      display: none;
    }
  }

  .type-tab {
    position: relative;
    display: inline-flex;
    gap: 6px;
    align-items: center;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    color: var(--el-text-color-regular);
    white-space: nowrap;
    cursor: pointer;
    background: transparent;
    border: none;
    border-radius: 18px;
    transition: all 0.2s ease;

    &:hover {
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
    }

    &.active {
      color: #fff;
      background: var(--el-color-primary);
    }

    .tab-count {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      padding: 0 5px;
      font-size: 11px;
      font-weight: 600;
      color: var(--el-color-primary);
      background: #fff;
      border-radius: 9px;
    }

    &.active .tab-count {
      color: var(--el-color-primary);
    }
  }

  .search-input {
    flex-shrink: 0;
    width: 240px;
  }
}

.message-list {
  min-height: 240px;
}

.message-card {
  position: relative;
  display: flex;
  margin-bottom: 12px;
  overflow: hidden;
  cursor: pointer;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  transition: all 0.25s ease;

  &:hover {
    border-color: var(--el-color-primary-light-5);
    box-shadow: 0 4px 16px rgb(0 0 0 / 6%);
    transform: translateY(-1px);

    .delete-btn {
      opacity: 1;
    }
  }

  &.unread {
    background: linear-gradient(
      90deg,
      var(--el-color-primary-light-9) 0%,
      var(--el-bg-color) 30%
    );
    border-color: var(--el-color-primary-light-7);
  }

  .type-stripe {
    flex-shrink: 0;
    width: 4px;

    &.type-announcement {
      background: linear-gradient(180deg, #409eff, #79bbff);
    }

    &.type-business {
      background: linear-gradient(180deg, #13c2c2, #5cdbd3);
    }

    &.type-member {
      background: linear-gradient(180deg, #fa8c16, #ffc069);
    }

    &.type-alert {
      background: linear-gradient(180deg, #faad14, #ffd666);
    }

    &.type-critical_alert {
      background: linear-gradient(180deg, #f5222d, #ff7875);
    }

    &.type-inbox {
      background: linear-gradient(180deg, #8c8c8c, #bfbfbf);
    }
  }

  .card-body {
    flex: 1;
    padding: 14px 18px;
  }

  .card-meta {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 8px;

    .type-tag {
      display: inline-flex;
      gap: 4px;
      align-items: center;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 500;
      border-radius: 4px;

      &.tag-announcement {
        color: #409eff;
        background: #ecf5ff;
      }

      &.tag-business {
        color: #13c2c2;
        background: #e6fffb;
      }

      &.tag-member {
        color: #fa8c16;
        background: #fff7e6;
      }

      &.tag-alert {
        color: #faad14;
        background: #fffbe6;
      }

      &.tag-critical_alert {
        color: #f5222d;
        background: #fff1f0;
      }

      &.tag-inbox {
        color: #8c8c8c;
        background: #fafafa;
      }
    }

    .priority-flag {
      display: inline-flex;
      gap: 2px;
      align-items: center;
      font-size: 12px;
      font-weight: 500;
      color: #f5222d;
    }

    .time-text {
      margin-left: auto;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .card-title {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;

    .unread-dot {
      flex-shrink: 0;
      width: 8px;
      height: 8px;
      background: var(--el-color-primary);
      border-radius: 50%;
      box-shadow: 0 0 0 3px var(--el-color-primary-light-9);
    }

    .title-text {
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 15px;
      font-weight: 500;
      color: var(--el-text-color-primary);
      white-space: nowrap;
    }
  }

  &.unread .card-title .title-text {
    font-weight: 600;
  }

  .card-summary {
    display: -webkit-box;
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-line-clamp: 2;
    font-size: 13px;
    line-height: 1.5;
    color: var(--el-text-color-secondary);
    -webkit-box-orient: vertical;
  }

  .card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 8px;

    .jump-link {
      display: inline-flex;
      gap: 2px;
      align-items: center;
      font-size: 12px;
      color: var(--el-color-primary);
    }

    .delete-btn {
      opacity: 0;
      transition: opacity 0.2s ease;
    }
  }
}

.empty-text {
  margin: 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

@media (width <= 768px) {
  .filter-bar {
    flex-direction: column;
    align-items: stretch;

    .search-input {
      width: 100%;
    }
  }

  .message-card {
    .card-footer .delete-btn {
      opacity: 1;
    }
  }
}
</style>
