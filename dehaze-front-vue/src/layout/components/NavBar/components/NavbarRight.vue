<template>
  <div class="flex">
    <!-- 消息通知 -->
    <el-popover
      v-model:visible="popoverVisible"
      :width="380"
      placement="bottom-end"
      trigger="click"
      popper-class="message-popover"
    >
      <template #reference>
        <div class="setting-item message-entry" @click="handleIconClick">
          <el-badge
            :value="notificationStore.unreadCount"
            :max="99"
            :hidden="notificationStore.unreadCount === 0"
            class="message-badge"
          >
            <svg-icon icon-class="message" class="message-icon" />
          </el-badge>
        </div>
      </template>

      <div class="popover-header">
        <span class="popover-title">消息通知</span>
        <span v-if="notificationStore.unreadCount > 0" class="popover-count">
          {{ notificationStore.unreadCount }} 条未读
        </span>
      </div>

      <div v-loading="recentLoading" class="popover-body">
        <template v-if="recentMessages.length > 0">
          <div
            v-for="msg in recentMessages"
            :key="msg.id"
            :class="['recent-item', { unread: msg.readStatus === 0 }]"
            @click="goDetail(msg.id)"
          >
            <span :class="['recent-type', `type-${msg.type}`]"></span>
            <div class="recent-content">
              <div class="recent-title">{{ msg.title }}</div>
              <div class="recent-time">{{ msg.createTime }}</div>
            </div>
          </div>
        </template>
        <el-empty v-else :image-size="60" description="暂无消息" />
      </div>

      <div class="popover-footer">
        <el-button link @click="goSettings">通知设置</el-button>
        <el-button
          v-if="notificationStore.unreadCount > 0"
          link
          type="primary"
          @click="markAllRead"
        >
          全部已读
        </el-button>
        <el-button link type="primary" @click="goMessageCenter"
          >查看全部</el-button
        >
      </div>
    </el-popover>

    <!-- 用户头像 -->
    <el-dropdown class="setting-item" trigger="click">
      <div class="flex-center h100% p10px">
        <img
          :src="userStore.user.avatar"
          class="rounded-full mr-10px w24px w24px"
        />
        <span>{{ userStore.user.username }}</span>
      </div>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item divided @click="logout">
            {{ $t("navbar.logout") }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>

    <!-- 设置 -->
    <template v-if="defaultSettings.showSettings">
      <div class="setting-item" @click="settingStore.settingsVisible = true">
        <svg-icon icon-class="setting" />
      </div>
    </template>
  </div>
</template>
<script lang="ts" setup>
import defaultSettings from "@/settings";
import { useNotificationStoreHook } from "@/store/modules/notification";
import { useSettingsStore, useTagsViewStore, useUserStore } from "@/store";
import { MessageAPI, MessageVO } from "dehaze-sdk-js";

const tagsViewStore = useTagsViewStore();
const userStore = useUserStore();
const settingStore = useSettingsStore();
const notificationStore = useNotificationStoreHook();

const route = useRoute();
const router = useRouter();

const popoverVisible = ref(false);
const recentLoading = ref(false);
const recentMessages = ref<MessageVO[]>([]);
let pollTimer: number | null = null;

function loadRecent() {
  recentLoading.value = true;
  MessageAPI.getPage({ pageNum: 1, pageSize: 5, readStatus: 0 })
    .then((data) => {
      recentMessages.value = data.list;
    })
    .finally(() => {
      recentLoading.value = false;
    });
}

function handleIconClick() {
  loadRecent();
}

function goDetail(id: number) {
  popoverVisible.value = false;
  router.push(`/notify/message/detail?id=${id}`);
}

function goMessageCenter() {
  popoverVisible.value = false;
  router.push("/notify/message");
}

function goSettings() {
  popoverVisible.value = false;
  router.push("/notify/settings");
}

function markAllRead() {
  MessageAPI.markAllRead().then((res) => {
    ElMessage.success(`已标记 ${res.affectedCount} 条消息为已读`);
    recentMessages.value = [];
    notificationStore.fetchUnreadCount();
  });
}

function startPolling() {
  stopPolling();
  pollTimer = window.setInterval(() => {
    if (document.visibilityState === "visible") {
      notificationStore.fetchUnreadCount();
    }
  }, 60000);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function handleVisibilityChange() {
  if (document.visibilityState === "visible") {
    notificationStore.fetchUnreadCount();
  }
}

/**
 * 注销
 */
function logout() {
  ElMessageBox.confirm("确定注销并退出系统吗？", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
    lockScroll: false,
  }).then(() => {
    userStore
      .logout()
      .then(() => {
        tagsViewStore.delAllViews();
      })
      .then(() => {
        router.push(`/login?redirect=${route.fullPath}`);
      });
  });
}

onMounted(() => {
  notificationStore.fetchUnreadCount();
  startPolling();
  document.addEventListener("visibilitychange", handleVisibilityChange);
});

onBeforeUnmount(() => {
  stopPolling();
  document.removeEventListener("visibilitychange", handleVisibilityChange);
});

watch(
  () => route.path,
  () => {
    if (popoverVisible.value) popoverVisible.value = false;
  }
);
</script>
<style lang="scss" scoped>
.setting-item {
  display: inline-block;
  min-width: 40px;
  height: $navbar-height;
  line-height: $navbar-height;
  color: var(--el-text-color);
  text-align: center;
  cursor: pointer;

  &:hover {
    background: rgb(0 0 0 / 10%);
  }
}

.message-entry {
  display: flex;
  align-items: center;
  justify-content: center;

  .message-icon {
    font-size: 20px;
    color: var(--el-text-color);
    transition: transform 0.2s ease;
  }

  &:hover .message-icon {
    transform: scale(1.1);
  }
}

.layout-top,
.layout-mix {
  .setting-item,
  .el-icon {
    color: var(--el-text-color);
  }
}

.dark .setting-item:hover {
  background: rgb(255 255 255 / 20%);
}
</style>

<style lang="scss">
.message-popover {
  padding: 0 !important;

  .popover-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--el-border-color-lighter);

    .popover-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .popover-count {
      font-size: 12px;
      color: var(--el-color-primary);
    }
  }

  .popover-body {
    min-height: 80px;
    max-height: 360px;
    overflow-y: auto;
  }

  .recent-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 16px;
    cursor: pointer;
    transition: background 0.2s ease;

    &:hover {
      background: var(--el-fill-color-light);
    }

    &.unread .recent-title {
      font-weight: 600;
    }

    .recent-type {
      flex-shrink: 0;
      width: 6px;
      height: 6px;
      margin-top: 6px;
      border-radius: 50%;

      &.type-announcement {
        background: #409eff;
      }

      &.type-business {
        background: #13c2c2;
      }

      &.type-member {
        background: #fa8c16;
      }

      &.type-alert {
        background: #faad14;
      }

      &.type-critical_alert {
        background: #f5222d;
      }

      &.type-inbox {
        background: #8c8c8c;
      }
    }

    .recent-content {
      flex: 1;
      min-width: 0;
    }

    .recent-title {
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 13px;
      line-height: 1.4;
      color: var(--el-text-color-primary);
      white-space: nowrap;
    }

    .recent-time {
      margin-top: 2px;
      font-size: 11px;
      color: var(--el-text-color-secondary);
    }
  }

  .popover-footer {
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-top: 1px solid var(--el-border-color-lighter);

    .el-button {
      flex: 1;
      font-size: 12px;
    }
  }
}
</style>
