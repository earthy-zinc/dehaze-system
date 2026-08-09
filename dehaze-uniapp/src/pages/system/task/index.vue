<template>
  <PageLayout level="L2" title="任务管理">
    <view class="page-body">
      <view class="search-bar">
        <input
          class="search-input"
          v-model="keyword"
          placeholder="搜索任务类型"
          confirm-type="search"
          @confirm="handleSearch"
        />
      </view>
      <view class="list-row list-row-head">
        <text class="cell">任务类型</text>
        <text class="cell">任务ID</text>
        <text class="cell">状态</text>
        <text class="cell">操作</text>
      </view>
      <view v-for="item in list" :key="item.taskId" class="list-row">
        <text class="cell">{{ taskTypeLabel(item.taskType) }}</text>
        <text class="cell cell-id">{{ item.taskId?.slice(0, 8) }}...</text>
        <view class="cell">
          <view class="tag tag-sm" :class="'tag-' + statusTagType(item.status)">
            {{ statusMap[item.status] || String(item.status) }}
          </view>
        </view>
        <view class="cell cell-actions">
          <button
            v-if="item.status === 1 || item.status === 2"
            class="btn btn-danger btn-sm"
            @click="cancelTask(item.taskId)"
          >
            取消
          </button>
          <button
            v-if="item.status === 4"
            class="btn btn-warning btn-sm"
            @click="retryTask(item.taskId)"
          >
            重试
          </button>
        </view>
      </view>
      <view v-if="!loading && list.length === 0" class="empty-tip"
        >暂无任务</view
      >
      <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import PageLayout from "@/layout/index.vue";
import { usePagedList } from "@/composables/usePagedList";
import { TaskAPI } from "dehaze-sdk-js";

const statusMap: Record<number, string> = {
  1: "待处理",
  2: "处理中",
  3: "已完成",
  4: "失败",
  5: "已取消",
};

const { list, keyword, hasMore, loading, fetchList, handleSearch, loadMore } =
  usePagedList<any>({
    fetcher: (p) =>
      TaskAPI.getPage({
        pageNum: p.pageNum,
        pageSize: 20,
        taskType: p.keyword,
      }).then((r) => r.list || []),
  });

const taskTypeLabel = (type?: string) => {
  const map: Record<string, string> = {
    dataset_export: "数据集导出",
    user_export: "用户导出",
    user_import: "用户导入",
    role_export: "角色导出",
    role_import: "角色导入",
    dept_export: "部门导出",
    dept_import: "部门导入",
    menu_export: "菜单导出",
    menu_import: "菜单导入",
    dict_export: "字典导出",
    dict_import: "字典导入",
    algorithm_export: "算法导出",
    algorithm_import: "算法导入",
  };
  return map[type || ""] || type || "未知类型";
};

const statusTagType = (s: number) => {
  switch (s) {
    case 3:
      return "success";
    case 4:
      return "danger";
    case 5:
      return "info";
    default:
      return "warning";
  }
};

const cancelTask = async (taskId: string) => {
  const res = await uni.showModal({
    title: "确认取消",
    content: "确定取消该任务吗？",
  });
  if (!res.confirm) return;
  try {
    await TaskAPI.cancel(taskId);
    uni.showToast({ title: "已取消", icon: "success" });
    fetchList(true);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};
const retryTask = async (taskId: string) => {
  try {
    await TaskAPI.retry(taskId);
    uni.showToast({ title: "已重试", icon: "success" });
    fetchList(true);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};

fetchList(true);
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.search-bar {
  margin-bottom: 20rpx;

  .search-input {
    width: 100%;
    box-sizing: border-box;
    padding: 14rpx 20rpx;
    font-size: 28rpx;
    background: $color-bg-secondary;
    border-radius: $radius-md;
  }
}
.list-row {
  display: flex;
  align-items: center;
  padding: 20rpx 16rpx;
  border-bottom: 1rpx solid $color-border;
  font-size: 26rpx;

  .cell {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cell-id {
    color: $color-text-secondary;
  }
  .cell-actions {
    display: flex;
    gap: 8rpx;
  }
}
.list-row-head {
  background: $color-bg-secondary;
  font-weight: 600;
  color: $color-text-secondary;
}
.tag {
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-sm {
  padding: 2rpx 10rpx;
}
.tag-primary {
  color: $color-primary;
  background: $color-primary-bg;
}
.tag-success {
  color: $color-success;
  background: $color-success-bg;
}
.tag-warning {
  color: $color-warning;
  background: $color-warning-bg;
}
.tag-danger {
  color: $color-danger;
  background: $color-danger-bg;
}
.tag-info {
  color: $color-text-secondary;
  background: $color-bg-secondary;
}
.btn {
  padding: 8rpx 20rpx;
  border-radius: $radius-sm;
  font-size: $font-sm;
  line-height: 1.6;
  &::after {
    border: none;
  }
}
.btn-sm {
  padding: 4rpx 16rpx;
  font-size: $font-xs;
}
.btn-warning {
  color: $color-white;
  background: $color-warning;
}
.btn-danger {
  color: $color-white;
  background: $color-danger;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
