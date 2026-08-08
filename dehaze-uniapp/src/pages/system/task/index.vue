<template>
  <PageLayout level="L2" title="任务管理">
    <view class="page-body">
      <view class="search-bar">
        <u-search
          v-model="keyword"
          placeholder="搜索任务类型"
          @search="handleSearch"
          @clear="handleSearch"
        />
      </view>
      <u-table>
        <u-tr v-for="item in list" :key="item.taskId">
          <u-td>{{ taskTypeLabel(item.taskType) }}</u-td>
          <u-td>{{ item.taskId?.slice(0, 8) }}...</u-td>
          <u-td>
            <u-tag
              :text="statusMap[item.status] || String(item.status)"
              :type="statusTagType(item.status)"
              size="mini"
            />
          </u-td>
          <u-td>
            <u-button
              v-if="item.status === 1 || item.status === 2"
              size="mini"
              type="error"
              @click="cancelTask(item.taskId)"
              >取消</u-button
            >
            <u-button
              v-if="item.status === 4"
              size="mini"
              type="warning"
              @click="retryTask(item.taskId)"
              >重试</u-button
            >
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="!loading && list.length === 0" text="暂无任务" />
      <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import { TaskAPI } from "dehaze-sdk-js";
import type { TaskStatus } from "dehaze-sdk-js";

const statusMap: Record<TaskStatus, string> = {
  1: "待处理",
  2: "处理中",
  3: "已完成",
  4: "失败",
  5: "已取消",
};
const list = ref<any[]>([]);
const keyword = ref("");
const pageNum = ref(1);
const hasMore = ref(false);
const loading = ref(false);

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

const fetchList = async (reset = false) => {
  if (reset) {
    pageNum.value = 1;
    list.value = [];
  }
  loading.value = true;
  try {
    const res = await TaskAPI.getPage({
      pageNum: pageNum.value,
      pageSize: 20,
      taskType: keyword.value || undefined,
    });
    const records = res.list || [];
    if (reset) list.value = records;
    else list.value.push(...records);
    hasMore.value = records.length === 20;
    pageNum.value++;
  } finally {
    loading.value = false;
  }
};

const handleSearch = () => fetchList(true);
const loadMore = () => fetchList();
const statusTagType = (s: number) =>
  s === 3 ? "success" : s === 4 ? "error" : s === 5 ? "info" : "warning";

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
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
