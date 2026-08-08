<template>
  <PageLayout level="L2" title="消息详情">
    <view class="message-detail-page">
      <view v-if="loading" class="message-detail-status">
        <text>加载中...</text>
      </view>
      <view v-else-if="!message" class="message-detail-status">
        <text>消息不存在</text>
      </view>
      <template v-else>
        <view class="message-detail-header">
          <view class="message-detail-type">
            <text>{{ message.typeLabel }}</text>
          </view>
          <text class="message-detail-title">{{ message.title }}</text>
          <text class="message-detail-time">{{ formatDateTime(message.createTime) }}</text>
        </view>
        <view class="message-detail-body">
          <rich-text
            v-if="message.content"
            :nodes="message.content"
            class="message-detail-content"
          />
          <text v-else class="message-detail-summary">
            {{ message.summary || "暂无内容" }}
          </text>
        </view>
        <view class="message-detail-footer">
          <view
            v-if="message.jumpUrl"
            class="message-detail-action"
            @click="handleJump"
          >
            <text>查看详情 →</text>
          </view>
          <view class="message-detail-delete" @click="handleDelete">
            <text>{{ deleting ? "删除中..." : "删除" }}</text>
          </view>
        </view>
      </template>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { MessageAPI } from "dehaze-sdk-js";
import type { MessageVO } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

const message = ref<MessageVO | null>(null);
const loading = ref(true);
const deleting = ref(false);

function formatDateTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

onMounted(() => {
  const pages = getCurrentPages();
  const currentPage = pages[pages.length - 1] as { options?: { id?: string } };
  const id = currentPage?.options?.id;

  if (!id) {
    uni.showToast({ title: "参数错误", icon: "none" });
    setTimeout(() => uni.navigateBack(), 1000);
    loading.value = false;
    return;
  }

  MessageAPI.getDetail(Number(id))
    .then((res) => {
      message.value = res;
      if (res.readStatus === 0) {
        MessageAPI.markRead(res.id).catch(() => {});
      }
    })
    .catch((error) => {
      uni.showToast({
        title: getErrorMessage(error, "加载失败"),
        icon: "none",
      });
    })
    .finally(() => {
      loading.value = false;
    });
});

const handleJump = () => {
  if (message.value?.jumpUrl) {
    uni.navigateTo({ url: message.value.jumpUrl });
  }
};

const handleDelete = async () => {
  if (!message.value || deleting.value) return;
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除这条消息吗？",
  });
  if (!res.confirm) return;
  deleting.value = true;
  try {
    await MessageAPI.deleteByIds(String(message.value.id));
    uni.showToast({ title: "已删除", icon: "success" });
    setTimeout(() => uni.navigateBack(), 800);
  } catch (error) {
    uni.showToast({ title: getErrorMessage(error, "删除失败"), icon: "none" });
  } finally {
    deleting.value = false;
  }
};
</script>

<style lang="scss" scoped>
.message-detail-page {
  min-height: 100vh;
  padding: 24rpx;
}

.message-detail-header {
  padding: 32rpx 24rpx;
  margin-bottom: 24rpx;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-sm;
}

.message-detail-type {
  display: inline-block;
  padding: 4rpx 16rpx;
  margin-bottom: 16rpx;
  font-size: $font-xs;
  color: $color-primary;
  background: $color-primary-bg;
  border-radius: $radius-sm;
}

.message-detail-title {
  display: block;
  margin-bottom: 12rpx;
  font-size: $font-lg;
  font-weight: 700;
  line-height: 1.4;
  color: $color-text-primary;
}

.message-detail-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.message-detail-body {
  min-height: 200rpx;
  padding: 24rpx;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-sm;
}

.message-detail-content {
  font-size: $font-md;
  line-height: 1.8;
  color: $color-text-primary;
}

.message-detail-summary {
  font-size: $font-md;
  line-height: 1.8;
  color: $color-text-secondary;
}

.message-detail-footer {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
  margin-top: 32rpx;
}

.message-detail-action {
  padding: 24rpx;
  font-size: $font-md;
  font-weight: 600;
  color: $color-primary;
  text-align: center;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-sm;
}

.message-detail-delete {
  padding: 24rpx;
  font-size: $font-md;
  color: $color-danger;
  text-align: center;
  background: $color-white;
  border-radius: $radius-lg;
  box-shadow: $shadow-sm;
}

.message-detail-status {
  display: flex;
  justify-content: center;
  padding: 96rpx;
  font-size: $font-sm;
  color: $color-text-placeholder;
}
</style>
