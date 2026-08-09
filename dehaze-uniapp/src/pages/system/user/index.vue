<template>
  <PageLayout level="L2" title="用户管理">
    <view class="page-body">
      <view class="search-bar">
        <input
          class="search-input"
          v-model="keyword"
          placeholder="搜索用户名/手机号"
          confirm-type="search"
          @confirm="handleSearch"
        />
      </view>
      <view class="list-row" v-for="user in list" :key="user.id">
        <text class="cell" @click="goDetail(user.id)">{{ user.nickname }}</text>
        <text class="cell" @click="goDetail(user.id)"
          >@{{ user.username }}</text
        >
        <view class="cell" @click="goDetail(user.id)">
          <view
            class="tag"
            :class="user.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ user.status === 1 ? "正常" : "禁用" }}
          </view>
        </view>
        <view class="cell row-actions">
          <SvgIcon v-if="canEdit" name="edit-pen" @click="goDetail(user.id)" />
          <SvgIcon
            v-if="canDelete"
            name="trash"
            color="#ef4444"
            @click="delUser(user.id)"
          />
        </view>
      </view>
      <view v-if="!loading && list.length === 0" class="empty-tip"
        >暂无用户</view
      >
      <view v-if="hasMore" class="load-more" @click="loadMore">加载更多</view>
    </view>
    <FabButton v-if="canAdd" @click="goDetail(0)">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
  </PageLayout>
</template>

<script setup lang="ts">
import { computed } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import { usePagedList } from "@/composables/usePagedList";
import { UserAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:user:add"));
const canEdit = computed(() => authStore.hasPerm("sys:user:edit"));
const canDelete = computed(() => authStore.hasPerm("sys:user:delete"));

const { list, keyword, hasMore, loading, fetchList, handleSearch, loadMore } =
  usePagedList({
    fetcher: (p) =>
      UserAPI.getPage({
        pageNum: p.pageNum,
        pageSize: p.pageSize,
        keywords: p.keyword,
      }).then((r) => r.list || []),
  });

const goDetail = (id?: number) => {
  if (id === undefined) return;
  uni.navigateTo({ url: `/pages/system/user/detail?id=${id}` });
};

const delUser = async (id?: number) => {
  if (id === undefined) return;
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除该用户吗？",
  });
  if (!res.confirm) return;
  try {
    await UserAPI.deleteByIds(String(id));
    uni.showToast({ title: "删除成功", icon: "success" });
    fetchList(true);
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
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
.search-input {
  width: 100%;
  height: 72rpx;
  padding: 0 20rpx;
  background: $color-bg-secondary;
  border-radius: $radius-md;
  font-size: $font-sm;
}
.list-row {
  display: flex;
  align-items: center;
  padding: 24rpx 20rpx;
  border-bottom: 1rpx solid $color-border;

  .cell {
    flex: 1;
    font-size: $font-sm;
    color: $color-text-primary;
  }

  .row-actions {
    display: flex;
    gap: 16rpx;
  }
}
.tag {
  display: inline-block;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-success {
  background: $color-success-bg;
  color: $color-success;
}
.tag-danger {
  background: $color-danger-bg;
  color: $color-danger;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
