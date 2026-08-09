<template>
  <PageLayout level="L2" title="会员管理">
    <view class="page-body">
      <view class="search-bar">
        <input
          class="search-input"
          v-model="keyword"
          placeholder="搜索会员昵称/用户名"
          confirm-type="search"
          @confirm="handleSearch"
        />
      </view>
      <view class="filter-row">
        <view
          v-for="lv in levelFilters"
          :key="lv.value"
          class="tag tag-sm"
          :class="levelFilter === lv.value ? 'tag-primary' : 'tag-info'"
          @click="handleLevelFilter(lv.value)"
        >
          {{ lv.label }}
        </view>
      </view>
      <view class="list-row list-row-head">
        <text class="cell">会员</text>
        <text class="cell">等级</text>
        <text class="cell">成长值</text>
        <text class="cell">状态</text>
        <text class="cell"></text>
      </view>
      <view
        v-for="item in list"
        :key="item.userId"
        class="list-row"
        @click="goDetail(item.userId)"
      >
        <text class="cell">{{ item.nickname || item.username }}</text>
        <text class="cell">{{ item.levelName || item.levelCode }}</text>
        <text class="cell">{{ item.growthValue || 0 }}</text>
        <view class="cell">
          <view
            class="tag tag-sm"
            :class="item.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ item.status === 1 ? "正常" : "冻结" }}
          </view>
        </view>
        <view class="cell"><SvgIcon name="arrow-right" /></view>
      </view>
      <view v-if="!loading && list.length === 0" class="empty-tip"
        >暂无会员</view
      >
      <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { usePagedList } from "@/composables/usePagedList";
import { MemberAPI } from "dehaze-sdk-js";

const levelFilters = [
  { label: "全部", value: "" },
  { label: "普通用户", value: "level_1" },
  { label: "白银会员", value: "level_2" },
  { label: "黄金会员", value: "level_3" },
];

const levelFilter = ref("");

const { list, keyword, hasMore, loading, fetchList, handleSearch, loadMore } =
  usePagedList<any>({
    fetcher: (p) => {
      const params: any = { pageNum: p.pageNum, pageSize: 20 };
      if (p.keyword) params.keywords = p.keyword;
      if (levelFilter.value) params.levelCode = levelFilter.value;
      return MemberAPI.getPage(params).then((r) => r.list || []);
    },
  });

const handleLevelFilter = (val: string) => {
  levelFilter.value = val;
  fetchList(true);
};
const goDetail = (id: number) =>
  uni.navigateTo({ url: `/pages/system/member/detail?id=${id}` });

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
.filter-row {
  display: flex;
  gap: 12rpx;
  flex-wrap: wrap;
  margin-bottom: 20rpx;
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
.tag-danger {
  color: $color-danger;
  background: $color-danger-bg;
}
.tag-info {
  color: $color-text-secondary;
  background: $color-bg-secondary;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
