<template>
  <PageLayout level="L2" title="会员管理">
    <view class="page-body">
      <view class="search-bar">
        <u-search
          v-model="keyword"
          placeholder="搜索会员昵称/用户名"
          @search="handleSearch"
          @clear="handleSearch"
        />
      </view>
      <view class="filter-row">
        <u-tag
          v-for="lv in levelFilters"
          :key="lv.value"
          :text="lv.label"
          :type="levelFilter === lv.value ? 'primary' : 'info'"
          size="mini"
          @click="handleLevelFilter(lv.value)"
        />
      </view>
      <u-table>
        <u-tr v-for="item in list" :key="item.userId" @click="goDetail(item.userId)">
          <u-td>{{ item.nickname || item.username }}</u-td>
          <u-td>{{ item.levelName || item.levelCode }}</u-td>
          <u-td>{{ item.growthValue || 0 }}</u-td>
          <u-td>
            <u-tag
              :text="item.status === 1 ? '正常' : '冻结'"
              :type="item.status === 1 ? 'success' : 'error'"
              size="mini"
            />
          </u-td>
          <u-td><SvgIcon name="arrow-right" /></u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="!loading && list.length === 0" text="暂无会员" />
      <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { MemberAPI } from "dehaze-sdk-js";

const levelFilters = [
  { label: "全部", value: "" },
  { label: "普通用户", value: "level_1" },
  { label: "白银会员", value: "level_2" },
  { label: "黄金会员", value: "level_3" },
];

const list = ref<any[]>([]);
const keyword = ref("");
const levelFilter = ref("");
const pageNum = ref(1);
const hasMore = ref(false);
const loading = ref(false);

const fetchList = async (reset = false) => {
  if (reset) {
    pageNum.value = 1;
    list.value = [];
  }
  loading.value = true;
  try {
    const params: any = {
      pageNum: pageNum.value,
      pageSize: 20,
    };
    if (keyword.value) params.keywords = keyword.value;
    if (levelFilter.value) params.levelCode = levelFilter.value;
    const res = await MemberAPI.getPage(params);
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
const handleLevelFilter = (val: string) => {
  levelFilter.value = val;
  fetchList(true);
};
const loadMore = () => fetchList();
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
}
.filter-row {
  display: flex;
  gap: 12rpx;
  flex-wrap: wrap;
  margin-bottom: 20rpx;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
