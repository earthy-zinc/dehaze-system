<template>
  <PageLayout level="L2" title="用户管理">
    <view class="page-body">
      <view class="search-bar">
        <u-search
          v-model="keyword"
          placeholder="搜索用户名/手机号"
          @search="handleSearch"
          @clear="handleSearch"
        />
      </view>
      <u-table>
        <u-tr v-for="user in list" :key="user.id">
          <u-td @click="goDetail(user.id)">{{ user.nickname }}</u-td>
          <u-td @click="goDetail(user.id)">@{{ user.username }}</u-td>
          <u-td @click="goDetail(user.id)">
            <u-tag
              :type="user.status === 1 ? 'success' : 'error'"
              :text="user.status === 1 ? '正常' : '禁用'"
              size="mini"
            />
          </u-td>
          <u-td>
            <view class="row-actions">
              <SvgIcon name="edit-pen" @click="goDetail(user.id)" />
              <SvgIcon
                name="trash"
                @click="delUser(user.id)"
                color="$color-error"
              />
            </view>
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="!loading && list.length === 0" text="暂无用户" />
      <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
    </view>
    <view class="fab-btn" @click="goDetail(0)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { UserAPI } from "dehaze-sdk-js";

const list = ref<any[]>([]);
const keyword = ref("");
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
    const res = await UserAPI.getPage({
      pageNum: pageNum.value,
      pageSize: 20,
      keywords: keyword.value || undefined,
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
const goDetail = (id: number) =>
  uni.navigateTo({ url: `/pages/system/user/detail?id=${id}` });

const delUser = async (id: number) => {
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
.fab-btn {
  position: fixed;
  right: 40rpx;
  bottom: 100rpx;
  width: 96rpx;
  height: 96rpx;
  border-radius: 50%;
  background: $color-primary;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.2);
  z-index: 99;
}
.row-actions {
  display: flex;
  gap: 16rpx;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
