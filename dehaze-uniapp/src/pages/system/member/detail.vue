<template>
  <PageLayout level="L2" title="会员详情">
    <view class="page-body">
      <view class="info-card">
        <view class="info-row"
          ><text class="label">昵称</text
          ><text>{{ member.nickname }}</text></view
        >
        <view class="info-row"
          ><text class="label">用户名</text
          ><text>{{ member.username }}</text></view
        >
        <view class="info-row"
          ><text class="label">等级</text
          ><text>{{ member.levelName || member.levelCode }}</text></view
        >
        <view class="info-row"
          ><text class="label">成长值</text
          ><text>{{ member.growthValue }}</text></view
        >
        <view class="info-row"
          ><text class="label">进度</text
          ><text>{{ member.progressPercent }}%</text></view
        >
        <view class="info-row">
          <text class="label">状态</text>
          <u-tag
            :text="member.status === 1 ? '正常' : '冻结'"
            :type="member.status === 1 ? 'success' : 'error'"
            size="mini"
          />
          <u-switch :checked="member.status === 1" @change="toggleStatus" />
        </view>
      </view>
      <view class="section-title">成长日志</view>
      <u-table>
        <u-tr v-for="log in growthLogs" :key="log.id">
          <u-td>{{ log.reason || "成长值变动" }}</u-td>
          <u-td>{{ log.changeValue > 0 ? "+" + log.changeValue : log.changeValue }}</u-td>
          <u-td>{{ log.createTime }}</u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="growthLogs.length === 0" text="暂无成长日志" />
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, reactive } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import { MemberAPI } from "dehaze-sdk-js";

const id = ref(0);
const member = reactive<any>({});
const growthLogs = ref<any[]>([]);

onLoad((options: any) => {
  id.value = +(options?.id || 0);
  fetchDetail();
  fetchGrowthLogs();
});

const fetchDetail = async () => {
  try {
    const res = await MemberAPI.getDetail(id.value);
    Object.assign(member, res);
  } catch {}
};
const fetchGrowthLogs = async () => {
  try {
    const res = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 50 });
    growthLogs.value = res.list || [];
  } catch {}
};
const toggleStatus = async (val: boolean) => {
  try {
    await MemberAPI.updateStatus(id.value, { status: (val ? 1 : 0) as any });
    member.status = val ? 1 : 0;
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.info-card {
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx;
  margin-bottom: 30rpx;
}
.info-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.info-row:last-child {
  border-bottom: none;
}
.label {
  color: $color-text-secondary;
}
.section-title {
  font-size: 30rpx;
  font-weight: bold;
  padding: 20rpx 0;
}
</style>
