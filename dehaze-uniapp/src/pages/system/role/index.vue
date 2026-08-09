<template>
  <PageLayout level="L2" title="角色管理">
    <view class="page-body">
      <view class="list-row" v-for="role in list" :key="role.id">
        <text class="cell" @click="goDetail(role.id)">{{ role.name }}</text>
        <text class="cell" @click="goDetail(role.id)">{{ role.code }}</text>
        <view class="cell" @click="goDetail(role.id)">
          <view
            class="tag"
            :class="role.status === 1 ? 'tag-success' : 'tag-danger'"
          >
            {{ role.status === 1 ? "启用" : "禁用" }}
          </view>
        </view>
        <view class="cell row-actions">
          <SvgIcon v-if="canEdit" name="lock" @click="goPermission(role.id)" />
          <SvgIcon v-if="canEdit" name="edit-pen" @click="goDetail(role.id)" />
          <SvgIcon
            v-if="canDelete"
            name="trash"
            color="#ef4444"
            @click="delRole(role.id)"
          />
        </view>
      </view>
      <view v-if="!loading && list.length === 0" class="empty-tip"
        >暂无角色</view
      >
    </view>
    <FabButton v-if="canAdd" @click="goDetail(0)">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import { RoleAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:role:add"));
const canEdit = computed(() => authStore.hasPerm("sys:role:edit"));
const canDelete = computed(() => authStore.hasPerm("sys:role:delete"));

const list = ref<any[]>([]);
const loading = ref(false);

const fetchList = async () => {
  loading.value = true;
  try {
    const res = await RoleAPI.getPage({ pageNum: 1, pageSize: 100 });
    list.value = res.list || [];
  } finally {
    loading.value = false;
  }
};
const goDetail = (id: number) =>
  uni.navigateTo({ url: `/pages/system/role/detail?id=${id}` });
const goPermission = (id: number) =>
  uni.navigateTo({ url: `/pages/system/role/permission?id=${id}` });
const delRole = async (id: number) => {
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除该角色吗？",
  });
  if (!res.confirm) return;
  try {
    await RoleAPI.deleteByIds(String(id));
    uni.showToast({ title: "删除成功", icon: "success" });
    fetchList();
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

fetchList();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
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
</style>
