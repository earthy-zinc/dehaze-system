<template>
  <PageLayout level="L2" title="角色管理">
    <view class="page-body">
      <u-table>
        <u-tr v-for="role in list" :key="role.id">
          <u-td @click="goDetail(role.id)">{{ role.name }}</u-td>
          <u-td @click="goDetail(role.id)">{{ role.code }}</u-td>
          <u-td @click="goDetail(role.id)">
            <u-tag
              :text="role.status === 1 ? '启用' : '禁用'"
              :type="role.status === 1 ? 'success' : 'error'"
              size="mini"
            />
          </u-td>
          <u-td>
            <view class="row-actions">
              <SvgIcon name="lock" @click="goPermission(role.id)" />
              <SvgIcon name="edit-pen" @click="goDetail(role.id)" />
              <SvgIcon
                name="trash"
                @click="delRole(role.id)"
                color="$color-error"
              />
            </view>
          </u-td>
        </u-tr>
      </u-table>
      <u-empty v-if="!loading && list.length === 0" text="暂无角色" />
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
import { RoleAPI } from "dehaze-sdk-js";

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
.row-actions {
  display: flex;
  gap: 16rpx;
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
</style>
