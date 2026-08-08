<template>
  <PageLayout level="L2" title="权限分配">
    <view class="page-body">
      <view class="menu-tree">
        <view v-for="menu in menuTree" :key="menu.id" class="tree-node">
          <view class="tree-item" @click="toggleMenu(menu)">
            <u-checkbox
              :checked="checkedIds.includes(menu.id)"
              @click.stop="togglePerm(menu.id)"
            />
            <text class="tree-label">{{ menu.name }}</text>
            <SvgIcon
              v-if="menu.children?.length"
              :name="
                expandedIds.includes(menu.id) ? 'arrow-down' : 'arrow-right'
              "
              size="16"
            />
          </view>
          <view
            v-if="expandedIds.includes(menu.id) && menu.children"
            class="tree-children"
          >
            <view
              v-for="child in menu.children"
              :key="child.id"
              class="tree-item child"
              @click="togglePerm(child.id)"
            >
              <u-checkbox :checked="checkedIds.includes(child.id)" />
              <text class="tree-label">{{ child.name }}</text>
            </view>
          </view>
        </view>
      </view>
      <view class="btn-area">
        <u-button type="primary" @click="handleSave" :loading="saving"
          >保存权限</u-button
        >
      </view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { RoleAPI, MenuAPI } from "dehaze-sdk-js";

const roleId = ref(0);
const menuTree = ref<any[]>([]);
const checkedIds = ref<number[]>([]);
const expandedIds = ref<number[]>([]);
const saving = ref(false);

onLoad((options: any) => {
  roleId.value = +(options?.id || 0);
  fetchMenuTree();
  fetchRolePerms();
});

const fetchMenuTree = async () => {
  try {
    menuTree.value = await MenuAPI.getList({});
  } catch {}
};
const fetchRolePerms = async () => {
  try {
    checkedIds.value = await RoleAPI.getRoleMenuIds(roleId.value);
  } catch {}
};
const toggleMenu = (menu: any) => {
  const idx = expandedIds.value.indexOf(menu.id);
  if (idx > -1) expandedIds.value.splice(idx, 1);
  else expandedIds.value.push(menu.id);
};
const togglePerm = (id: number) => {
  const idx = checkedIds.value.indexOf(id);
  if (idx > -1) checkedIds.value.splice(idx, 1);
  else checkedIds.value.push(id);
};
const handleSave = async () => {
  saving.value = true;
  try {
    await RoleAPI.updateRoleMenus(roleId.value, checkedIds.value);
    uni.showToast({ title: "保存成功", icon: "success" });
    setTimeout(() => uni.navigateBack(), 500);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.menu-tree {
  margin-bottom: 40rpx;
}
.tree-node {
  margin-bottom: 4rpx;
}
.tree-item {
  display: flex;
  align-items: center;
  gap: 12rpx;
  padding: 16rpx;
  border-bottom: 1rpx solid $color-border;
}
.tree-item.child {
  padding-left: 60rpx;
  background: $color-bg-secondary;
}
.tree-label {
  flex: 1;
}
.tree-children {
  border-left: 2rpx solid $color-border;
  margin-left: 24rpx;
}
.btn-area {
  padding: 20rpx 0;
}
</style>
