<template>
  <PageLayout level="L2" title="部门管理">
    <view class="page-body">
      <view v-for="dept in deptTree" :key="dept.id" class="tree-node">
        <view class="tree-item" @click="toggleExpand(dept)">
          <text class="tree-label">{{ dept.name }}</text>
          <view class="tree-actions" @click.stop>
            <SvgIcon name="edit-pen" @click="editDept(dept)" />
            <SvgIcon
              name="trash"
              @click="delDept(dept.id)"
              color="$color-error"
            />
          </view>
          <SvgIcon
            v-if="dept.children?.length"
            :name="expandedIds.includes(dept.id) ? 'arrow-down' : 'arrow-right'"
            size="16"
          />
        </view>
        <view
          v-if="expandedIds.includes(dept.id) && dept.children"
          class="tree-children"
        >
          <view
            v-for="child in dept.children"
            :key="child.id"
            class="tree-item child"
          >
            <text class="tree-label">{{ child.name }}</text>
            <view class="tree-actions">
              <SvgIcon name="edit-pen" @click="editDept(child)" />
              <SvgIcon
                name="trash"
                @click="delDept(child.id)"
                color="$color-error"
              />
            </view>
          </view>
        </view>
      </view>
      <u-empty v-if="deptTree.length === 0" text="暂无部门" />
    </view>
    <view class="fab-btn" @click="editDept(null)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
    <u-popup :show="showForm" @close="showForm = false" round>
      <view class="popup-content">
        <view class="popup-title">{{ form.id ? "编辑部门" : "新增部门" }}</view>
        <u-form :model="form">
          <u-form-item label="名称"
            ><u-input v-model="form.name" placeholder="部门名称"
          /></u-form-item>
          <u-form-item label="上级部门">
            <u-input
              v-model.number="form.parentId"
              type="number"
              placeholder="上级部门ID（0为顶级）"
            />
          </u-form-item>
          <u-form-item label="排序"
            ><u-input
              v-model.number="form.sort"
              type="number"
              placeholder="排序"
          /></u-form-item>
          <u-form-item label="状态">
            <u-switch
              :checked="form.status === 1"
              @change="(val: boolean) => (form.status = val ? 1 : 0)"
            />
          </u-form-item>
        </u-form>
        <u-button type="primary" @click="handleSave" :loading="saving"
          >保存</u-button
        >
      </view>
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { DeptAPI } from "dehaze-sdk-js";

const deptTree = ref<any[]>([]);
const expandedIds = ref<number[]>([]);
const showForm = ref(false);
const form = ref<any>({ name: "", parentId: 0, sort: 0, status: 1 });
const saving = ref(false);

const fetchTree = async () => {
  try {
    deptTree.value = await DeptAPI.getList();
  } catch {}
};
const toggleExpand = (dept: any) => {
  const idx = expandedIds.value.indexOf(dept.id);
  if (idx > -1) expandedIds.value.splice(idx, 1);
  else expandedIds.value.push(dept.id);
};
const editDept = (dept: any) => {
  if (dept) form.value = { ...dept };
  else form.value = { name: "", parentId: 0, sort: 0, status: 1 };
  showForm.value = true;
};
const handleSave = async () => {
  saving.value = true;
  try {
    if (form.value.id) await DeptAPI.update(form.value.id, form.value);
    else await DeptAPI.add(form.value);
    showForm.value = false;
    fetchTree();
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};
const delDept = async (id: number) => {
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除该部门吗？",
  });
  if (!res.confirm) return;
  try {
    await DeptAPI.deleteByIds(String(id));
    fetchTree();
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

fetchTree();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
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
.tree-actions {
  display: flex;
  gap: 16rpx;
}
.tree-children {
  border-left: 2rpx solid $color-border;
  margin-left: 24rpx;
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
.popup-content {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: 32rpx;
  font-weight: bold;
  margin-bottom: 20rpx;
}
</style>
