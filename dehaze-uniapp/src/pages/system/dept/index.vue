<template>
  <PageLayout level="L2" title="部门管理">
    <view class="page-body">
      <view v-for="dept in deptTree" :key="dept.id" class="tree-node">
        <view class="tree-item" @click="toggleExpand(dept)">
          <text class="tree-label">{{ dept.name }}</text>
          <view class="tree-actions" @click.stop>
            <SvgIcon v-if="canEdit" name="edit-pen" @click="editDept(dept)" />
            <SvgIcon
              v-if="canDelete"
              name="trash"
              color="#ef4444"
              @click="delDept(dept.id)"
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
              <SvgIcon
                v-if="canEdit"
                name="edit-pen"
                @click="editDept(child)"
              />
              <SvgIcon
                v-if="canDelete"
                name="trash"
                color="#ef4444"
                @click="delDept(child.id)"
              />
            </view>
          </view>
        </view>
      </view>
      <view v-if="deptTree.length === 0" class="empty-tip">暂无部门</view>
    </view>
    <FabButton v-if="canAdd" @click="editDept(null)">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
    <Popup :show="showForm" mode="center" round @close="showForm = false">
      <view class="popup-body">
        <view class="popup-title">{{ form.id ? "编辑部门" : "新增部门" }}</view>
        <view class="form-row">
          <text class="form-label">名称</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="部门名称"
          />
        </view>
        <view class="form-row">
          <text class="form-label">上级部门</text>
          <input
            class="form-input"
            type="number"
            v-model.number="form.parentId"
            placeholder="上级部门ID（0为顶级）"
          />
        </view>
        <view class="form-row">
          <text class="form-label">排序</text>
          <input
            class="form-input"
            type="number"
            v-model.number="form.sort"
            placeholder="排序"
          />
        </view>
        <view class="form-row">
          <text class="form-label">状态</text>
          <switch
            :checked="form.status === 1"
            @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
          />
        </view>
        <button class="btn btn-primary" :loading="saving" @click="handleSave">
          保存
        </button>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import Popup from "@/components/common/Popup.vue";
import { DeptAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:dept:add"));
const canEdit = computed(() => authStore.hasPerm("sys:dept:edit"));
const canDelete = computed(() => authStore.hasPerm("sys:dept:delete"));

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
.popup-body {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: $font-lg;
  font-weight: bold;
  margin-bottom: 20rpx;
}
.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.form-label {
  width: 180rpx;
  flex-shrink: 0;
  color: $color-text-primary;
}
.form-input {
  flex: 1;
  font-size: $font-sm;
}
.btn {
  width: 100%;
  margin-top: 40rpx;
  padding: 16rpx 0;
  border-radius: $radius-sm;
  font-size: $font-sm;

  &::after {
    border: none;
  }
}
.btn-primary {
  background: $color-primary;
  color: $color-white;
}
</style>
