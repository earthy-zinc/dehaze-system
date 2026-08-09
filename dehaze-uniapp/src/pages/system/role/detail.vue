<template>
  <PageLayout level="L2" :title="isEdit ? '编辑角色' : '新增角色'">
    <view class="page-body">
      <view class="form-row">
        <text class="form-label"><text class="required">*</text>角色名称</text>
        <input
          class="form-input"
          v-model="form.name"
          placeholder="请输入角色名称"
        />
      </view>
      <view class="form-row">
        <text class="form-label"><text class="required">*</text>角色编码</text>
        <input
          class="form-input"
          v-model="form.code"
          placeholder="请输入角色编码"
        />
      </view>
      <view class="form-row">
        <text class="form-label">排序</text>
        <input
          class="form-input"
          type="number"
          v-model.number="form.sort"
          placeholder="请输入排序"
        />
      </view>
      <view v-if="isEdit" class="form-row">
        <text class="form-label">状态</text>
        <switch
          :checked="form.status === 1"
          @change="(e: any) => (form.status = e.detail.value ? 1 : 0)"
        />
      </view>
      <view class="btn-area">
        <button
          v-if="isEdit ? canEdit : canAdd"
          class="btn btn-primary"
          :loading="submitting"
          @click="handleSubmit"
        >
          保存
        </button>
        <button
          v-if="isEdit && canEdit"
          class="btn btn-warning"
          @click="goPermission"
        >
          权限分配
        </button>
      </view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import { RoleAPI } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:role:add"));
const canEdit = computed(() => authStore.hasPerm("sys:role:edit"));

const id = ref(0);
const isEdit = computed(() => id.value > 0);
const form = ref<any>({ name: "", code: "", sort: 0, status: 1 });
const submitting = ref(false);

onLoad((options: any) => {
  id.value = +(options?.id || 0);
  if (isEdit.value) fetchRole();
});

const fetchRole = async () => {
  try {
    const d = await RoleAPI.getFormData(id.value);
    form.value = {
      name: d.name,
      code: d.code,
      sort: d.sort || 0,
      status: d.status,
    };
  } catch {}
};
const handleSubmit = async () => {
  submitting.value = true;
  try {
    if (isEdit.value) await RoleAPI.update(id.value, form.value);
    else await RoleAPI.add(form.value);
    uni.showToast({ title: "保存成功", icon: "success" });
    setTimeout(() => uni.navigateBack(), 500);
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    submitting.value = false;
  }
};
const goPermission = () =>
  uni.navigateTo({ url: `/pages/system/role/permission?id=${id.value}` });
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
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
.required {
  color: $color-danger;
  margin-right: 4rpx;
}
.form-input {
  flex: 1;
  font-size: $font-sm;
}
.btn-area {
  display: flex;
  gap: 20rpx;
  margin-top: 40rpx;
}
.btn {
  padding: 8rpx 20rpx;
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
.btn-warning {
  background: $color-warning;
  color: $color-white;
}
</style>
